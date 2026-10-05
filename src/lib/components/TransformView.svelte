<script lang="ts">
	/**
	 * 变换过程动画：由上一个矩阵逐行算出当前矩阵，可串成一条链。
	 *
	 *   1) 只有一段：                 [输入] ──op──▶ [输出]
	 *   2) 两段逐行（`then`）：        [输入] ──op1──▶ [中间] ──op2──▶ [结果]
 *   3) 第二段是矩阵乘（`then.b`）：[输入] ──op1──▶ ┌────────┐
 *                                                │ [B]    │  ← B 画在右上
 *                                                │ [输出] │[C] ← 输出同时是左操作数
 *                                                └────────┘
 *   4) 链尾再加一段逐行（`tail`）：在 3) 的 C 右边接 `op3` 与 [D]——
 *      `[输入] ──op1──▶ [A] ──op2(B)──▶ [C] ──op3──▶ [D]`，C 因此只画一次
 *      （用于"输出再做一次逆 RoPE"这种收尾：`O_h` 旁边就是 `O_h′`）。
 *   另外 `preMask`：输入块先逐格掩码，再做逐行变换（掩码与变换共用一个块，只画一次）。
 *
 * 各段**按先后顺序播**（不并发），时间按各段"工作量"分配，所以每格耗时一致。
 */
import type { MaskView, MatRef, TransformView } from '$lib/core/types';
import { transformCellSize, transformPhaseMs } from '$lib/core/steps';
	import MaskGrid from './MaskView.svelte';
	import MatrixGrid from './MatrixGrid.svelte';
	import ValueRow from './ValueRow.svelte';
	import { realLabel } from './realShape';
	import Emph from './Emph.svelte';

	let {
		view,
		progress = 0,
		compact = false,
		cellSize: cellSizeProp,
		scanRows
	}: {
		view: TransformView;
		progress?: number;
		compact?: boolean;
		cellSize?: number;
		/** 外部驱动的扫描行块：高亮输出矩阵的这一段行（如 `-sape` 页源矩阵被窗口槽位打分扫描） */
		scanRows?: [number, number] | null;
	} = $props();

	const rows = $derived(view.input.data?.length ?? 0);
	const cols = $derived(view.input.data?.[0]?.length ?? 0);

	// ── 各段时长与时间边界（毫秒，已含每段下限） ──────────
	// 与页级时长（stepDurationMs）共用同一份计算，避免两边算法漂移
	const phaseWork = $derived(transformPhaseMs(view));
	const maskWork = $derived(phaseWork.mask);
	const rowWork = $derived(phaseWork.rows);
	const row2Work = $derived(phaseWork.rows2);
	const mmWork = $derived(phaseWork.mm);
	const tailWork = $derived(phaseWork.tail);
	const totalWork = $derived(Math.max(1, maskWork + rowWork + row2Work + mmWork + tailWork));
	const endMask = $derived(maskWork / totalWork);
	const endRow = $derived((maskWork + rowWork) / totalWork);
	const endRow2 = $derived((maskWork + rowWork + row2Work) / totalWork);
	const endMm = $derived((maskWork + rowWork + row2Work + mmWork) / totalWork);

	const maskP = $derived(maskWork ? Math.min(1, progress / endMask) : 1);
	const rowP = $derived(
		rowWork ? Math.min(1, Math.max(0, (progress - endMask) / (endRow - endMask))) : 1
	);
	/** 第二段逐行（`then` 无 `b`）：**接着第一段**才跑，所以 `X ──op1──▶ Y ──op2──▶ Z` 是先后算的 */
	const row2P = $derived(
		row2Work ? Math.min(1, Math.max(0, (progress - endRow) / (endRow2 - endRow))) : 1
	);
	const mmP = $derived(mmWork ? Math.min(1, Math.max(0, (progress - endRow2) / (endMm - endRow2))) : 1);
	const tailP = $derived(
		tailWork ? Math.min(1, Math.max(0, (progress - endMm) / (1 - endMm))) : 1
	);

	// ── 逐行段 ───────────────────────────────────────────
	const doneRows = $derived(Math.min(rows, Math.round(rowP * rows)));
	const currentRow = $derived(Math.min(Math.max(0, rows - 1), doneRows));
	/** 第二段逐行（`then` 无 `b`）已经算完的行数 */
	const thenRows = $derived(view.then && !view.then.b ? (view.then.result.data?.length ?? 0) : 0);
	const thenDoneRows = $derived(Math.min(thenRows, Math.round(row2P * thenRows)));
	/** 链尾逐行段（`tail`）已经转完的行数 */
	const tailDoneRows = $derived(Math.min(rows, Math.round(tailP * rows)));
	const currentTailRow = $derived(Math.min(Math.max(0, rows - 1), tailDoneRows));
	const reveal = (d: number[][] | undefined, n = doneRows) =>
		d ? d.map((row, i) => (i < n ? row : row.map(() => null))) : [];

	// ── 矩阵乘段 ─────────────────────────────────────────
	const mmRows = $derived(view.then?.result.shape[0] ?? 0);
	const mmCols = $derived(view.then?.result.shape[1] ?? 0);
	const mmTotal = $derived(Math.max(1, mmRows * mmCols));
	const mmDone = $derived(Math.round(mmP * mmTotal));
	const mmIdx = $derived(Math.min(mmTotal - 1, mmDone));
	const mmI = $derived(Math.floor(mmIdx / Math.max(1, mmCols)));
	const mmJ = $derived(mmIdx % Math.max(1, mmCols));
	const mmShown = $derived.by(() => {
		const d = view.then?.result.data;
		if (!d) return [];
		return d.map((row, i) => row.map((v, j) => (i * mmCols + j < mmDone ? v : null)));
	});
	const mmK = $derived(view.output.shape[1]);
	const aRow = $derived(view.output.data?.[mmI] ?? []);
	const bCol = $derived(view.then?.b?.data?.map((r) => r[mmJ]) ?? []);
	const mmSum = $derived(aRow.reduce((s, v, p) => s + v * (bCol[p] ?? 0), 0));

	/**
	 * 格子边长。默认 34（数字读得清）；**链尾还接了一段**（`tail`）时链上多出两块矩阵，
	 * 整行会超宽，于是按列数反推一个放得下的尺寸（下限 `MIN_TEXT_CELL`）。
	 *
	 * `cellSizeProp`：`StepPanel` 给标了 `fillWidth` 的行量出"占满可用宽度"的边长之后传下来，
	 * 优先用它；没有（SSR / 首帧 / 别的行）就退回上面的静态兜底。
	 */
	// `cellSizeProp`（fillWidth 量出的占满边长）优先；`view.cellSize` 是视图自己要求放大
	// （如 c128 的 4-5-6-7 步）；都没有（SSR / 首帧 / 别的行）退回静态兜底。
	const cellSize = $derived(cellSizeProp ?? view.cellSize ?? transformCellSize(cols, compact, !!view.tail));
	/** 每一列的宽度上限：**矩阵宽 + 1/4**（原则 ④）；最后一列不封顶、把剩下的全吃掉（原则 ⑤） */
	const cap = $derived((cols * cellSize * 5) / 4);
	const inRow = $derived(view.input.data?.[currentRow] ?? []);
	const outRow = $derived(view.output.data?.[currentRow] ?? []);
	const thenRow = $derived(view.then?.result.data?.[currentRow] ?? []);

	const fmt = (v: number | null | undefined) =>
		v === null || v === undefined ? '—' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2);

	/** 掩码段复用 MaskView（bare：不画它自己的表头/计数/图例，这里已经画了） */
	const maskView = $derived<MaskView | null>(
		view.preMask
			? {
					kind: 'mask',
					name: view.input.name,
					shape: view.input.shape,
					scores: view.input.data ?? [],
					mask: view.preMask,
					// 分来源段时把段表透传给 MaskGrid，由它按段逐行处理列区间
					...(view.preMaskParts ? { parts: view.preMaskParts } : {})
				}
			: null
	);

	// ── 掩码段的"当前来源段"（分段时进度条要显示正在掩哪一段） ──
	// MaskGrid 按"总单元 = 段数 × 行数"均分进度，所以段号 = floor(maskP × 段数)
	const maskSeg = $derived(
		view.preMaskParts?.length
			? Math.min(view.preMaskParts.length - 1, Math.floor(maskP * view.preMaskParts.length))
			: -1
	);
</script>

{#snippet head(ref: MatRef, accent = false)}
	<!-- data-handoff 让 core/handoff.ts 在步骤切换时把这个矩阵"飞"过来 -->
	<div class="head" data-handoff={ref.handoffId} data-handoff-name={ref.name}>
		<span class="nm" class:accent>{ref.name}</span>
		<span class="sz">[{ref.shape.join(' × ')}]</span>
	</div>
{/snippet}

{#snippet anno(ref: MatRef)}
	<!-- 注解（真实尺寸 / 来源）放在矩阵**上面**、紧挨着名字。行数不同会顶歪矩阵顶边，
	     所以 `StepPanel` 会给这一页的 `.head` / `.real` / `.src` **统一高度**。 -->
	<div class="real">{realLabel(ref) ?? '\u00a0'}</div>
	<div class="src"><Emph text={ref.label ?? '\u00a0'} /></div>
{/snippet}

{#snippet progBar()}
	<div class="prog">
		{#if mmWork > 0 && mmP > 0}
			已算出 <b>{mmDone}</b> / {mmTotal} 个元素
		{:else}
			已算 <b>{doneRows}</b> / {rows} 行
		{/if}
	</div>
{/snippet}

<!-- 同 `MatmulView`：并排（`compact`）时不设变量（兜底 `auto` = 原样），独占整行才摊 -->
<div class="tf" class:fit={view.fillWidth} style={compact ? null : `--cap:${cap}px;--last:1fr`}>
	<!-- 第二段是矩阵乘时整体排成 4 列 2 行：B 在右上、输出块（=左操作数 A）在左下、C 在右下 -->
	<div class="row" class:chained={!!view.then} class:with-b={!!view.then?.b} class:has-tail={!!view.tail} class:no-input={!!view.hideInput}>
		<!-- 来源参考条：放在**第一行**（与 V 同行的空位），列左边界对齐输入矩阵的 `colFrom` 列
		     （如 V4 融合注意力的 top-k 选择矩阵对准 S 的压缩列）——不占第二行的高度，
		     S 与右侧矩阵的对齐不受影响；absolute 定位也不参与轨道尺寸（S 列不会因此变宽） -->
		{#if view.topRef}
			<div class="side top-ref">
				<div class="ref-top side" style={`left:${view.topRef.colFrom * cellSize}px`}>
					<div class="head">
						<span class="nm">{view.topRef.view.matrixName ?? view.topRef.view.name}</span>
						<span class="sz">[{view.topRef.view.shape.join(' × ')}]</span>
					</div>
					<div class="real">{'\u00a0'}</div>
					<div class="src"><Emph text={view.topRef.view.label ?? '\u00a0'} /></div>
					<MaskGrid bare view={view.topRef.view} progress={1} {compact} {cellSize} />
				</div>
			</div>
		{/if}
		{#if view.hideInput}
			<!-- 输入块不画（`hideInput`）：行首放一个"算子名 + 箭头"的引头，链从输出开始——
			     输入就是左边那块矩阵（如 `-compress-pool` 吞并 `-compress-kv` 时，v̄ 由 pool 的结果画） -->
			<div class="mid lead">
				<div class="op">{view.op}</div>
				<div class="ar">──▶</div>
			</div>
		{:else}
			<!-- ① 输入块（可选先做掩码） -->
			<div class="side first">
				{@render head(view.input)}
				{#if maskView}
					<!-- 掩码块也要走同样的"注解在上面"结构：少了这两行，它的矩阵会比旁边的
					     块高两行，对不齐 -->
					{@render anno(view.input)}
					<MaskGrid bare view={maskView} progress={maskP} {compact} {cellSize} />
				<div class="prog">
					{#if maskP < 1}
						{#if view.preMaskParts?.length}
							<span class="now"
								>掩码中 · 段 {maskSeg + 1}/{view.preMaskParts.length}：{view.preMaskParts[maskSeg].label}</span
							>
						{:else}
							<span class="now"
								>掩码中 · 已处理 {Math.round(maskP * rows * cols)} / {rows * cols}</span
							>
						{/if}
					{:else}
						<span class="done">
							掩码完成 · 屏蔽 {view.preMask?.flat().filter((m) => !m).length} 个
						</span>
					{/if}
				</div>
				{:else}
					{@render anno(view.input)}
					<MatrixGrid
						data={view.input.data ?? []}
						{cellSize}
						highlightRow={currentRow}
						highlightCols={view.highlightCols}
					/>
				{/if}
			</div>

			<div class="mid">
				<div class="op">{view.op}</div>
				<div class="ar">──▶</div>
				<div class="hint">逐行</div>
			</div>
		{/if}

		<!-- ② 输出块（第二段是矩阵乘时，它就是这次矩阵乘的左操作数） -->
		<div class="side result">
			{@render head(view.output, true)}
			{@render anno(view.output)}
			<MatrixGrid
				data={reveal(view.output.data)}
				{cellSize}
				highlightRow={mmWork > 0 && mmP > 0 ? mmI : currentRow}
				scanRows={scanRows}
				bands={view.output.bands}
				highlightCols={view.tail && !view.then && tailP > 0
					? view.tail.highlightCols
					: view.highlightCols}
			/>
			{#if !view.then}{@render progBar()}{/if}
		</div>

		{#if view.then?.b}
			<!-- ③ B 在右上 -->
			<div class="side top-right">
				{@render head(view.then.b)}
				{@render anno(view.then.b)}
				<MatrixGrid
					data={view.then.b.data ?? []}
					{cellSize}
					highlightCol={mmP > 0 ? mmJ : undefined}
					bands={view.then.b.bands}
				/>
			</div>

			<!-- ④ C 在右下 -->
			<div class="side final">
				{@render head(view.then.result, true)}
				{@render anno(view.then.result)}
				<MatrixGrid
					data={mmShown}
					{cellSize}
					highlightRow={mmP > 0 ? mmI : undefined}
					highlightCell={mmP > 0 && mmDone < mmTotal ? [mmI, mmJ] : null}
					highlightCols={tailP > 0 ? view.tail?.highlightCols : undefined}
				/>
				{@render progBar()}
			</div>
		{:else if view.then}
			<div class="mid">
				<div class="op">{view.then.op}</div>
				<div class="ar">──▶</div>
				<div class="hint">逐行</div>
			</div>
			<div class="side final">
				{@render head(view.then.result, true)}
				{@render anno(view.then.result)}
				<MatrixGrid
					data={reveal(view.then.result.data, thenDoneRows)}
					{cellSize}
					highlightRow={thenDoneRows}
					highlightCols={tailP > 0 ? view.tail?.highlightCols : undefined}
				/>
				<div class="prog">
					{#if row2P < 1}
						<span class="now">已算 <b>{thenDoneRows}</b> / {thenRows} 行</span>
					{:else}
						<span class="done">已算完 {thenRows} 行</span>
					{/if}
				</div>
			</div>
		{/if}

		<!--
			链尾再一段逐行（`tail`，如 RoPE / 逆 RoPE）：上一块已经是它的输入，所以上一块只画一次。
			"框出来的那几列"同时落在**上一块**和**这一块**上，而且**只在这一段计算时才出现**
			（`tailP > 0`）——先看到整块矩阵，算到这一步才把那几维框出来。
		-->
		{#if view.tail}
			<div class="mid tail-mid">
				<div class="op">{view.tail.op}</div>
				<div class="ar">──▶</div>
				<div class="hint">逐行</div>
			</div>
			<div class="side tail-final">
				{@render head(view.tail.result, true)}
				{@render anno(view.tail.result)}
				<MatrixGrid
					data={reveal(view.tail.result.data, tailDoneRows)}
					{cellSize}
					highlightRow={tailP > 0 ? currentTailRow : undefined}
					highlightCols={tailP > 0 ? view.tail.highlightCols : undefined}
				/>
				<div class="prog">
					{#if tailP < 1}
						<span class="now">已转 <b>{tailDoneRows}</b> / {rows} 行</span>
					{:else}
						<span class="done">已转完 {rows} 行</span>
					{/if}
				</div>
			</div>
		{/if}
	</div>

	<div class="detail">
		{#if mmWork > 0 && mmP > 0}
			<!-- 矩阵乘段：给出当前元素的计算过程 -->
			<div class="eq">
				<b>{view.then?.result.name}<span class="idx">[{mmI}]</span>[{mmJ}]</b>
				<span class="eqop">=</span>
				<span class="sigma">Σ<sub>p=0…{mmK - 1}</sub></span>
				{view.output.name}<span class="idx">[{mmI}]</span>[p]
				<span class="eqop">·</span>
				{view.then?.b?.name}[p][{mmJ}]
			</div>
			<div class="terms">
				{#each aRow as v, p (p)}
					<span class="term">
						<span class="tv">{fmt(v)}</span>
						<span class="tx">×</span>
						<span class="tv">{fmt(bCol[p])}</span>
					</span>
					{#if p < aRow.length - 1}<span class="plus">+</span>{/if}
				{/each}
				<span class="eqop">=</span>
				<b class="sum">{fmt(mmSum)}</b>
			</div>
		{:else}
			<div class="detail-title">第 {currentRow} 行：</div>
			<ValueRow title="输入" data={inRow} />
			<ValueRow title={`${view.op} 之后`} data={outRow} />
			{#if view.then}
				<ValueRow title={`${view.then.op} 之后`} data={thenRow} />
			{/if}
			{#if view.tail}
				<ValueRow
					title={`${view.tail.op} 之后`}
					data={view.tail.result.data?.[currentTailRow] ?? []}
				/>
			{/if}
		{/if}
	</div>
</div>

<style>
	.tf {
		display: flex;
		flex-direction: column;
		gap: 0.7rem;
	}
	/*
	 * 列宽：装得下内容之后还能长到 `--cap`（= 矩阵宽 + 1/4，原则 ④：矩阵之间的间距变大、
	 * 注解跟着变宽）；**最后一列** `1fr` 把剩下的全吃掉（原则 ⑤：注解一直延伸到行右边界，
	 * 且天然不超出容器）。
	 *
	 * 下限是 `auto`（= min-content）而不是 `max-content`：并排（`compact`）时要能压缩，
	 * 否则整行顶出去；而这个视图的列内容就是矩阵本身（`.detail` 是 `width: 0; min-width: 100%`），
	 * 所以 `auto` 不会比 `max-content` 窄——`--cap` 只限制"长多宽"，压不窄内容
	 * （`ConcatView` / `SumView` / `LookupView` 里有 nowrap 说明文字，它们必须用 `max-content`）。
	 */
	.row {
		display: grid;
		grid-template-columns: minmax(auto, var(--cap, auto)) auto minmax(auto, var(--last, auto));
		align-items: start;
		justify-content: start;
		column-gap: 0.8rem;
		row-gap: 0.7rem;
		overflow-x: auto;
		padding-bottom: 0.2rem;
	}
	.row.chained {
		grid-template-columns:
			minmax(auto, var(--cap, auto)) auto minmax(auto, var(--cap, auto)) auto
			minmax(auto, var(--last, auto));
	}
	/* 只有链尾、没有第二段（如"归一化 ──▶ RoPE"）：5 列一行排开 */
	.row.has-tail:not(.with-b) {
		grid-template-columns:
			minmax(auto, var(--cap, auto)) auto minmax(auto, var(--cap, auto)) auto
			minmax(auto, var(--last, auto));
	}
	/*
	 * 输入块不画（`hideInput`）：链从输出开始——行首的"引头"（算子名 + 箭头）占第 1 列，
	 * 输出 / 中间箭头 / 最终结果各占一列。放在 `.chained` 之后（同特异性，后者生效）。
	 */
	.row.no-input {
		grid-template-columns: auto minmax(auto, var(--cap, auto)) auto minmax(auto, var(--last, auto));
	}
	.row.no-input .mid.lead {
		align-self: center;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.1rem;
	}
	/*
	 * 链式变换中间那一段（算子名 + 箭头）的宽度**就是**两块矩阵之间的距离：
	 * 列间隙和算子名字号各收一点，S 与 P_h 就贴近一些。
	 * 再往下压就得把 `softmax` 竖排或删掉了——那三个字本身还有 ~40px。
	 */
	.row.chained {
		column-gap: 0.6rem;
	}
	.row.chained .op {
		font-size: 0.64rem;
	}
	/* 第二段是矩阵乘：4 列 2 行，B 右上 / A 左下 / C 右下 */
	.row.with-b {
		grid-template-columns:
			minmax(auto, var(--cap, auto)) auto minmax(auto, var(--cap, auto))
			minmax(auto, var(--last, auto));
	}
	.row.with-b .side.first {
		grid-area: 2 / 1;
	}
	.row.with-b .mid {
		grid-area: 2 / 2;
	}
	.row.with-b .side.result {
		grid-area: 2 / 3;
	}
	.row.with-b .side.top-right {
		grid-area: 1 / 4;
	}
	.row.with-b .side.final {
		grid-area: 2 / 4;
	}
	/* 链尾再接一段（`tail`）：多出"中段箭头 + 末块"两列，排在 C 的右边 */
	.row.with-b.has-tail {
		grid-template-columns:
			minmax(auto, var(--cap, auto)) auto minmax(auto, var(--cap, auto))
			minmax(auto, var(--cap, auto)) auto minmax(auto, var(--last, auto));
	}
	.row.with-b.has-tail .mid.tail-mid {
		grid-area: 2 / 5;
	}
	.row.with-b.has-tail .side.tail-final {
		grid-area: 2 / 6;
	}
	.side {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		/* 所有块给同样的内边距——这样并排时矩阵顶边能对齐 */
		padding: 0.2rem 0.3rem;
	}
	.head {
		display: flex;
		align-items: baseline;
		gap: 0.35rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	/* 来源参考条：grid 项放在**第一行**（与 V 同行）、absolute 定位到输入矩阵的对应列段。
	   这样它既不占第二行的高度（S 与右侧矩阵的对齐不受影响）、也不参与轨道尺寸
	   （S 列不会因此变宽）；表头/注解走同款的 `.head`/`.real`/`.src`，measure ②
	   会把它们与 V 的注解统一高度 → 矩阵顶边和 V 精确对齐。 */
	.row.with-b .side.top-ref {
		grid-area: 1 / 1;
	}
	.side.top-ref {
		position: relative;
	}
	.ref-top {
		position: absolute;
		top: 0;
		width: max-content;
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		padding: 0.2rem 0.3rem;
	}
	.nm {
		font-size: 0.78rem;
		font-weight: 700;
		color: #334155;
	}
	/* 产物用靛蓝名字标识，**不加底色框** */
	.nm.accent {
		color: #3730a3;
	}
	.sz {
		font-size: 0.66rem;
		color: #4f46e5;
	}
	.real {
		font-size: 0.58rem;
		color: #94a3b8;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.src {
		font-size: 0.6rem;
		color: #059669;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.mid {
		align-self: center;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.1rem;
	}
	.op {
		font-size: 0.72rem;
		font-weight: 700;
		color: #7c3aed;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.ar {
		color: #cbd5e1;
		font-size: 0.8rem;
		letter-spacing: -0.15em;
	}
	.hint {
		font-size: 0.58rem;
		color: #94a3b8;
	}
	.prog {
		font-size: 0.6rem;
		color: #94a3b8;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		white-space: nowrap;
	}
	.prog b {
		color: #3730a3;
	}
	.now {
		color: #b45309;
	}
	.done {
		color: #059669;
	}
	.detail {
		display: flex;
		align-items: flex-start;
		gap: 1rem;
		flex-wrap: wrap;
		padding: 0.5rem 0.6rem;
		border-radius: 0.4rem;
		background: #fbfdff;
		border: 1px dashed #cbd5e1;
	}
	.detail-title {
		font-size: 0.68rem;
		color: #64748b;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		padding-top: 0.9rem;
	}
	.eq {
		display: flex;
		align-items: baseline;
		gap: 0.3rem;
		flex-wrap: wrap;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.74rem;
		color: #475569;
	}
	.eq b {
		color: #3730a3;
	}
	/*
	 * 索引与数值都**定宽**，否则这一行会随动画逐格抖动：
	 * 矩阵乘段每 60ms 换一格，`[15][7]` 的索引位数会变（9 → 10）、
	 * `1.76 × -0.39` 这种数值宽度也会变，右边的累加项就跟着一起左右跳。
	 * 定宽之后整行宽度只由形状决定，不再由"当前算到哪一格"决定。
	 */
	.idx {
		display: inline-block;
		min-width: 4ch;
		text-align: right;
	}
	.tv {
		display: inline-block;
		min-width: 3.2em;
		text-align: right;
	}
	.eqop {
		color: #94a3b8;
	}
	.sigma {
		color: #7c3aed;
	}
	.sigma sub {
		font-size: 0.6em;
	}
	.terms {
		display: flex;
		align-items: center;
		gap: 0.2rem;
		flex-wrap: wrap;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.68rem;
	}
	.term {
		display: inline-flex;
		align-items: center;
		gap: 0.15rem;
		padding: 0.1rem 0.3rem;
		border-radius: 0.2rem;
		background: #eef2ff;
		border: 1px solid #c7d2fe;
		color: #3730a3;
	}
	.tx {
		color: #94a3b8;
	}
	.plus {
		color: #cbd5e1;
	}
	.sum {
		color: #0f172a;
		font-size: 0.8rem;
		/* 同 `.tv`：和值也定宽，否则行尾会随数值位数左右跳 */
		display: inline-block;
		min-width: 3.6em;
		text-align: right;
	}
</style>
