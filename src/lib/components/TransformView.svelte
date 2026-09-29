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
	 *   另外 `preMask`：输入块先逐格掩码，再做逐行变换（掩码与变换共用一个块，只画一次）。
	 *
	 * 各段**按先后顺序播**（不并发），时间按各段"工作量"分配，所以每格耗时一致。
	 */
	import type { MaskView, MatRef, TransformView } from '$lib/core/types';
	import { transformPhaseMs } from '$lib/core/steps';
	import MaskGrid from './MaskView.svelte';
	import MatrixGrid from './MatrixGrid.svelte';
	import ValueRow from './ValueRow.svelte';

	let {
		view,
		progress = 0,
		compact = false
	}: { view: TransformView; progress?: number; compact?: boolean } = $props();

	const rows = $derived(view.input.data?.length ?? 0);
	const cols = $derived(view.input.data?.[0]?.length ?? 0);

	// ── 各段时长与时间边界（毫秒，已含每段下限） ──────────
	// 与页级时长（stepDurationMs）共用同一份计算，避免两边算法漂移
	const phaseWork = $derived(transformPhaseMs(view));
	const maskWork = $derived(phaseWork.mask);
	const rowWork = $derived(phaseWork.rows);
	const mmWork = $derived(phaseWork.mm);
	const totalWork = $derived(Math.max(1, maskWork + rowWork + mmWork));
	const endMask = $derived(maskWork / totalWork);
	const endRow = $derived((maskWork + rowWork) / totalWork);

	const maskP = $derived(maskWork ? Math.min(1, progress / endMask) : 1);
	const rowP = $derived(
		rowWork ? Math.min(1, Math.max(0, (progress - endMask) / (endRow - endMask))) : 1
	);
	const mmP = $derived(mmWork ? Math.min(1, Math.max(0, (progress - endRow) / (1 - endRow))) : 1);

	// ── 逐行段 ───────────────────────────────────────────
	const doneRows = $derived(Math.min(rows, Math.round(rowP * rows)));
	const currentRow = $derived(Math.min(Math.max(0, rows - 1), doneRows));
	const reveal = (d: number[][] | undefined) =>
		d ? d.map((row, i) => (i < doneRows ? row : row.map(() => null))) : [];

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

	const cellSize = $derived(compact ? 24 : 34);
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
					mask: view.preMask
				}
			: null
	);
</script>

{#snippet head(ref: MatRef, accent = false)}
	<!-- data-handoff 让 core/handoff.ts 在步骤切换时把这个矩阵"飞"过来 -->
	<div class="head" data-handoff={ref.handoffId} data-handoff-name={ref.name}>
		<span class="nm" class:accent>{ref.name}</span>
		<span class="sz">[{ref.shape.join(' × ')}]</span>
	</div>
	<!-- 来源标签行**恒定占位**（没有标签时塞一个不换行空格）。
	     条件渲染会让同一行里"有标签的块"比"没标签的块"低一行，矩阵顶边就对不齐了。 -->
	<div class="src">{ref.label ?? '\u00a0'}</div>
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

<div class="tf">
	<!-- 第二段是矩阵乘时整体排成 4 列 2 行：B 在右上、输出块（=左操作数 A）在左下、C 在右下 -->
	<div class="row" class:chained={!!view.then} class:with-b={!!view.then?.b}>
		<!-- ① 输入块（可选先做掩码） -->
		<div class="side first">
			{@render head(view.input)}
			{#if maskView}
				<MaskGrid bare view={maskView} progress={maskP} {compact} />
				<div class="prog">
					{#if maskP < 1}
						<span class="now">掩码中 · 已处理 {Math.round(maskP * rows * cols)} / {rows * cols}</span>
					{:else}
						<span class="done">
							掩码完成 · 屏蔽 {view.preMask?.flat().filter((m) => !m).length} 个
						</span>
					{/if}
				</div>
			{:else}
				<MatrixGrid data={view.input.data ?? []} {cellSize} highlightRow={currentRow} />
			{/if}
		</div>

		<div class="mid">
			<div class="op">{view.op}</div>
			<div class="ar">──▶</div>
			<div class="hint">逐行</div>
		</div>

		<!-- ② 输出块（第二段是矩阵乘时，它就是这次矩阵乘的左操作数） -->
		<div class="side result">
			{@render head(view.output, true)}
			<MatrixGrid
				data={reveal(view.output.data)}
				{cellSize}
				highlightRow={mmWork > 0 && mmP > 0 ? mmI : currentRow}
			/>
			{#if !view.then}{@render progBar()}{/if}
		</div>

		{#if view.then?.b}
			<!-- ③ B 在右上 -->
			<div class="side top-right">
				{@render head(view.then.b)}
				<MatrixGrid
					data={view.then.b.data ?? []}
					{cellSize}
					highlightCol={mmP > 0 ? mmJ : undefined}
				/>
			</div>

			<!-- ④ C 在右下 -->
			<div class="side final">
				{@render head(view.then.result, true)}
				<MatrixGrid
					data={mmShown}
					{cellSize}
					highlightRow={mmP > 0 ? mmI : undefined}
					highlightCell={mmP > 0 && mmDone < mmTotal ? [mmI, mmJ] : null}
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
				<MatrixGrid data={reveal(view.then.result.data)} {cellSize} highlightRow={currentRow} />
				{@render progBar()}
			</div>
		{/if}
	</div>

	<div class="detail">
		{#if mmWork > 0 && mmP > 0}
			<!-- 矩阵乘段：给出当前元素的计算过程 -->
			<div class="eq">
				<b>{view.then?.result.name}[{mmI}][{mmJ}]</b>
				<span class="eqop">=</span>
				<span class="sigma">Σ<sub>p=0…{mmK - 1}</sub></span>
				{view.output.name}[{mmI}][p]
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
		{/if}
	</div>
</div>

<style>
	.tf {
		display: flex;
		flex-direction: column;
		gap: 0.7rem;
	}
	.row {
		display: grid;
		grid-template-columns: auto auto auto;
		align-items: start;
		justify-content: start;
		column-gap: 0.8rem;
		row-gap: 0.7rem;
		overflow-x: auto;
		padding-bottom: 0.2rem;
	}
	.row.chained {
		grid-template-columns: auto auto auto auto auto;
	}
	/* 第二段是矩阵乘：4 列 2 行，B 右上 / A 左下 / C 右下 */
	.row.with-b {
		grid-template-columns: auto auto auto auto;
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
	}
</style>
