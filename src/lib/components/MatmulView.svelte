<script lang="ts">
	/**
	 * "两个矩阵 → 一个矩阵"的 2×2 过程视图，管两种运算：
	 *
	 *   - `matmul`：`C[i][j] = Σ_p A[i][p]·B[p][j]`（取行 × 取列 × 逐项相加）
	 *   - `ewise`：`out[i][j] = a[i][j] ⊙ b[i][j]`（两个**同位置**的格子相乘，没有 Σ）
	 *
	 * 两者的排布与动画**完全一样**，所以共用一个组件：
	 *   (0,0) = 进度信息        (0,1) = B / b（右上）
	 *   (1,0) = A / a（左下）   (1,1) = C / out（右下）
	 * 读法：横向左操作数 → 结果，纵向上操作数 → 结果。
	 *
	 * 差别只在三处：格子边长的算法、高亮（矩阵乘是"一行 × 一列"，逐元素是**同一个格子**）、
	 * 以及算式框（矩阵乘是 `Σ_p`，逐元素只有一项）。
	 *
	 * 动画**匀速 + 本地可控**：每个步骤自带 ▶ 播放 / 暂停、速度选择、重置，
	 * 用户不碰本地控制时，仍然跟随总播放器的进度。
	 */
	import type { EwiseView, MatRef, MatmulView } from '$lib/core/types';
	import { refKey } from '$lib/core/overview';
	import { matmulCellSize } from '$lib/core/steps';
	import MatrixGrid from './MatrixGrid.svelte';
	import { realLabel } from './realShape';
	import Emph from './Emph.svelte';

	let {
		view,
		progress = 0,
		compact = false,
		cellSize: cellSizeProp
	}: {
		view: MatmulView | EwiseView;
		progress?: number;
		compact?: boolean;
		cellSize?: number;
	} = $props();

	/** 逐元素模式（`ewise`）：`out[i][j] = a[i][j] ⊙ b[i][j]`，没有 Σ */
	const ewise = $derived(view.kind === 'ewise');
	const opSym = $derived(view.kind === 'ewise' ? (view.op ?? '⊙') : '@');
	/**
	 * 逐格揭示的顺序。
	 *
	 * 逐元素视图可以要求**按列**（`revealOrder: 'col'`）：一整列算完再走下一列，
	 * 用在"结果要按列归约"的场合——右边那个 `v̄` 就是按列出数的（见 `StepPanel`/`RowView`）。
	 */
	/** 逐格揭示的顺序：`colFirst`（按列）之上再按**组**切——见下面 `cells` 的说明 */
	const colFirst = $derived(view.kind === 'ewise' && view.revealOrder === 'col');
	/** 逐组揭示：一组几行（0 = 不分组，沿用原来的"一整列"顺序） */
	const groupRows = $derived(view.kind === 'ewise' ? (view.groupRows ?? 0) : 0);

	const aData = $derived(view.a.data ?? []);
	const bData = $derived(view.b.data ?? []);
	const outData = $derived(view.out.data ?? []);

	const rows = $derived(outData.length);
	const cols = $derived(outData[0]?.length ?? 0);
	const k = $derived(aData[0]?.length ?? 0);
	const bCols = $derived(bData[0]?.length ?? 0);

	/**
	 * 输出格子的揭示序列。
	 *
	 * - 默认行优先；
	 * - `colFirst`（按列）：一整列算完再走下一列——"按列归约"的场合用；
	 * - 再叠上 `groupRows`（**逐组**）：一组一组算，组内仍按列。
	 *   8 个 token、一组 4 个 → 先算前 4 个（得到第 1 条）、再算后 4 个（第 2 条）。
	 */
	const cells = $derived.by(() => {
		const list: [number, number][] = [];
		if (colFirst) {
			if (groupRows > 0) {
				for (let g0 = 0; g0 < rows; g0 += groupRows)
					for (let j = 0; j < cols; j++)
						for (let i = g0; i < Math.min(g0 + groupRows, rows); i++) list.push([i, j]);
			} else {
				for (let j = 0; j < cols; j++) for (let i = 0; i < rows; i++) list.push([i, j]);
			}
		} else {
			for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) list.push([i, j]);
		}
		return list;
	});

	// ── 由进度推出"算到哪一格"（匀速） ────────────
	// 逐组揭示时**整组的一列一起出**（一次跳 `groupRows` 个格子）：`w ⊙ v` 那一步
	// 一组 4 个 token 的同一列是"一起算、加起来得到那一条"的，一格一格点会把整段播放
	// 拖得很慢也看不出组的概念。结果 `v̄` 的 `groupRows` 换算式
	// 正好是每 4 格出 1 格，和这里的整列 chunk 严格同步。
	const revealed = $derived(
		groupRows > 0
			? Math.min(cells.length, Math.round((progress * cells.length) / groupRows) * groupRows)
			: Math.min(cells.length, Math.round(progress * cells.length))
	);
	const currentIdx = $derived(Math.min(cells.length - 1, revealed));
	const current = $derived(cells[currentIdx] ?? [0, 0]);
	const ci = $derived(current[0]);
	const cj = $derived(current[1]);
	const showTerms = $derived(progress > 0.01);

	/**
	 * 逐组揭示时"**正在算的那一组**"的行区间——框跟着动画走（第 1 组算完就挪到第 2 组），
	 * 不能写死在某一组上。全算完（或没分组）时返回 `null`（那时没有"正在算"的东西，不框）。
	 */
	const activeGroupRows = $derived.by<[number, number] | null>(() => {
		if (groupRows <= 0 || revealed >= cells.length) return null;
		const g = Math.floor(ci / groupRows);
		return [g * groupRows, Math.min(rows - 1, g * groupRows + groupRows - 1)];
	});

	const outDisplay = $derived.by(() => {
		const d: (number | null)[][] = outData.map((row) => row.map(() => null));
		for (let n = 0; n < revealed && n < cells.length; n++) {
			const [i, j] = cells[n];
			d[i][j] = outData[i][j];
		}
		return d;
	});

	const aRow = $derived(aData[ci] ?? []);
	const bCol = $derived(bData.map((r) => r[cj]));
	/**
	 * 算式框里的项。
	 *
	 * 矩阵乘：`A[i][p] × B[p][j]`，p 跑遍收缩维；
	 * 逐元素：**只有一项** `a[i][j] × b[i][j]`（两个同位置的格子）。
	 */
	const products = $derived(
		ewise
			? [{ a: aRow[cj], b: bData[ci]?.[cj], prod: (aRow[cj] ?? 0) * (bData[ci]?.[cj] ?? 0) }]
			: aRow.map((v, p) => ({ a: v, b: bCol[p], prod: v * (bCol[p] ?? 0) }))
	);
	const sum = $derived(products.reduce((s, t) => s + t.prod, 0));

	// ── 逐元素的"当前乘式"明细 ──────────────────────────────
	/**
	 * 当前正在算的那一组一列的全部格子——明细框**跟着动画走**：计算到哪几个格子，
	 * 就显示哪几个格子。一次出 `groupRows` 个（`w ⊙ v` 里一组 4 个 token 的同一列是一起算的）。
	 * 全算完（`activeGroupRows` 为 null）时显示最后一组最后一列，明细框不空着。
	 */
	const activeChunkCells = $derived.by<[number, number][]>(() => {
		const grp = activeGroupRows;
		const g0 = grp ? grp[0] : rows - (rows % groupRows || groupRows);
		const g1 = grp ? grp[1] : Math.min(rows - 1, g0 + groupRows - 1);
		const j = grp ? cj : cols - 1;
		const list: [number, number][] = [];
		for (let i = g0; i <= g1; i++) list.push([i, j]);
		return list;
	});
	/** 这一组的 4 个乘式**按列加起来** = 结果 `v̄` 的那一格（"算出 v̄ 的过程"就在这） */
	const chunkSum = $derived(activeChunkCells.reduce((s, [i, j]) => s + aData[i][j] * bData[i][j], 0));
	/** 正在算的这一组在结果 `v̄` 里的行号（组序） */
	const chunkGroup = $derived(activeGroupRows ? activeGroupRows[0] / groupRows : rows / groupRows - 1);

	// 横向并排时（compact）把格子缩小，让 Q·Kᵀ → mask → P·V 能排进一行。
	// 公式与 verify 共用（`core/steps.ts`）——"能不能逐格显示数字"只有一份判定。
	//
	// 矩阵乘传的是 A/B/C 三块的**总列数**：它们左右排着，决定整行宽度的就是这三块之和。
	// 逐元素没有收缩维，2×2 的宽度 = `a` 的列数 + `max(b, out)` 的列数（B 在 C 上方）。
	//
	// `cellSizeProp`：`StepPanel` 给标了 `fillWidth` 的行量出"占满可用宽度"的边长之后传下来，
	// 优先用它；`view.cellSize`：视图自己要求放大（如 c128 的 4-5-6-7 步）；
	// 都没有（SSR / 首帧 / 别的行）就退回静态兜底。
	const cellSize = $derived(
		cellSizeProp ??
			view.cellSize ??
			(ewise
				? matmulCellSize(k + Math.max(bCols, cols), compact)
				: matmulCellSize(k + bCols + cols, compact))
	);

	/**
	 * 列的宽度上限：**矩阵宽 + 1/4**（用户原则 ④："增加现在左边矩阵的 1/4 的宽度"）。
	 *
	 * 这是**纯数据**（列数 × 格子边长），不看渲染结果——`max-content` 和 `1fr` 都由浏览器算，
	 * 所以不会出现"量 → 改 → 再量"的自激循环。
	 */
	const cap = $derived((k * cellSize * 5) / 4);

	const fmt = (v: number | null | undefined) =>
		v === null || v === undefined ? '—' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2);

	// ── 下方算式框的宽度上限 ─────────────────────────────
	// 算式 / 乘积累加项（.terms）的 max-content 宽能到 600px+，会把整行撑宽、
	// 把同行并排的另一个矩阵挤出可视区（`moe-experts` 的 E2 就被挤走过）。
	// 这里**量出** 2×2 网格的真实宽度（列宽受矩阵、名字、"真实尺寸"标签共同影响，
	// 静态算不准），把算式框限制在这个宽度内，长式子换行、与上方网格左右对齐。
	let cellXEl: HTMLElement | undefined = $state();
	let cellWEl: HTMLElement | undefined = $state();
	let gridW = $state(0);
	/** 逐元素时明细框要占满的整行宽度（从最近的 `.hrow` 量） */
	let rowW = $state(0);

	$effect(() => {
		if (!compact || !cellXEl || !cellWEl) return;
		// 列是 auto 尺寸、不随外层拉伸，所以"col1 左边界 → col2 右边界"就是网格宽度
		const measure = () => {
			gridW = cellWEl!.getBoundingClientRect().right - cellXEl!.getBoundingClientRect().left;
			// 逐元素：明细框占满这一整行（右侧空着，右侧已经没有其他的东西了，这个框的范围可以放大）
			const row = cellXEl!.closest('.hrow');
			if (row) rowW = row.getBoundingClientRect().width;
		};
		measure();
		const ro = new ResizeObserver(measure);
		ro.observe(cellXEl);
		ro.observe(cellWEl);
		return () => ro.disconnect();
	});

	const ratio = (f: number) => (f >= 10 ? String(Math.round(f)) : f.toFixed(1));
</script>

{#snippet matBlock(o: {
	ref: MatRef;
	data: (number | null)[][];
	accent?: boolean;
	hlRow?: number;
	hlCol?: number;
	hlCell?: [number, number] | null;
	/** 框出这几列（只有结果块用：如 `[v | s]` 里"打分那几列"） */
	hlCols?: number[];
	/** 框出这几行（只有结果块用：如"这条压缩条目看的窗口那几行"） */
	hlRows?: [number, number];
	/** 正在累加的那一列（整列加淡底）——逐元素视图按列揭示时用 */
	busyCol?: number;
	/** 淡底只盖这几行（逐组揭示时"正在算的那一组"） */
	busyRows?: [number, number];
})}
	<!-- data-handoff 让 core/handoff.ts 在步骤切换时把这个矩阵"飞"到下一步 -->
	<div
		class="mblock"
		data-handoff={o.ref.handoffId}
		data-handoff-name={o.ref.name}
	>
		<!--
			名字 → 注解（真实尺寸 / 来源）→ **矩阵**。注解放在矩阵**上面**、紧挨着名字：
			放下面时它离名字很远、又和下方的"图注 + 算式框"挤在一起，读不出是哪一块的。
			行数不同会把矩阵顶边顶歪——`StepPanel` 会给这一页的 `.head` / `.real` / `.src`
			**统一高度**（取全页最大那份），所以矩阵顶边照样对齐。
		-->
		<div class="head">
			<span class="mname" class:accent={o.accent}>{o.ref.name}</span>
			<span class="msize">[{o.ref.shape.join(' × ')}]</span>
		</div>
		<!-- 两行注解**恒定占位**：并排的块结构一致，矩阵顶边才对齐 -->
		<div class="real">{realLabel(o.ref) ?? '\u00a0'}</div>
		<div class="src"><Emph text={o.ref.label ?? '\u00a0'} /></div>
		<MatrixGrid
			data={o.data}
			{cellSize}
			highlightRow={o.hlRow}
			highlightCol={o.hlCol}
			highlightCell={o.hlCell}
			highlightCols={o.hlCols}
			highlightRows={o.hlRows}
			bands={o.ref.bands}
			busyCol={o.busyCol}
			busyRows={o.busyRows}
		/>
	</div>
{/snippet}

<div class="mm" class:fit={view.fillWidth}>
	<!-- 2×2 乘法表式布局：B / b 右上、A / a 左下、C / out 右下 -->
	<!--
		`compact`（和别的列并排，宽度由 `.hcol` 决定）时**不设变量**——`var(…, auto)` 兜底
		就是原来的 `auto auto`，一格都不变。只有独占整行（`!compact`）时才真的摊剩余宽度：
		并排时如果也摊，会把这一格的 max-content 撑大（并排的"内容宽度"正是它决定的）→
		顶出整行；`fillWidth` 那两页还靠内容宽度反推格子，撑大后格子会缩到下限。
	-->
	<div class="matrices" style={compact ? null : `--cap:${cap}px;--last:1fr`}>
		<div class="cell-info">
			<div class="counter">
				{#if compact}
					<b>{revealed}</b>/{cells.length} 格
				{:else}
					已算出 <b>{revealed}</b> / {cells.length} 个元素
				{/if}
			</div>
			<div class="now">
				正在算 <b>{view.out.name}[{ci}][{cj}]</b>
			</div>
			{#if !compact}
				<div class="legend">
					{#if ewise}
						<!-- 逐元素：两个操作数取的是**同一个格子**，不是"一行 × 一列" -->
						<i class="sw row"></i>{view.a.name}[{ci}][{cj}] × {view.b.name}[{ci}][{cj}]
					{:else}
						<i class="sw row"></i>A 的第 {ci} 行
						<i class="sw col"></i>B 的第 {cj} 列
					{/if}
				</div>
			{/if}
		</div>

		<div class="cell-w" bind:this={cellWEl}>
			{@render matBlock(
				ewise
					? { ref: view.b, data: bData, hlCell: current, hlRows: activeGroupRows ?? undefined }
					: { ref: view.b, data: bData, hlCol: cj }
			)}
		</div>

		<div class="cell-x" bind:this={cellXEl}>
			{@render matBlock(
				ewise
					? { ref: view.a, data: aData, hlCell: current, hlRows: activeGroupRows ?? undefined }
					: { ref: view.a, data: aData, hlRow: ci }
			)}
		</div>

		<div class="cell-q">
			{@render matBlock({
				ref: view.out,
				data: outDisplay,
				accent: true,
				hlCell: current,
				hlCols: view.highlightCols,
				// 逐组揭示时框**正在算的那一组**（不写死）；别的场合沿用视图自己的 `highlightRows`
				hlRows: activeGroupRows ?? (view.kind === 'ewise' ? view.highlightRows : undefined),
				busyCol: colFirst ? cj : undefined,
				// 淡底只盖"正在算的那一组"——不然另外几组也跟着变黄，像在加它们
				busyRows: activeGroupRows ?? undefined
			})}
		</div>
	</div>

	<!--
		等式放在矩阵**下面**（图注惯例）——放上面会把整块往下推，和并排的另一个视图错开。
	-->
	<div class="bar">
		<div class="caption">
			<b>{view.out.name}</b> = <b>{view.a.name}</b> {opSym} <b>{view.b.name}</b>
			{#if view.kind === 'matmul' && view.scaleNote}<span class="scale"
					>（再 × {view.scaleNote}）</span
				>{/if}
		</div>
	</div>

	<!--
		计算明细。
		逐元素（`ewise`）：显示**当前正在算的那一组一列**的乘式——一次 4 个、跟着动画走。
		宽不钳到矩阵网格：这一行右侧本来就是空的，明细框占满整行。
		矩阵乘：仍是当前元素 `Σ_p A[i][p]·B[p][j]` 的逐项明细（那是它的教学点，一格一格走）。
	-->
	<div class="calc" style={!ewise && compact && gridW ? `max-width:${gridW}px` : ewise && rowW ? `width:${Math.min(rowW, 340)}px` : null}>
		{#if ewise}
			<!-- 4 个乘式排 2×2：一格一列（`repeat(4, 1fr)`）时每列只有 ~120px，乘式（nowrap）
			     会溢出轨道压到邻居 -->
			<div
				class="bulk"
				style={`grid-template-columns:repeat(${Math.min(2, Math.max(1, activeChunkCells.length))}, minmax(0, 1fr))`}
			>
				{#each activeChunkCells as c (c[0] * cols + c[1])}
					{@const bi = c[0]}
					{@const bj = c[1]}
					<div class="pair">
						<span class="pv">{view.a.name}[{bi}][{bj}]</span>
						<span class="tx">×</span>
						<span class="pv">{view.b.name}[{bi}][{bj}]</span>
						<span class="eqop">=</span>
						<b class="psum">{fmt(aData[bi][bj] * bData[bi][bj])}</b>
					</div>
				{/each}
			</div>
			<!-- 这一组的 4 个乘式**按列加起来** = 右边结果 v̄ 的那一格——"算出 v̄"的过程就展示在这里 -->
			<div class="bulk-sum">
				<span class="sumcap">按列加起来</span>
				<span class="eqop">=</span>
				<b class="psum">{fmt(chunkSum)}</b>
				<span class="eqop">→</span>
				<b class="vbar">v̄[{chunkGroup}][{cj}]</b>
			</div>
		{:else}
			<div class="eq">
				<b>{view.out.name}[{ci}][{cj}]</b>
				<span class="eqop">=</span>
				<span class="sigma">Σ<sub>p=0…{k - 1}</sub></span>
				{view.a.name}[{ci}][p]
				<span class="eqop">·</span>
				{view.b.name}[p][{cj}]
			</div>
			{#if showTerms}
				<div class="terms">
					{#each products as t, p (p)}
						<span class="term">
							<span class="tv">{fmt(t.a)}</span>
							<span class="tx">×</span>
							<span class="tv">{fmt(t.b)}</span>
						</span>
						{#if p < products.length - 1}<span class="plus">+</span>{/if}
					{/each}
					<span class="eqop">=</span>
					<b class="sum">{fmt(sum)}</b>
				</div>
			{:else}
				<div class="hint">点上面的 ▶ 播放，逐格看每个数是怎么算出来的</div>
			{/if}
		{/if}
	</div>
</div>

<style>
	.mm {
		display: flex;
		flex-direction: column;
		gap: 0.8rem;
	}
	.bar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		flex-wrap: wrap;
	}
	.caption {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.85rem;
		color: #334155;
	}
	.caption b {
		color: #3730a3;
	}
	.caption .scale {
		color: #a21caf;
		font-size: 0.72rem;
	}

	/*
		2×2 坐标布局（乘法表式）：
		  (0,0) 进度信息     (0,1) B 权重（右上）
		  (1,0) A 输入（左下）(1,1) C 输出（右下）
		读法：横向 X → Q，纵向 W → Q。
	*/
	.matrices {
		display: grid;
		grid-template-columns: auto auto;
		gap: 0.9rem 1.4rem;
		justify-content: start;
		align-items: start;
		overflow-x: auto;
		padding-bottom: 0.2rem;
	}
	/*
	 * 把这一行**剩下的横向宽度**摊给两列（"每个矩阵一个方框宽度"）。
	 *
	 *   左列 `minmax(auto, --cap)`：装得下内容之后还能长到"矩阵宽 + 1/4"——
	 *     矩阵之间的间距因此变大、注解跟着变宽（原则 ④）；没有富余就停在内容宽。
	 *   右列 `minmax(auto, 1fr)`：把剩下的全吃掉，注解一直延伸到这一行右边界，
	 *     而且天然不会超出容器（原则 ⑤："注释可以往右侧延伸，但是不要超过屏幕"）。
	 *
	 * 分配是**浏览器**算的（`max-content` / `fr`），JS 只给一个纯数据的上限——
	 * 没有"量 → 改 → 再量"，所以不会自激。
	 *
	 * 下限用 `auto`（= min-content）而不是 `max-content`：这两列**必须能被压缩**，
	 * 否则并排（`compact`）时整行顶出去；而这个视图的列内容就是矩阵本身
	 * （`.calc` / `.bar` 是 `width: 0; min-width: 100%`，不参与列宽），
	 * 所以 `auto` 不会比 `max-content` 窄——`--cap` 也就只限制"长多宽"，压不窄内容。
	 * > `ConcatView` / `SumView` / `LookupView` 不一样：那边列里还有 nowrap 的说明文字
	 * > （如"已拼 8 / 8 列 · 拼接完成"）会比矩阵宽，所以它们的下限必须是 `max-content`。
	 */
	.matrices {
		grid-template-columns: minmax(auto, var(--cap, auto)) minmax(auto, var(--last, auto));
	}
	.cell-info {
		grid-area: 1 / 1;
		align-self: center;
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.68rem;
		color: #64748b;
		max-width: 8rem;
	}
	.cell-w {
		grid-area: 1 / 2;
	}
	.cell-x {
		grid-area: 2 / 1;
	}
	.cell-q {
		grid-area: 2 / 2;
	}
	.counter b {
		color: #3730a3;
		font-size: 0.82rem;
	}
	.now b {
		color: #0f172a;
	}
	.legend {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		flex-wrap: wrap;
		color: #94a3b8;
		font-size: 0.62rem;
	}
	.sw {
		display: inline-block;
		width: 0.6rem;
		height: 0.6rem;
		border-radius: 2px;
		margin-left: 0.3rem;
	}
	.sw.row {
		background: #f59e0b;
	}
	.sw.col {
		background: #0ea5e9;
	}
	.head {
		display: flex;
		align-items: baseline;
		gap: 0.35rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.mname {
		font-size: 0.82rem;
		font-weight: 700;
		color: #334155;
	}
	.mname.accent {
		color: #3730a3;
	}
	.msize {
		font-size: 0.68rem;
		color: #4f46e5;
	}
	.mblock {
		display: flex;
		flex-direction: column;
		/* 和 `TransformView` 的 `.side` 一样的内边距——差 3px 并排时顶边就不齐 */
		padding: 0.2rem 0.3rem;
	}
	/* 来源标签行：固定高度，保证同一行的三个矩阵网格起点一致 */
	/* 来源注解：和 `TransformView` 的 `.src` 同款（绿字），这样并排时两边结构一致 */
	.src {
		font-size: 0.6rem;
		color: #059669;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		margin-top: 0.15rem;
	}
	.real {
		font-size: 0.6rem;
		color: #94a3b8;
		margin-bottom: 0.2rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}

	.calc {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
		padding: 0.6rem 0.7rem;
		border-radius: 0.4rem;
		background: #fbfdff;
		border: 1px dashed #cbd5e1;
	}
	/* 逐元素的"当前乘式"明细：显示正在算的那一组一列（一次 4 个），
	   每个占一个 `1fr` 列、内容居中——明细框占满整行（右侧空着），4 个乘式均匀铺开 */
	.bulk {
		display: grid;
		gap: 0.4rem 0.7rem;
		align-items: center;
		width: 100%;
	}
	.pair {
		display: flex;
		align-items: baseline;
		justify-content: center;
		gap: 0.28rem;
		padding: 0.2rem 0.4rem;
		border-radius: 0.3rem;
		background: #eef2ff;
		border: 1px solid #a5b4fc;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.6rem;
		color: #475569;
		white-space: nowrap;
	}
	.pv {
		color: #64748b;
	}
	.tx {
		color: #94a3b8;
	}
	.psum {
		color: #3730a3;
		font-weight: 700;
	}
	.bulk-sum {
		display: flex;
		align-items: baseline;
		gap: 0.35rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.7rem;
		color: #475569;
		padding-top: 0.15rem;
		border-top: 1px dashed #e2e8f0;
	}
	.sumcap {
		color: #64748b;
	}
	.vbar {
		color: #3730a3;
		font-weight: 700;
		background: #eef2ff;
		padding: 0.05rem 0.35rem;
		border-radius: 0.25rem;
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
	.hint {
		font-size: 0.7rem;
		color: #94a3b8;
	}
</style>
