<script lang="ts">
	/**
	 * 矩阵乘过程视图。
	 *
	 * 布局按乘法表式排成 2×2：
	 *   (0,0) = 进度信息        (0,1) = B 权重（右上）
	 *   (1,0) = A 输入（左下）  (1,1) = C 输出（右下）
	 * 读法：横向 X → Q，纵向 W → Q；完整等式写在顶部标题栏。
	 *
	 * 动画改为**匀速 + 本地可控**：每个步骤自带 ▶ 播放 / 暂停、速度选择、重置，
	 * 用户不碰本地控制时，仍然跟随总播放器的进度。
	 */
	import type { MatRef, MatmulView } from '$lib/core/types';
	import { refKey } from '$lib/core/overview';
	import { matmulCellSize } from '$lib/core/steps';
	import MatrixGrid from './MatrixGrid.svelte';
	import { realLabel } from './realShape';
	import Emph from './Emph.svelte';

	let {
		view,
		progress = 0,
		compact = false
	}: { view: MatmulView; progress?: number; compact?: boolean } = $props();

	const aData = $derived(view.a.data ?? []);
	const bData = $derived(view.b.data ?? []);
	const outData = $derived(view.out.data ?? []);

	const rows = $derived(outData.length);
	const cols = $derived(outData[0]?.length ?? 0);
	const k = $derived(aData[0]?.length ?? 0);

	/** 输出格子的行优先序列 */
	const cells = $derived.by(() => {
		const list: [number, number][] = [];
		for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) list.push([i, j]);
		return list;
	});


	// ── 由进度推出"算到哪一格"（匀速） ────────────
	const revealed = $derived(Math.min(cells.length, Math.round(progress * cells.length)));
	const currentIdx = $derived(Math.min(cells.length - 1, revealed));
	const current = $derived(cells[currentIdx] ?? [0, 0]);
	const ci = $derived(current[0]);
	const cj = $derived(current[1]);
	const showTerms = $derived(progress > 0.01);

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
	const products = $derived(
		aRow.map((v, p) => ({ a: v, b: bCol[p], prod: v * (bCol[p] ?? 0) }))
	);
	const sum = $derived(products.reduce((s, t) => s + t.prod, 0));

	// 横向并排时（compact）把格子缩小，让 Q·Kᵀ → mask → P·V 能排进一行。
	// 公式与 verify 共用（`core/steps.ts`）——"能不能逐格显示数字"只有一份判定。
	const cellSize = $derived(matmulCellSize(Math.max(cols, k, 1), compact));

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

	$effect(() => {
		if (!compact || !cellXEl || !cellWEl) return;
		// 列是 auto 尺寸、不随外层拉伸，所以"col1 左边界 → col2 右边界"就是网格宽度
		const measure = () => {
			gridW = cellWEl!.getBoundingClientRect().right - cellXEl!.getBoundingClientRect().left;
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
})}
	<!-- data-handoff 让 core/handoff.ts 在步骤切换时把这个矩阵"飞"到下一步 -->
	<div
		class="mblock"
		data-handoff={o.ref.handoffId}
		data-handoff-name={o.ref.name}
	>
		<div class="head">
			<span class="mname" class:accent={o.accent}>{o.ref.name}</span>
			<span class="msize">[{o.ref.shape.join(' × ')}]</span>
		</div>
		<!-- 来源标签行**恒定占位**：只有被复用的矩阵才有标签，
		     若条件渲染会导致同一行里 A/B/C 的网格起点错开一行 -->
		<div class="badge-row">
			{#if o.ref.label}
				<span class="badge"><Emph text={o.ref.label} /></span>
			{/if}
		</div>
		<!-- "真实尺寸"这行同理：有些操作数没有 realShape，条件渲染一样会错开一行 -->
		<div class="real">{realLabel(o.ref) ?? '\u00a0'}</div>
		<MatrixGrid
			data={o.data}
			{cellSize}
			highlightRow={o.hlRow}
			highlightCol={o.hlCol}
			highlightCell={o.hlCell}
		/>
	</div>
{/snippet}

<div class="mm">
	<!-- 顶部：等式（播放控制统一放在页级，见 StepPanel） -->
	<div class="bar">
		<div class="caption">
			<b>{view.out.name}</b> = <b>{view.a.name}</b> @ <b>{view.b.name}</b>
			{#if view.scaleNote}<span class="scale">（再 × {view.scaleNote}）</span>{/if}
		</div>
	</div>

	<!-- 2×2 乘法表式布局 -->
	<div class="matrices">
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
					<i class="sw row"></i>A 的第 {ci} 行
					<i class="sw col"></i>B 的第 {cj} 列
				</div>
			{/if}
		</div>

		<div class="cell-w" bind:this={cellWEl}>
			{@render matBlock({ ref: view.b, data: bData, hlCol: cj })}
		</div>

		<div class="cell-x" bind:this={cellXEl}>
			{@render matBlock({ ref: view.a, data: aData, hlRow: ci })}
		</div>

		<div class="cell-q">
			{@render matBlock({
				ref: view.out,
				data: outDisplay,
				accent: true,
				hlCell: current
			})}
		</div>
	</div>

	<!-- 当前元素的计算过程（compact 下限制宽度，长式子换行，不挤走同行的其他矩阵） -->
	<div class="calc" style={compact && gridW ? `max-width:${gridW}px` : null}>
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
	}
	/* 来源标签行：固定高度，保证同一行的三个矩阵网格起点一致 */
	.badge-row {
		position: relative;
		height: 1.25rem;
	}
	/* 绝对定位 → 不参与宽度计算，且 left/right:0 让它与下方矩阵等宽、左右对齐 */
	.badge {
		position: absolute;
		left: 0;
		right: 0;
		top: 0;
		text-align: center;
		line-height: 1rem;
		padding: 0.05rem 0.4rem;
		border-radius: 999px;
		background: #ecfdf5;
		border: 1px solid #a7f3d0;
		color: #047857;
		font-size: 0.62rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
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
