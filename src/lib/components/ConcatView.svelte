<script lang="ts">
	/**
	 * 拼接过程动画。
	 *
	 * 把各块（如各头的输出）并排摆开，然后**逐列揭示**拼接结果：
	 * 进度走过某个块的列区间时，那几列才出现在结果里，同时该块高亮。
	 * 于是"结果的哪几列来自哪个头"一目了然。
	 *
	 * 可选 `then`：拼接结果再当左操作数做一次矩阵乘（`Concat · W_O → O`），
	 * 于是 `Concat` 只画一次、两段按先后顺序播。
	 */
	import type { ConcatView } from '$lib/core/types';
	import { concatPhaseMs } from '$lib/core/steps';
	import { maxAbsOf } from './color';
	import MatrixGrid from './MatrixGrid.svelte';
	import { realLabel } from './realShape';
	import Emph from './Emph.svelte';

	let { view, progress = 0 }: { view: ConcatView; progress?: number } = $props();

	// ── 两段的工作量与时间边界（与页级时长共用一份计算） ──
	const phaseWork = $derived(concatPhaseMs(view));
	const concatWork = $derived(phaseWork.cols);
	const mmWork = $derived(phaseWork.mm);
	const totalWork = $derived(Math.max(1, concatWork + mmWork));
	const concatP = $derived(concatWork ? Math.min(1, progress / (concatWork / totalWork)) : 1);
	const mmP = $derived(
		mmWork ? Math.min(1, Math.max(0, (progress - concatWork / totalWork) / (mmWork / totalWork))) : 1
	);

	/** 每个块在结果里占的列区间 [from, to) */
	const spans = $derived.by(() => {
		const out: [number, number][] = [];
		let c = 0;
		for (const p of view.parts) {
			const w = p.shape[1];
			out.push([c, c + w]);
			c += w;
		}
		return out;
	});

	const totalCols = $derived(spans.length ? spans[spans.length - 1][1] : 0);
	const revealedCols = $derived(Math.round(concatP * totalCols));
	/** 当前正在拼的那一块 */
	const activePart = $derived(
		Math.max(
			0,
			spans.findIndex(([a, b]) => revealedCols >= a && revealedCols < b)
		)
	);

	/** 结果矩阵：只显示已拼接的列 */
	const resultData = $derived.by(() => {
		const data = view.result.data;
		if (!data) return [];
		return data.map((row) => row.map((v, j) => (j < revealedCols ? v : null)));
	});

	// ── 矩阵乘段 ────────────────────────────────────────
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
	const mmK = $derived(view.result.shape[1]);
	const aRow = $derived(view.result.data?.[mmI] ?? []);
	const bCol = $derived(view.then?.b?.data?.map((r) => r[mmJ]) ?? []);
	const mmSum = $derived(aRow.reduce((s, v, p) => s + v * (bCol[p] ?? 0), 0));

	// 带矩阵乘段时格子缩小一号，否则一行放不下（"横向放不下就把矩阵缩小一些"）
	const cellSize = $derived(view.then ? 24 : 34);
	/** 列的宽度上限：**最宽那块矩阵的宽 + 1/4**（原则 ④）；最后一列不封顶、把剩下的全吃掉（原则 ⑤） */
	const cap = $derived(
		(Math.max(...view.parts.map((p) => p.shape[1] ?? 0), view.result.shape[1] ?? 0) * cellSize * 5) / 4
	);
	const maxAbs = $derived(view.result.data ? maxAbsOf(view.result.data) : 1);
	const fmt = (v: number | null | undefined) =>
		v === null || v === undefined ? '—' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2);
</script>

<div class="concat" style={`--cap:${cap}px;--last:1fr`}>
	<!--
		带矩阵乘段时排成 4 列 2 行（与 TransformView 的 with-b 同款）：
		  (1,4) B=W_O
		  (2,1) 各块 (2,2) = (2,3) 拼接结果=A (2,4) C=O
		这样第 2 行里的"各块 / = / Concat / O"都在同一条基线上，B 在 C 上方。
	-->
	<div class="row" class:with-mm={!!view.then}>
		<div class="parts">
			{#each view.parts as p, i (p.name)}
				{#if i > 0}<span class="op">+</span>{/if}
				<div class="part" class:active={i === activePart}>
					<div class="head">
						<span class="nm">{p.name}</span>
						<span class="sz">[{p.shape.join(' × ')}]</span>
					</div>
					<div class="real">{realLabel(p) ?? '\u00a0'}</div>
					<div class="src"><Emph text={p.label ?? '\u00a0'} /></div>
					<MatrixGrid data={p.data ?? []} {cellSize} />
					<div class="cols">
						占结果的第 {spans[i][0]}–{spans[i][1] - 1} 列
					</div>
				</div>
			{/each}
		</div>

		<span class="op eq">=</span>

		<!-- 拼接结果：第二段是矩阵乘时，它就是这次矩阵乘的左操作数 -->
		<div class="part result">
			<div class="head">
				<span class="nm">{view.result.name}</span>
				<span class="sz">[{view.result.shape.join(' × ')}]</span>
			</div>
			<div class="real">{realLabel(view.result) ?? '\u00a0'}</div>
			<div class="src"><Emph text={view.result.label ?? '\u00a0'} /></div>
			<MatrixGrid
				data={resultData}
				{cellSize}
				highlightRow={view.then && mmP > 0 ? mmI : undefined}
			/>
			<div class="cols">
				已拼 <b>{revealedCols}</b> / {totalCols} 列
				{#if revealedCols < totalCols}
					<span class="now">· 正在拼「{view.parts[activePart]?.name}」</span>
				{:else}
					<span class="done">· 拼接完成</span>
				{/if}
			</div>
		</div>

		{#if view.then}
			<!-- B 在右上、C 在右下（与 MatmulView 的"乘法表"一致） -->
			<div class="part b">
				<div class="head">
					<span class="nm">{view.then.b.name}</span>
					<span class="sz">[{view.then.b.shape.join(' × ')}]</span>
				</div>
				<div class="real">{realLabel(view.then.b) ?? '\u00a0'}</div>
				<div class="src"><Emph text={view.then.b.label ?? '\u00a0'} /></div>
				<MatrixGrid
					data={view.then.b.data ?? []}
					{cellSize}
					highlightCol={mmP > 0 ? mmJ : undefined}
				/>
			</div>

			<div class="part c">
				<div class="head">
					<span class="nm accent">{view.then.result.name}</span>
					<span class="sz">[{view.then.result.shape.join(' × ')}]</span>
				</div>
				<div class="real">{realLabel(view.then.result) ?? '\u00a0'}</div>
				<div class="src"><Emph text={view.then.result.label ?? '\u00a0'} /></div>
				<MatrixGrid
					data={mmShown}
					{cellSize}
					highlightRow={mmP > 0 ? mmI : undefined}
					highlightCell={mmP > 0 && mmDone < mmTotal ? [mmI, mmJ] : null}
				/>
				<div class="cols">
					已算出 <b>{mmDone}</b> / {mmTotal} 个元素
				</div>
			</div>
		{/if}
	</div>

	{#if view.then && mmP > 0}
		<div class="detail">
			<div class="eq">
				<b>{view.then.result.name}[{mmI}][{mmJ}]</b>
				<span class="eqop">=</span>
				<span class="sigma">Σ<sub>p=0…{mmK - 1}</sub></span>
				{view.result.name}[{mmI}][p]
				<span class="eqop">·</span>
				{view.then.b.name}[p][{mmJ}]
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
		</div>
	{/if}
</div>

<style>
	.concat {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
	}
	/* 列宽：下限是 `max-content`（**装得下内容**），上限才是 `--cap`（矩阵宽 + 1/4）——
	   `--cap` 只限制"有富余时长多宽"，绝不会把本来就宽过它的那一列**压窄**。
	   `max-content` 作下限时"growth limit < base"会退化成 base，天然不压内容。
	   最后一列 `1fr` 把剩下的全吃掉。 */
	.row {
		display: grid;
		grid-template-columns:
			minmax(max-content, var(--cap, auto)) auto
			minmax(max-content, var(--last, 1fr));
		align-items: center;
		justify-content: start;
		column-gap: 0.7rem;
		row-gap: 0.7rem;
		overflow-x: auto;
		padding-bottom: 0.2rem;
	}
	/* 带矩阵乘段：4 列 2 行 —— B 在右上、A 在左下、C 在右下 */
	.row.with-mm {
		grid-template-columns:
			minmax(max-content, var(--cap, auto)) auto minmax(max-content, var(--cap, auto))
			minmax(max-content, var(--last, 1fr));
		align-items: start;
	}
	.row.with-mm .parts {
		grid-area: 2 / 1;
	}
	.row.with-mm .op.eq {
		grid-area: 2 / 2;
		align-self: center;
	}
	.row.with-mm .part.result {
		grid-area: 2 / 3;
	}
	.row.with-mm .part.b {
		grid-area: 1 / 4;
	}
	.row.with-mm .part.c {
		grid-area: 2 / 4;
	}
	/* 各块 + "+" 组成一组（同一行内居中，所以 "+" 与矩阵对齐） */
	.parts {
		display: flex;
		align-items: center;
		gap: 0.7rem;
	}
	.op {
		font-size: 0.9rem;
		color: #94a3b8;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.op.eq {
		font-size: 1.1rem;
		color: #cbd5e1;
	}
	.part {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		padding: 0.3rem 0.4rem;
		border-radius: 0.4rem;
		border: 1px solid transparent;
		transition:
			border-color 0.25s,
			background 0.25s;
	}
	.part.active {
		border-color: #f59e0b;
		background: #fffbeb;
	}
	/* 结果块不加底色/边框，只把名字标成靛蓝 */
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
	.cols {
		font-size: 0.6rem;
		color: #94a3b8;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		white-space: nowrap;
	}
	.cols b {
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
