<script lang="ts">
	/**
	 * 加权求和动画：若干同形矩阵**逐行相加**（可带逐行系数），可选再接一段追加相加。
	 *
	 *   [项1 × w1] ┐
	 *   [项2 × w2] ┼──Σ──▶ ┌ routed ┐
	 *              ┘       │ shared │──+──▶ [y]
	 *                      └────────┘
	 *
	 * 两段**按先后顺序播**（先算出 routed，再叠加 shared 得到 y）；
	 * `routed` 与追加项纵向堆叠在一起——它们都是第二个 `+` 的输入，
	 * 摆成一列既贴近 `+`，也不会把 `routed` 画两遍。
	 */
	import type { SumView } from '$lib/core/types';
	import { sumPhaseMs } from '$lib/core/steps';
	import MatrixGrid from './MatrixGrid.svelte';
	import ValueRow from './ValueRow.svelte';
	import { realLabel } from './realShape';
	import Emph from './Emph.svelte';

	let {
		view,
		progress = 0,
		compact = false
	}: { view: SumView; progress?: number; compact?: boolean } = $props();

	const rows = $derived(view.terms[0]?.data?.length ?? 0);

	// ── 两段的时长与时间边界（与页级时长共用一份计算） ──
	const phaseMs = $derived(sumPhaseMs(view));
	const sumMs = $derived(phaseMs.sum);
	const addMs = $derived(phaseMs.add);
	const totalMs = $derived(Math.max(1, sumMs + addMs));
	const sumP = $derived(sumMs ? Math.min(1, progress / (sumMs / totalMs)) : 1);
	const addP = $derived(
		addMs ? Math.min(1, Math.max(0, (progress - sumMs / totalMs) / (addMs / totalMs))) : 1
	);

	/** 第一段：逐行算出 routed */
	const doneRows = $derived(Math.min(rows, Math.round(sumP * rows)));
	const currentRow = $derived(Math.min(Math.max(0, rows - 1), doneRows));
	/** 第二段：逐行加上 shared 得到 y */
	const doneRows2 = $derived(Math.min(rows, Math.round(addP * rows)));
	const currentRow2 = $derived(Math.min(Math.max(0, rows - 1), doneRows2));

	const reveal = (d: number[][] | undefined, done: number) =>
		d ? d.map((row, i) => (i < done ? row : row.map(() => null))) : [];

	// 链式相加要塞 4 列矩阵（项 / routed / shared / y），格子得小一号才排得进一行
	const cellSize = $derived(compact || view.then ? 24 : 34);
	/** 列的宽度上限：**最宽那块矩阵的宽 + 1/4**（原则 ④）；最后一列不封顶（原则 ⑤） */
	const cap = $derived(
		(Math.max(...view.terms.map((t) => t.shape[1] ?? 0), view.result.shape[1] ?? 0) * cellSize * 5) / 4
	);
	const hasWeights = $derived(!!view.weights);

	/** 第 j 项在第 t 行的系数 */
	const wOf = (j: number, t: number) => view.weights?.[j]?.[t] ?? 1;
	const fmtW = (v: number) => v.toFixed(2);

	const outRow = $derived(view.result.data?.[currentRow] ?? []);
	const finalRow = $derived(view.then?.result.data?.[currentRow2] ?? []);
</script>

{#snippet progBar()}
	<div class="prog">
		{#if view.then && addP > 0}
			已算 <b>{doneRows2}</b> / {rows} 行（加共享专家）
		{:else}
			已算 <b>{doneRows}</b> / {rows} 行
			{#if doneRows < rows}<span class="now">· 正在算第 {currentRow} 行</span
				>{:else}<span class="done">· 加权和完成</span>{/if}
		{/if}
	</div>
{/snippet}

<div class="sm" style={`--cap:${cap}px;--last:1fr`}>
	<div class="row" class:chained={!!view.then}>
		<div class="terms">
			{#each view.terms as term, j (term.name)}
				{#if j > 0}
				<!--
					两项是**各算各的**（不同专家各自的输出），这里只是把它们加起来——
					中间画一条浅色虚线表示"断开"。
				-->
					<span class="tsep" aria-hidden="true"></span>
				{/if}
				<div class="side">
					<div class="head">
						<span class="nm">{term.name}</span>
						<span class="sz">[{term.shape.join(' × ')}]</span>
						{#if hasWeights}<span class="w">× {fmtW(wOf(j, currentRow))}</span>{/if}
					</div>
					<div class="real">{realLabel(term) ?? '\u00a0'}</div>
					<div class="src"><Emph text={term.label ?? '\u00a0'} /></div>
					<MatrixGrid data={term.data ?? []} {cellSize} highlightRow={currentRow} />
				</div>
			{/each}
		</div>

		<div class="mid">
			<div class="op">Σ</div>
			<div class="ar">──▶</div>
			<div class="hint">逐行相加</div>
		</div>

		<!-- 第一段的结果（跨 1~2 行、居中在 O_1/O_2 之间） -->
		<div class="side result">
				<div class="head">
					<span class="nm">{view.result.name}</span>
					<span class="sz">[{view.result.shape.join(' × ')}]</span>
				</div>
				<div class="real">{realLabel(view.result) ?? '\u00a0'}</div>
				<div class="src"><Emph text={view.result.label ?? '\u00a0'} /></div>
				<MatrixGrid
					data={reveal(view.result.data, doneRows)}
					{cellSize}
					highlightRow={addP > 0 ? currentRow2 : currentRow}
				/>
				{#if !view.then}{@render progBar()}{/if}
			</div>

		<!-- 第二段追加的项：跟在 routed 下面（第 3 行） -->
		{#if view.then}
			{#each view.then.terms as term (term.name)}
				<div class="side extra">
					<div class="head">
						<span class="nm">{term.name}</span>
						<span class="sz">[{term.shape.join(' × ')}]</span>
					</div>
					<div class="real">{realLabel(term) ?? '\u00a0'}</div>
					<div class="src"><Emph text={term.label ?? '\u00a0'} /></div>
					<MatrixGrid data={term.data ?? []} {cellSize} highlightRow={currentRow2} />
				</div>
			{/each}
		{/if}

		{#if view.then}
			<div class="mid2">
				<div class="op">+</div>
				<div class="ar">──▶</div>
			</div>

			<div class="side final">
				<div class="head">
					<span class="nm">{view.then.result.name}</span>
					<span class="sz">[{view.then.result.shape.join(' × ')}]</span>
				</div>
				<div class="real">{realLabel(view.then.result) ?? '\u00a0'}</div>
				<div class="src"><Emph text={view.then.result.label ?? '\u00a0'} /></div>
				<MatrixGrid
					data={reveal(view.then.result.data, doneRows2)}
					{cellSize}
					highlightRow={currentRow2}
				/>
				{@render progBar()}
			</div>
		{/if}
	</div>

	<div class="detail">
		{#if view.then && addP > 0}
			<div class="detail-title">第 {currentRow2} 行：</div>
			<ValueRow title={view.result.name} data={view.result.data?.[currentRow2] ?? []} />
			{#each view.then.terms as term (term.name)}
				<ValueRow title={`+ ${term.name}`} data={term.data?.[currentRow2] ?? []} />
			{/each}
			<ValueRow title={`= ${view.then.result.name}`} data={finalRow} />
		{:else}
			<div class="detail-title">第 {currentRow} 行：</div>
			{#each view.terms as term, j (term.name)}
				<ValueRow
					title={hasWeights ? `${term.name} × ${fmtW(wOf(j, currentRow))}` : term.name}
					data={term.data?.[currentRow] ?? []}
				/>
			{/each}
			<ValueRow title={`= ${view.result.name}`} data={outRow} />
		{/if}
	</div>
</div>

<style>
	.sm {
		display: flex;
		flex-direction: column;
		gap: 0.7rem;
	}
	/* 列宽：下限是 `max-content`（装得下内容），上限才是 `--cap`（矩阵宽 + 1/4，原则 ④）——
	   `--cap` 只限制"有富余时长多宽"，不会把宽过它的那一列压窄（见 `ConcatView` 里的说明）。
	   最后一列 `1fr` 把剩下的全吃掉（原则 ⑤）。 */
	.row {
		display: grid;
		grid-template-columns:
			minmax(max-content, var(--cap, auto)) auto
			minmax(max-content, var(--last, 1fr));
		align-items: center;
		justify-content: start;
		column-gap: 0.8rem;
		overflow-x: auto;
		padding-bottom: 0.2rem;
	}
	/*
		链式相加排成 3 行 5 列，让"位置"表达运算关系：
		  ┌─────────┬─────┬─────────┬─────┬─────┐
		  │  O_1    │  Σ  │ routed  │  +  │  y  │   ← routed 跨 1~2 行、垂直居中在 O_1/O_2 之间
		  │  O_2    │     │         │     │     │
		  ├─────────┼─────┼─────────┼─────┼─────┤
		  │         │     │ shared  │     │     │   ← 追加项跟在 routed 下面，紧贴第二个 "+"
		  └─────────┴─────┴─────────┴─────┴─────┘
		Σ 与 routed 都跨 1~2 行并居中 → 箭头正好指着"两项之和"；
		+ 与 y 跨 1~3 行并居中 → 落在 routed 与 shared 之间。
	*/
	.row.chained {
		grid-template-columns:
			minmax(max-content, var(--cap, auto)) auto minmax(max-content, var(--cap, auto)) auto
			minmax(max-content, var(--last, 1fr));
		grid-template-rows: auto auto auto;
		row-gap: 0.7rem;
	}
	.row.chained .terms {
		grid-area: 1 / 1 / 3 / 2;
	}
	.row.chained .mid {
		grid-area: 1 / 2 / 3 / 3;
		align-self: center;
	}
	.row.chained .side.result {
		grid-area: 1 / 3 / 3 / 4;
		align-self: center;
	}
	.row.chained .side.extra {
		grid-area: 3 / 3 / 4 / 4;
	}
	.row.chained .mid2 {
		grid-area: 1 / 4 / 4 / 5;
		align-self: center;
	}
	.row.chained .side.final {
		grid-area: 1 / 5 / 4 / 6;
		align-self: center;
	}
	.terms {
		display: flex;
		flex-direction: column;
		gap: 0.7rem;
	}
	/*
	 * 两项之间的分隔虚线。只给 `border-top`、不给高度 → 元素高度就是那 1px 边框
	 * （`box-sizing: border-box` 下 `height: 0` 会被边框撑到 1px，不如直接不写）。
	 * 上下各留 `gap` 的间距，所以线是"浮"在两项中间的。
	 */
	.tsep {
		border-top: 1px dashed #cbd5e1;
	}
	.mid,
	.mid2 {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.1rem;
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
	.side.result .nm,
	.side.final .nm {
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
	.w {
		font-size: 0.66rem;
		color: #b45309;
	}
	.src {
		font-size: 0.6rem;
		color: #059669;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.op {
		font-size: 1rem;
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
</style>
