<script lang="ts">
	/**
	 * 查表过程视图：按 key 从表里把行取出来。
	 *
	 *   输入 token（key）  t0 = #3   t1 = #7  ...
	 *   ┌──────────┐  查表  ┌──────────┐
	 *   │ W_E      │ ────▶  │ x_0      │   左边整张表（正在取的那行高亮），
	 *   │ 当前行高亮│        │ 逐行揭示  │   右边取出来的行摞成结果矩阵
	 *   └──────────┘        └──────────┘
	 *   下方给出"结果第 i 行就是表里第 key 行"的对照，强调**没有任何计算**。
	 *
	 * 与 `TransformView` 的区别：变换是"逐行算出新值"（行号一一对应），
	 * 查表是"按 key 取任意行"（行号由 key 决定）。数学上查表能写成
	 * `onehot · W_E`，但那不是 embedding 的真实语义，所以单列一种视图。
	 */
	import type { LookupView, MatRef } from '$lib/core/types';
	import MatrixGrid from './MatrixGrid.svelte';
	import ValueRow from './ValueRow.svelte';
	import { realLabel } from './realShape';
	import Emph from './Emph.svelte';

	let { view, progress = 0 }: { view: LookupView; progress?: number } = $props();

	const rows = $derived(view.rows.length);
	const doneRows = $derived(Math.min(rows, Math.round(progress * rows)));
	/** 正在取第几行 */
	const current = $derived(Math.min(Math.max(0, rows - 1), doneRows));
	const key = $derived(view.keys[current] ?? 0);

	/** 未取到的行留白，取到的写真实数字 */
	const revealed = $derived(
		view.rows.map((r, i) => (i < doneRows ? r : r.map(() => null)))
	);

	const cellSize = 24;
</script>

{#snippet head(ref: MatRef, accent = false)}
	<!-- data-handoff 让 core/handoff.js 在步骤切换时把这个矩阵"飞"过来 -->
	<div class="head" data-handoff={ref.handoffId} data-handoff-name={ref.name}>
		<span class="nm" class:accent>{ref.name}</span>
		<span class="sz">[{ref.shape.join(' × ')}]</span>
	</div>
	<!-- 这两行都**恒定占位**（没有内容时塞不换行空格）：
	     条件渲染会让同一行里各矩阵的顶边错开一行 -->
	<div class="real">{realLabel(ref) ?? '\u00a0'}</div>
	<div class="src"><Emph text={ref.label ?? '\u00a0'} /></div>
{/snippet}

<div class="lk">
	<div class="keys">
		<span class="keys-title">{view.keyName ?? 'key'}</span>
		{#each view.keys as k, i (i)}
			<span class="chip" class:done={i < doneRows} class:on={i === current}>t{i} = #{k}</span>
		{/each}
	</div>

	<div class="row">
		<div class="side">
			{@render head(view.table)}
			<MatrixGrid data={view.table.data ?? []} {cellSize} highlightRow={key} />
		</div>

		<div class="mid">
			<div class="op">查表</div>
			<div class="ar">──▶</div>
			<div class="hint">按 id 取行</div>
		</div>

		<div class="side result">
			{@render head(view.result, true)}
			<MatrixGrid data={revealed} {cellSize} highlightRow={current} />
			<div class="prog">已取出 <b>{doneRows}</b> / {rows} 行</div>
		</div>
	</div>

	<div class="detail">
		<div class="eq">
			<b>{view.result.name}[{current}]</b>
			<span class="eqop">=</span>
			<b>{view.table.name}[{key}]</b>
			<span class="eqop">— 直接取第 {key} 行，没有任何乘法</span>
		</div>
		<ValueRow title={`${view.table.name}[${key}]`} data={view.rows[current] ?? []} />
	</div>
</div>

<style>
	.lk {
		display: flex;
		flex-direction: column;
		gap: 0.7rem;
	}
	.keys {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		flex-wrap: wrap;
	}
	.keys-title {
		font-size: 0.68rem;
		color: #64748b;
		margin-right: 0.2rem;
	}
	.chip {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.66rem;
		padding: 0.1rem 0.35rem;
		border-radius: 0.25rem;
		border: 1px solid #e2e8f0;
		background: #f8fafc;
		color: #94a3b8;
	}
	.chip.done {
		background: #eef2ff;
		border-color: #c7d2fe;
		color: #4338ca;
	}
	.chip.on {
		background: #fef3c7;
		border-color: #f59e0b;
		color: #b45309;
		font-weight: 700;
	}
	.row {
		display: grid;
		grid-template-columns: auto auto auto;
		align-items: start;
		justify-content: start;
		column-gap: 0.8rem;
		overflow-x: auto;
		padding-bottom: 0.2rem;
	}
	.side {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		/* 与 TransformView 一致：各块同内边距，并排时矩阵顶边能对齐 */
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
		white-space: nowrap;
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
	.detail {
		display: flex;
		align-items: center;
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
</style>
