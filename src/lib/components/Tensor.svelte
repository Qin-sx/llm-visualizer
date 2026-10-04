<script lang="ts">
	/**
	 * 张量视图分发器：渲染层只认识 shape / row / matrix / tiles / bars / matmul /
	 * mask / concat / transform / sum / lookup / kvcache 十二种，完全不知道它们来自哪个算子。
	 *
	 * `progress`（当前步内进度 0..1）会传给需要做动画的视图。
	 */
	import type { TensorView } from '$lib/core/types';
	import ConcatView from './ConcatView.svelte';
	import KvCacheView from './KvCacheView.svelte';
	import LookupView from './LookupView.svelte';
	import MaskView from './MaskView.svelte';
	import MatrixGrid from './MatrixGrid.svelte';
	import MatmulView from './MatmulView.svelte';
	import ShapeTag from './ShapeTag.svelte';
	import SumView from './SumView.svelte';
	import TransformView from './TransformView.svelte';
	import ValueRow from './ValueRow.svelte';
	import Emph from './Emph.svelte';

	/**
	 * `cellSize`：标了 `fillWidth` 的行由 `StepPanel` 量出来（"占满可用宽度"的格子边长），
	 * 原样透给带格子网格的视图；没有就由各视图自己按列数兜底。
	 */
	let {
		view,
		progress = 0,
		compact = false,
		cellSize,
		scanRows
	}: { view: TensorView; progress?: number; compact?: boolean; cellSize?: number; scanRows?: [number, number] | null } = $props();

	const barMax = $derived(view.kind === 'bars' ? Math.max(1, ...view.data) : 1);
	/** 已长出来的柱子数（不带动画的分布一次画满） */
	const barDone = $derived(
		view.kind === 'bars' && view.animated ? Math.round(progress * view.data.length) : Infinity
	);
</script>

<div
	class="tensor"
	class:wide={view.wide ||
		view.kind === 'matmul' ||
		view.kind === 'mask' ||
		view.kind === 'concat' ||
		view.kind === 'transform' ||
		view.kind === 'sum' ||
		view.kind === 'lookup' ||
		view.kind === 'kvcache' ||
		// tiles 是"并排的几个小矩阵"（各头分数、复制出来的 K/V），
		// 落在默认的 220px 网格列里会折成 3+1 两行，占满整行才排得开
		view.kind === 'tiles'}
	data-tensor={view.name}
>
	<!-- mask 视图自带表头（要和并排的 transform 对齐），所以不用这里的通用标签行 -->
	{#if view.label && view.kind !== 'matmul' && view.kind !== 'mask'}
		<div class="label"><Emph text={view.label} /></div>
	{/if}

	{#if view.kind === 'shape'}
		<div>
			<ShapeTag name={view.name} shape={view.shape} note={view.note} accent={view.accent} />
		</div>
	{:else if view.kind === 'row'}
		<ValueRow
			data={view.data}
			labels={view.labels}
			scaleMax={view.scaleMax}
			reveal={view.animated ? progress : undefined}
			revealGroup={view.revealGroup}
		/>
	{:else if view.kind === 'matrix'}
		<!-- 有 `title`（显示名）时画一个和 transform / matmul 同款的表头——`matrix` 的
		     `name` 是内部 id，不给人看（如压缩器的结果矩阵 `v̄`） -->
		{#if view.title}
			<div class="matrix-head">
				<span class="nm">{view.title}</span>
				<span class="sz">[{view.shape.join(' × ')}]</span>
			</div>
		{/if}
		<MatrixGrid
			data={view.data}
			cellSize={cellSize ?? view.cellSize ?? (compact ? 22 : 26)}
			highlightCell={view.highlight ?? null}
			highlightRows={view.highlightRows}
			highlightCols={view.highlightCols}
			reveal={view.animated ? progress : undefined}
			revealOrder={view.revealOrder}
			revealGroup={view.revealGroup}
			groupRows={view.groupRows}
			scanRows={scanRows}
		/>
	{:else if view.kind === 'tiles'}
		<div class="flex flex-wrap gap-3">
			{#each view.data as m, i (i)}
				<div>
					{#if view.tileLabels?.[i]}
						<div class="tile-label">{view.tileLabels[i]}</div>
					{/if}
					<MatrixGrid data={m} cellSize={15} showValues={false} />
				</div>
			{/each}
		</div>
	{:else if view.kind === 'bars'}
		<div class="flex flex-col gap-1">
			{#each view.data as v, i (i)}
				<div class="flex items-center gap-2">
					<span class="w-16 shrink-0 text-[10px] text-slate-500">{view.labels?.[i] ?? i}</span>
					<div
						class="h-3 rounded-sm bg-indigo-500"
						style={`width:${i < barDone ? Math.max(2, (v / barMax) * 130) : 0}px`}
					></div>
					<span class="text-[10px] text-slate-400">{i < barDone ? v : ''}</span>
				</div>
			{/each}
		</div>
	{:else if view.kind === 'matmul'}
		<MatmulView {view} {progress} {compact} cellSize={cellSize ?? view.cellSize} />
	{:else if view.kind === 'ewise'}
		<!-- 逐元素二元运算和矩阵乘共用同一个 2×2 视图（排布、动画都一样，只是没有 Σ） -->
		<MatmulView {view} {progress} {compact} cellSize={cellSize ?? view.cellSize} />
	{:else if view.kind === 'mask'}
		<MaskView {view} {progress} {compact} cellSize={cellSize ?? view.cellSize} />
	{:else if view.kind === 'concat'}
		<ConcatView {view} {progress} />
	{:else if view.kind === 'transform'}
		<TransformView {view} {progress} {compact} cellSize={cellSize ?? view.cellSize} {scanRows} />
	{:else if view.kind === 'sum'}
		<SumView {view} {progress} {compact} />
	{:else if view.kind === 'lookup'}
		<LookupView {view} {progress} />
	{:else if view.kind === 'kvcache'}
		<KvCacheView {view} {progress} />
	{/if}
</div>

<style>
	.tensor {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		min-width: 0;
	}
	.label {
		font-size: 0.7rem;
		color: #64748b;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.tile-label {
		font-size: 0.6rem;
		color: #94a3b8;
		margin-bottom: 0.15rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	/* matrix 视图的表头（有 `title` 时）：和 transform / matmul 的 `.head` 同款，
	   名字 + 尺寸都是等宽字体，顶边才好跟同行矩阵对齐 */
	.matrix-head {
		display: flex;
		align-items: baseline;
		gap: 0.35rem;
		padding: 0 0.3rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.matrix-head .nm {
		font-size: 0.78rem;
		font-weight: 700;
		color: #334155;
	}
	.matrix-head .sz {
		font-size: 0.66rem;
		color: #4f46e5;
	}
</style>
