<script lang="ts">
	/**
	 * 一个矩阵乘的小组——按"乘法表"方位摆放：
	 *
	 *   ┌───────┬───────┐
	 *   │       │  W_Q  │   右操作数在上
	 *   ├───────┼───────┤
	 *   │   X   │   Q   │   左操作数 ｜ 结果
	 *   └───────┴───────┘
	 *
	 * 每块是一个**方框**（名字 + 尺寸）。不加小组边框/底色，只用一条竖直虚线分隔相邻小组。
	 * 点任意方框可跳到它所属的步骤。
	 */
	import type { MatmulItem } from '$lib/core/overview';
	import MiniBox from './MiniBox.svelte';

	let {
		block,
		active,
		last = false,
		onpick
	}: {
		block: MatmulItem;
		active: boolean;
		last?: boolean;
		onpick?: (stepId: string) => void;
	} = $props();

	const cells = $derived([
		{ ref: block.b, cls: 'top', role: 'operand' as const },
		{ ref: block.a, cls: 'left', role: 'operand' as const },
		{ ref: block.out, cls: 'out', role: 'product' as const }
	]);
</script>

<div class="blk" class:active class:last>
	<!-- 占位行：放"前置运算名"（如 掩码 → softmax）与并行角标，同时让各节点的基线对齐 -->
	<div class="preops">
		{#if block.ops?.length}<span class="ops">{block.ops.join(' → ')}</span>{/if}
		{#if block.parallel}<span class="par">×{block.parallel}</span>{/if}
	</div>

	<div class="grid">
		<div class="cell empty"></div>
		{#each cells as c (c.cls)}
			<div class="cell {c.cls}">
				<MiniBox
					ref={c.ref}
					role={c.role}
					{active}
					stepId={block.stepId}
					onpick={() => onpick?.(block.stepId)}
				/>
			</div>
		{/each}
	</div>
</div>

<style>
	.blk {
		position: relative;
		z-index: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.15rem;
		padding: 0.3rem 0.55rem;
		border-right: 1px dashed #e2e8f0;
	}
	.blk.last {
		border-right: none;
	}
	/* 占位，让矩阵乘组和单目组的基线对齐 */
	.preops {
		height: 0.75rem;
		display: flex;
		align-items: center;
		gap: 0.25rem;
		font-size: 0.52rem;
		white-space: nowrap;
	}
	.ops {
		color: #a21caf;
	}
	.par {
		color: #b45309;
	}
	.grid {
		display: grid;
		grid-template-columns: auto auto;
		gap: 3px;
	}
	.cell {
		display: flex;
		flex-direction: column;
		align-items: center;
	}
	.cell.empty {
		cursor: default;
	}
	.cell.top {
		grid-area: 1 / 2;
	}
	.cell.left {
		grid-area: 2 / 1;
	}
	.cell.out {
		grid-area: 2 / 2;
	}
</style>
