<script lang="ts">
	/**
	 * 单目运算组：输入在左、输出在右，运算名标在上方。
	 *
	 * 例：`S_h --掩码 → softmax--> P_h`
	 *
	 * 用方框 + 竖直虚线分隔。
	 */
	import type { UnaryItem } from '$lib/core/overview';
	import MiniBox from './MiniBox.svelte';

	let {
		item,
		active,
		last = false,
		onpick
	}: {
		item: UnaryItem;
		active: boolean;
		last?: boolean;
		onpick?: (stepId: string) => void;
	} = $props();

	const refs = $derived([
		{ ref: item.input, cls: 'in', role: 'operand' as const },
		{ ref: item.output, cls: 'out', role: 'product' as const }
	]);
</script>

<div class="un" class:active class:last>
	<div class="ops">
		{item.ops.join(' → ')}{#if item.parallel}<span class="par">×{item.parallel}</span>{/if}
	</div>

	<div class="row">
		{#each refs as c, i (c.cls)}
			{#if i > 0}<span class="ar">→</span>{/if}
			<div class="cell {c.cls}">
				<MiniBox
					ref={c.ref}
					role={c.role}
					{active}
					stepId={item.stepId}
					onpick={() => onpick?.(item.stepId)}
				/>
			</div>
		{/each}
	</div>
</div>

<style>
	.un {
		position: relative;
		z-index: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.15rem;
		padding: 0.3rem 0.55rem;
		border-right: 1px dashed #e2e8f0;
	}
	.un.last {
		border-right: none;
	}
	.ops {
		font-size: 0.52rem;
		color: #a21caf;
		height: 0.75rem;
		white-space: nowrap;
	}
	.par {
		color: #b45309;
		margin-left: 0.2rem;
	}
	.row {
		display: flex;
		align-items: center;
		gap: 0.25rem;
	}
	.ar {
		color: #cbd5e1;
		font-size: 0.7rem;
	}
	.cell {
		display: flex;
		flex-direction: column;
		align-items: center;
	}
</style>
