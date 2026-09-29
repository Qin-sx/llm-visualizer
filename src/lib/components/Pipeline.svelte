<script lang="ts">
	/**
	 * 主流水线。
	 *
	 * 节点**完全由流程提供**（`FlowSpec` 的 stages），本组件不写死任何节点。
	 *
	 * 每张卡片只有标题 + 副标题；模型级阶段用淡色区分（它不属于"正在检视的这一层"）。
	 * `StageInfo.depth`（L1 / L2 表示深度）仍由流程声明、`verify` 仍会断言，但不在卡片上显示。
	 */
	import type { StageInfo } from '$lib/core/flow';

	let {
		stages,
		activeStage,
		layer,
		layerCount
	}: {
		stages: StageInfo[];
		activeStage: string;
		layer: number;
		layerCount: number;
	} = $props();

	/** 模型级阶段（整个模型只做一次）与层级阶段（每层各一次）——页脚说明由数据推出来 */
	const modelStages = $derived(stages.filter((s) => s.scope === 'model'));
	const layerStages = $derived(stages.filter((s) => s.scope === 'layer'));
</script>

<div class="pipeline">
	{#each stages as n, i (n.id)}
		<div class="node" class:active={activeStage === n.id} class:model={n.scope === 'model'}>
			<div class="title">{n.title}</div>
			<div class="sub">{n.sub ?? ''}</div>
		</div>
		{#if i < stages.length - 1}
			<div class="arrow">→</div>
		{/if}
	{/each}
</div>

<div class="note">
	正在检视 <b>第 {layer + 1} 层</b>（共 {layerCount} 层）。{#if modelStages.length && layerStages.length}{modelStages
			.map((s) => s.title)
			.join(' / ')} 是<b>模型级</b>的（整个模型各一次）；{layerStages
			.map((s) => s.title)
			.join(' / ')} 展开的是正在检视的这一层。{/if}
</div>

<style>
	.pipeline {
		display: flex;
		align-items: stretch;
		gap: 0.5rem;
	}
	.node {
		flex: 1;
		border: 1px solid #e2e8f0;
		border-radius: 0.5rem;
		padding: 0.5rem 0.65rem;
		background: #ffffff;
		transition:
			border-color 0.2s,
			background 0.2s,
			box-shadow 0.2s;
	}
	/* 模型级阶段用淡色区分：它不属于"正在检视的这一层" */
	.node.model {
		background: #f8fafc;
	}
	.node.active {
		border-color: #6366f1;
		background: #eef2ff;
		box-shadow: 0 0 0 3px rgb(99 102 241 / 0.12);
	}
	.title {
		font-size: 0.8rem;
		font-weight: 600;
		color: #1e293b;
	}
	.sub {
		margin-top: 0.2rem;
		font-size: 0.68rem;
		color: #64748b;
	}
	.arrow {
		display: flex;
		align-items: center;
		color: #cbd5e1;
	}
	.note {
		margin-top: 0.6rem;
		font-size: 0.72rem;
		color: #64748b;
	}
</style>
