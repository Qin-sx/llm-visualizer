<script lang="ts">
	/** 只标尺寸的张量标签——大矩阵不再铺成热力图 */
	let {
		name,
		shape,
		note,
		accent = false
	}: { name: string; shape: number[]; note?: string; accent?: boolean } = $props();
	import Emph from './Emph.svelte';
</script>

<div class="shape-tag" class:accent>
	<span class="name">{name}</span>
	<span class="dims">{shape.join(' × ')}</span>
	{#if note}<span class="note"><Emph text={note} /></span>{/if}
</div>

<style>
	.shape-tag {
		display: inline-flex;
		align-items: baseline;
		gap: 0.4rem;
		padding: 0.25rem 0.5rem;
		border-radius: 0.375rem;
		background: #f8fafc;
		border: 1px solid #e2e8f0;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.72rem;
		white-space: nowrap;
		/*
		 * 宽度不能超过所在格子：`step.tensors` 是自动填充网格，两个非 wide 视图同在一行时，
		 * nowrap 的长文本会横穿盖住右边的内容。名字与尺寸仍然 nowrap，只有 note 允许折行。
		 */
		max-width: 100%;
	}
	.shape-tag.accent {
		background: #eef2ff;
		border-color: #c7d2fe;
	}
	.name {
		color: #334155;
		font-weight: 600;
	}
	.dims {
		color: #4f46e5;
	}
	.accent .name {
		color: #3730a3;
	}
	.note {
		color: #94a3b8;
		font-size: 0.65rem;
		/* 长 note 折行，而不是撑宽整个 tag（见上面 .shape-tag 的注释） */
		white-space: normal;
	}
</style>
