<script lang="ts">
	/** 只标尺寸的张量标签——大矩阵不再铺成热力图 */
	let {
		name,
		shape,
		note,
		accent = false
	}: { name: string; shape: number[]; note?: string; accent?: boolean } = $props();
</script>

<div class="shape-tag" class:accent>
	<span class="name">{name}</span>
	<span class="dims">{shape.join(' × ')}</span>
	{#if note}<span class="note">{note}</span>{/if}
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
	}
</style>
