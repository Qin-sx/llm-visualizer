<script lang="ts">
	/**
	 * 流程总览里的一个**方框**：张量名 + 尺寸（可选一条分段条）。
	 *
	 * 方框只保留"这一步产出了什么、它是什么角色"。
	 *
	 * 配色规则：填了 `ref.tone` 就用它；否则按角色默认——**产物靛蓝、操作数灰**
	 * （与各视图里"产物用靛蓝名字"是同一条规则）。
	 */
	import type { MatRef } from '$lib/core/types';
	import { refKey } from '$lib/core/overview';

	let {
		ref: r,
		role = 'operand',
		active = false,
		stepId = '',
		onpick
	}: {
		ref: MatRef;
		/** 这一步的产物还是操作数——决定默认配色 */
		role?: 'operand' | 'product';
		active?: boolean;
		/** 所属步骤——总览连线靠它判断"这条线是不是指向本步" */
		stepId?: string;
		onpick?: () => void;
	} = $props();

	const TONES: Record<string, { fill: string; stroke: string; text: string; sz: string }> = {
		operand: { fill: '#f8fafc', stroke: '#e2e8f0', text: '#475569', sz: '#94a3b8' },
		product: { fill: '#eef2ff', stroke: '#c7d2fe', text: '#3730a3', sz: '#6366f1' },
		input: { fill: '#f1f5f9', stroke: '#cbd5e1', text: '#334155', sz: '#94a3b8' },
		latent: { fill: '#eef2ff', stroke: '#a5b4fc', text: '#3730a3', sz: '#6366f1' },
		rope: { fill: '#fffbeb', stroke: '#fcd34d', text: '#92400e', sz: '#b45309' },
		head: { fill: '#faf5ff', stroke: '#d8b4fe', text: '#6b21a8', sz: '#a855f7' },
		result: { fill: '#ecfdf5', stroke: '#6ee7b7', text: '#065f46', sz: '#059669' },
		weight: { fill: '#f8fafc', stroke: '#cbd5e1', text: '#475569', sz: '#94a3b8' },
		cache: { fill: '#fef2f2', stroke: '#fca5a5', text: '#991b1b', sz: '#ef4444' }
	};

	const t = $derived(TONES[r.tone ?? role] ?? TONES.operand);
	const total = $derived((r.split ?? []).reduce((s, x) => s + x.ratio, 0) || 1);
</script>

<!--
	`data-ov-node` / `data-ov-step` 是**总览连线的锚点**（`Overview.svelte` 的 `redraw()` 靠它
	量出每个张量方框的位置，把前后复用的同一个张量连成曲线）。删掉这两个属性，连线会**静默消失**
	——页面不报错，只是不画。加方框时记得一起带上。
-->
<div
	class="box"
	class:active
	data-ov-node={refKey(r)}
	data-ov-step={stepId}
	style={`background:${t.fill};border-color:${active ? '#4f46e5' : t.stroke}`}
	title={`${r.name}  [${r.shape.join(' × ')}]`}
	role="button"
	tabindex="0"
	onclick={() => onpick?.()}
	onkeydown={(e) => e.key === 'Enter' && onpick?.()}
>
	<span class="nm" style={`color:${t.text}`}>{r.name}</span>

	{#if r.split?.length}
		<!-- 分段条：这个张量是几段拼起来的（如 Q_h = [nope | rope]） -->
		<div class="split">
			{#each r.split as seg (seg.label)}
				{@const st = TONES[seg.tone ?? 'head'] ?? TONES.head}
				<span
					class="seg"
					style={`width:${(seg.ratio / total) * 100}%;background:${st.fill};color:${st.text}`}
					>{seg.label}</span
				>
			{/each}
		</div>
	{:else}
		<span class="sz" style={`color:${t.sz}`}>[{r.shape.join(' × ')}]</span>
	{/if}
</div>

<style>
	.box {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 0.05rem;
		min-width: 3.6rem;
		padding: 0.16rem 0.28rem;
		border: 1px solid #e2e8f0;
		border-radius: 4px;
		cursor: pointer;
		transition: box-shadow 0.15s;
	}
	.box:hover {
		box-shadow: 0 0 0 2px rgb(99 102 241 / 0.15);
	}
	.box.active {
		border-width: 2px;
		padding: calc(0.16rem - 1px) calc(0.28rem - 1px);
	}
	.nm {
		font-size: 0.5rem;
		font-weight: 700;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		white-space: nowrap;
		max-width: 4.4rem;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.sz {
		font-size: 0.44rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		white-space: nowrap;
	}
	.split {
		display: flex;
		width: 100%;
		border-radius: 2px;
		overflow: hidden;
	}
	.seg {
		font-size: 0.42rem;
		text-align: center;
		white-space: nowrap;
		overflow: hidden;
	}
</style>
