<script lang="ts">
	/**
	 * 参考数据流图：把几个张量的来龙去脉画成一张分层 DAG。
	 *
	 * 位置由声明者给定（`col` / `row`）。走线一律**正交**（横 → 竖 → 横），
	 * 竖段落在目标列的左间隙里，所以只要相邻列的空行留对，线就不会穿过节点。
	 *
	 * 图是静态的：不参与动画计时，只把本步正在算的节点高亮出来。
	 */
	import type { DataflowEdge, DataflowNode, DataflowSpec } from '$lib/core/types';
	import Emph from './Emph.svelte';

	let { spec }: { spec: DataflowSpec } = $props();

	const NODE_W = 108;
	const NODE_H = 46;
	const GAP_X = 36;
	const GAP_Y = 16;
	const PAD = 8;

	const byId = $derived(new Map(spec.nodes.map((n) => [n.id, n])));
	const cols = $derived(Math.max(...spec.nodes.map((n) => n.col)) + 1);
	const rows = $derived(Math.max(...spec.nodes.map((n) => n.row)) + 1);
	const W = $derived(PAD * 2 + cols * NODE_W + (cols - 1) * GAP_X);
	const H = $derived(PAD * 2 + rows * NODE_H + (rows - 1) * GAP_Y);

	const active = $derived(new Set(spec.active ?? []));

	const x = (c: number) => PAD + c * (NODE_W + GAP_X);
	const y = (r: number) => PAD + r * (NODE_H + GAP_Y);
	const right = (n: DataflowNode) => x(n.col) + NODE_W;
	const cy = (n: DataflowNode) => y(n.row) + NODE_H / 2;

	/** 一条边的两端锚点 */
	function ends(e: DataflowEdge) {
		const a = byId.get(e.from);
		const b = byId.get(e.to);
		if (!a || !b) return null;
		const from = { x: right(a), y: cy(a) };
		const to =
			e.enter === 'bottom'
				? { x: x(b.col) + NODE_W / 2, y: y(b.row) + NODE_H }
				: { x: x(b.col), y: cy(b) };
		const mid = to.x - (e.routeOffset ?? GAP_X / 2);
		return { from, to, mid };
	}

	/** 正交走线：源右边界 → 目标左间隙 → 目标 */
	function path(e: DataflowEdge): string {
		const p = ends(e);
		if (!p) return '';
		const { from, to, mid } = p;
		if (Math.abs(from.y - to.y) < 0.5) return `M${from.x},${from.y} L${to.x},${to.y}`;
		return `M${from.x},${from.y} L${mid},${from.y} L${mid},${to.y} L${to.x},${to.y}`;
	}

	/**
	 * 边标注：**直边放在横线中点上方；拐弯的边放在竖段中点**。
	 *
	 * 这样从同一个节点分叉出去的两条边（如 `[k_nope|v]` 分别给 K_h 和 V_h）
	 * 标注自动落在不同位置，不会叠在一起。
	 */
	function labelAt(e: DataflowEdge): { x: number; y: number } | null {
		if (!e.label) return null;
		const p = ends(e);
		if (!p) return null;
		const { from, to, mid } = p;
		if (Math.abs(from.y - to.y) < 0.5) return { x: (from.x + to.x) / 2, y: from.y - 5 };
		return { x: mid, y: (from.y + to.y) / 2 };
	}

	const TONES: Record<
		string,
		{ fill: string; stroke: string; text: string; line: string; dash?: boolean }
	> = {
		input: { fill: '#f1f5f9', stroke: '#cbd5e1', text: '#334155', line: '#94a3b8' },
		latent: { fill: '#eef2ff', stroke: '#a5b4fc', text: '#3730a3', line: '#6366f1' },
		rope: { fill: '#fffbeb', stroke: '#fcd34d', text: '#92400e', line: '#d97706' },
		head: { fill: '#faf5ff', stroke: '#d8b4fe', text: '#6b21a8', line: '#a855f7' },
		result: { fill: '#ecfdf5', stroke: '#6ee7b7', text: '#065f46', line: '#059669' },
		cache: { fill: '#fef2f2', stroke: '#fca5a5', text: '#991b1b', line: '#ef4444', dash: true },
		// 权重（如吸收合并里"折出来"的 W̄^UQ / W̄^O）：灰色，和激活值区分开
		weight: { fill: '#f8fafc', stroke: '#94a3b8', text: '#475569', line: '#94a3b8' },
		// ── 下面是"同一算子不同实现"的差异标注（见 DataflowNode.tone 的注释）──
		/** 被"折掉"的环节（吸收合并里：q → q̄、折出来的两个权重） */
		absorbed: { fill: '#ecfeff', stroke: '#67e8f9', text: '#155e75', line: '#0891b2' },
		/** 这个实现不再需要的矩阵 */
		gone: { fill: '#f8fafc', stroke: '#cbd5e1', text: '#94a3b8', line: '#cbd5e1', dash: true }
	};
	const tone = (n: DataflowNode) => TONES[n.tone ?? 'input'];

	/** 分段节点的各段（起点 / 宽度 / 配色 / 标签），用来画"拼起来的条" */
	function segments(n: DataflowNode) {
		const list = n.split ?? [];
		const total = list.reduce((s, x) => s + x.ratio, 0) || 1;
		let acc = 0;
		return list.map((seg) => {
			const w = (seg.ratio / total) * NODE_W;
			const x0 = acc;
			acc += w;
			return { x: x0, w, ...seg, t: TONES[seg.tone ?? 'head'] };
		});
	}

	const LEGEND = [
		{ tone: 'latent', text: '低秩潜向量 / 压缩条目' },
		{ tone: 'rope', text: '位置编码段（只碰末尾几维）' },
		{ tone: 'head', text: '按头升维后的 Q / K / V' },
		{ tone: 'weight', text: '可学习的偏置' },
		{ tone: 'cache', text: 'KV Cache' },
		{ tone: 'absorbed', text: '吸收合并改动的环节' },
		{ tone: 'gone', text: '不参与本层计算的矩阵与路径' }
	];
	/** 只有图上真出现这种配色时才列进图例 */
	const shownLegend = $derived(
		LEGEND.filter((l) => spec.nodes.some((n) => (n.tone ?? 'input') === l.tone))
	);
</script>

<div class="df">
	<div class="df-head">
		<span class="df-title">数据流图</span>
		<span class="df-sub">本步在整条链里的位置（当前节点高亮）</span>
	</div>

	<div class="df-scroll">
		<svg viewBox={`0 0 ${W} ${H}`} class="df-svg" style={`max-width:${W}px`}>
			<defs>
				<marker
					id="df-arrow"
					viewBox="0 0 8 8"
					refX="7"
					refY="4"
					markerWidth="6"
					markerHeight="6"
					orient="auto-start-reverse"
				>
					<path d="M0,0 L8,4 L0,8 z" fill="#94a3b8"></path>
				</marker>
				<marker
					id="df-arrow-dim"
					viewBox="0 0 8 8"
					refX="7"
					refY="4"
					markerWidth="6"
					markerHeight="6"
					orient="auto-start-reverse"
				>
					<path d="M0,0 L8,4 L0,8 z" fill="#cbd5e1"></path>
				</marker>
			</defs>

			<!-- 边先画，节点盖在上面 -->
			{#each spec.edges as e, i (i)}
				{@const et = TONES[e.tone ?? 'input']}
				<path
					class="edge"
					d={path(e)}
					fill="none"
					stroke={e.dashed ? '#cbd5e1' : e.tone ? et.line : '#94a3b8'}
					stroke-width={active.has(e.from) || active.has(e.to) ? 1.8 : 1.2}
					stroke-dasharray={e.dashed || et.dash ? '4 3' : undefined}
					opacity={e.faded ? 0.4 : 1}
					marker-end={e.dashed ? 'url(#df-arrow-dim)' : 'url(#df-arrow)'}
				></path>
			{/each}

			<!-- 节点 -->
			{#each spec.nodes as n (n.id)}
				{@const t = tone(n)}
				{@const on = active.has(n.id)}
				<g class="node" class:on opacity={n.faded ? 0.5 : 1}>
					{#if n.split}
						<!-- 分段的条：Q/K 的「nope 段 + rope 段」一眼可见 -->
						<defs>
							<clipPath id={`df-clip-${n.id}`}>
								<rect x={x(n.col)} y={y(n.row)} width={NODE_W} height={NODE_H} rx="6"></rect>
							</clipPath>
						</defs>
						<g clip-path={`url(#df-clip-${n.id})`}>
							{#each segments(n) as seg (seg.label)}
								<rect
									x={x(n.col) + seg.x}
									y={y(n.row)}
									width={seg.w}
									height={NODE_H}
									fill={seg.t.fill}
								></rect>
							{/each}
						</g>
						{#each segments(n).slice(1) as seg (seg.label)}
							<line
								x1={x(n.col) + seg.x}
								y1={y(n.row)}
								x2={x(n.col) + seg.x}
								y2={y(n.row) + NODE_H}
								stroke={t.stroke}
								stroke-width="1"
							></line>
						{/each}
						<rect
							x={x(n.col)}
							y={y(n.row)}
							width={NODE_W}
							height={NODE_H}
							rx="6"
							fill="none"
							stroke={on ? '#4f46e5' : t.stroke}
							stroke-width={on ? 2.4 : 1.2}
						></rect>
					{:else}
						<rect
							x={x(n.col)}
							y={y(n.row)}
							width={NODE_W}
							height={NODE_H}
							rx="6"
							fill={t.fill}
							stroke={on ? '#4f46e5' : t.stroke}
							stroke-width={on ? 2.4 : 1.2}
							stroke-dasharray={t.dash ? '5 3' : undefined}
						></rect>
					{/if}
					<text
						x={x(n.col) + NODE_W / 2}
						y={y(n.row) + (n.split ? 18 : n.sub ? 18 : 27)}
						text-anchor="middle"
						font-size="11.5"
						font-weight="700"
						fill={t.text}
						font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{n.label}</text
					>
					{#if n.split}
						{#each segments(n) as seg (seg.label)}
							<text
								x={x(n.col) + seg.x + seg.w / 2}
								y={y(n.row) + 36}
								text-anchor="middle"
								font-size="8"
								fill={seg.t.text}>{seg.label}</text
							>
						{/each}
					{:else if n.sub}
						<text
							x={x(n.col) + NODE_W / 2}
							y={y(n.row) + 33}
							text-anchor="middle"
							font-size="9"
							fill={t.text}
							opacity="0.75">{n.sub}</text
						>
					{/if}
				</g>
			{/each}

			<!-- 边标注：加白色描边，压在线上也能看清 -->
			{#each spec.edges as e, i (i)}
				{#if labelAt(e)}
					{@const p = labelAt(e)}
					<text
						x={p!.x}
						y={p!.y}
						text-anchor="middle"
						font-size="9"
						fill="#475569"
						stroke="#ffffff"
						stroke-width="3"
						paint-order="stroke"
						font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{e.label}</text
					>
				{/if}
			{/each}
		</svg>
	</div>

	<div class="df-legend">
		{#each shownLegend as l (l.tone)}
			<span class="lg">
				<i style={`background:${TONES[l.tone].fill};border-color:${TONES[l.tone].stroke}`}></i>
				{l.text}
			</span>
		{/each}
		<span class="lg"><i class="dash"></i>虚线 = decode 时从缓存取</span>
	</div>

	{#if spec.hint}
		<div class="df-hint"><Emph text={spec.hint} /></div>
	{/if}
</div>

<style>
	.df {
		display: flex;
		flex-direction: column;
		gap: 0.45rem;
		margin-top: 0.4rem;
		padding: 0.7rem 0.8rem 0.75rem;
		border: 1px dashed #cbd5e1;
		border-radius: 0.5rem;
		background: #fcfdff;
	}
	.df-head {
		display: flex;
		align-items: baseline;
		gap: 0.5rem;
	}
	.df-title {
		font-size: 0.78rem;
		font-weight: 700;
		color: #1e293b;
	}
	.df-sub {
		font-size: 0.68rem;
		color: #94a3b8;
	}
	.df-scroll {
		overflow-x: auto;
		padding-bottom: 0.2rem;
	}
	/*
	 * 图按容器宽度**等比缩放**（`viewBox` + 百分比宽），而不是直接横向滚动：
	 * 常见窗口下刚好铺满，窄一点也不会一进来就被截掉右边的"注意力"节点。
	 * 下限 860px 是给文字留的可读尺寸——再窄就退回滚动（`.df-scroll` 负责）。
	 */
	.df-svg {
		display: block;
		width: 100%;
		height: auto;
		min-width: 860px;
	}
	.node rect {
		transition: stroke 0.2s;
	}
	.df-legend {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem 1rem;
		font-size: 0.64rem;
		color: #64748b;
	}
	.lg {
		display: inline-flex;
		align-items: center;
		gap: 0.28rem;
	}
	.lg i {
		width: 0.7rem;
		height: 0.7rem;
		border-radius: 0.15rem;
		border: 1px solid #cbd5e1;
		background: #f1f5f9;
		display: inline-block;
	}
	.lg i.dash {
		height: 0;
		border: none;
		border-top: 2px dashed #cbd5e1;
		border-radius: 0;
	}
	.df-hint {
		font-size: 0.66rem;
		line-height: 1.5;
		color: #64748b;
	}
</style>
