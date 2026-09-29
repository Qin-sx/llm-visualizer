<script lang="ts">
	/**
	 * 掩码过程动画。
	 *
	 * 按**行优先**顺序逐个处理格子：
	 *   - 还没处理的格子：原值淡显（背景近白、文字浅灰）
	 *   - 处理且保留（j ≤ i）：原值正常显示
	 *   - 处理且屏蔽（j > i）：变成 −∞
	 *
	 * 进度跑完后就是标准的下三角掩码矩阵。
	 */
	import type { MaskView } from '$lib/core/types';
	import { cellColor, cellTextColor, fmtCell, maxAbsOf } from './color';

	let {
		view,
		progress = 0,
		compact = false,
		bare = false
	}: { view: MaskView; progress?: number; compact?: boolean; bare?: boolean } = $props();

	const rows = $derived(view.scores.length);
	const cols = $derived(view.scores[0]?.length ?? 0);
	const total = $derived(rows * cols);
	const maxAbs = $derived(maxAbsOf(view.scores));

	/** 已经处理到第几个格子（行优先） */
	const done = $derived(Math.round(progress * total));
	const currentIdx = $derived(Math.min(total - 1, done));

	const cellSize = $derived(compact ? 24 : 34);
	const w = $derived(cols * cellSize);
	const h = $derived(rows * cellSize);

	type State = 'pending' | 'kept' | 'masked';

	function stateOf(i: number, j: number): State {
		const idx = i * cols + j;
		if (idx >= done) return 'pending';
		return view.mask[i][j] ? 'kept' : 'masked';
	}

	function valueOf(i: number, j: number): number | null {
		const st = stateOf(i, j);
		if (st === 'masked') return Number.NEGATIVE_INFINITY;
		return view.scores[i][j];
	}
</script>

<div class="mask-view" class:bare>
	<!-- bare：被别的视图（如 transform 的 preMask）嵌进去时只画矩阵，
	     表头 / 计数 / 图例由外层负责，内边距也交给外层（否则会多出一层 padding 导致错位） -->
	<div class="side">
		{#if !bare}
			<!-- 表头与 .src 行跟 transform / sum 视图保持同款，
			     这样并步并排时矩阵顶边能对齐（少一行就会往上偏 ~27px） -->
			<div class="head">
				<span class="nm">{view.matrixName ?? view.name}</span>
				<span class="sz">[{rows} × {cols}]</span>
			</div>
			<div class="src">{view.label ?? '\u00a0'}</div>
		{/if}
		<svg width={w} height={h} class="block">
			{#each view.scores as row, i (i)}
				{#each row as _v, j (j)}
					{@const st = stateOf(i, j)}
					{@const v = valueOf(i, j)}
					<rect
						x={j * cellSize}
						y={i * cellSize}
						width={cellSize}
						height={cellSize}
						fill={st === 'pending' ? '#fdfdfd' : cellColor(v, maxAbs)}
						stroke="#e2e8f0"
						stroke-width="0.6"
						opacity={st === 'pending' ? 0.75 : 1}
					/>
					{#if st === 'masked'}
						<!-- 斜纹表示被屏蔽 -->
						<line
							x1={j * cellSize}
							y1={i * cellSize + cellSize}
							x2={j * cellSize + cellSize}
							y2={i * cellSize}
							stroke="#e11d48"
							stroke-width="1"
							opacity="0.35"
						/>
					{/if}
					<text
						x={j * cellSize + cellSize / 2}
						y={i * cellSize + cellSize / 2}
						text-anchor="middle"
						dominant-baseline="central"
						font-size="9.5"
						fill={st === 'pending' ? '#cbd5e1' : cellTextColor(v, maxAbs)}>{fmtCell(v)}</text
					>
				{/each}
			{/each}

			<!-- 当前正在处理的格子 -->
			{#if done < total}
				<rect
					x={(currentIdx % cols) * cellSize}
					y={Math.floor(currentIdx / cols) * cellSize}
					width={cellSize}
					height={cellSize}
					fill="none"
					stroke="#0f172a"
					stroke-width="2.5"
				/>
			{/if}
		</svg>
	</div>

	{#if !bare}
		<div class="meta">
			<span>
				已处理 <b>{Math.min(done, total)}</b> / {total}
			</span>
			<span class="killed">屏蔽 {view.mask.flat().filter((m) => !m).length} 个</span>
		</div>
		<div class="legend"><i class="sw kept"></i>保留 <i class="sw masked"></i>置为 −∞</div>
	{/if}
</div>

<style>
	.mask-view {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
	}
	.side {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		padding: 0.2rem 0.3rem;
	}
	/* 被嵌进去时不加内边距（外层已经有同样的内边距，否则会错位 3~5px） */
	.mask-view.bare .side {
		padding: 0;
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
	.sz {
		font-size: 0.66rem;
		color: #4f46e5;
	}
	.src {
		font-size: 0.6rem;
		color: #059669;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.meta {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.6rem;
		flex-wrap: wrap;
		font-size: 0.65rem;
		color: #64748b;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.meta b {
		color: #3730a3;
	}
	.killed {
		color: #e11d48;
	}
	.legend {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		font-size: 0.62rem;
		color: #94a3b8;
		white-space: nowrap;
	}
	.sw {
		display: inline-block;
		width: 0.6rem;
		height: 0.6rem;
		border-radius: 2px;
	}
	.sw.kept {
		background: #6366f1;
	}
	.sw.masked {
		background: #f1f5f9;
		border: 1px solid #e11d48;
	}
</style>
