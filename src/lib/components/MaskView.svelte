<script lang="ts">
	/**
	 * 掩码过程动画。
	 *
	 * 按**行**为单位依次处理（一行里该留哪些列、该屏蔽哪些列是一次定下来的）：
	 *   - 还没处理的行：原值淡显（背景近白、文字浅灰）
	 *   - 处理且保留（j ≤ i）：原值正常显示
	 *   - 处理且屏蔽（j > i）：变成 −∞
	 *
	 * 进度跑完后就是标准的下三角掩码矩阵。
	 */
	import type { MaskView } from '$lib/core/types';
	import { cellColor, cellTextColor, fmtCell, maxAbsOf } from './color';
	import Emph from './Emph.svelte';

let {
	view,
	progress = 0,
	compact = false,
	bare = false,
	cellSize: cellSizeProp
}: {
	view: MaskView;
	progress?: number;
	compact?: boolean;
	bare?: boolean;
	/** 被嵌进链式变换时由外层给定格子边长——必须和链上别的块一致，否则 S 与 P_h 大小对不上 */
	cellSize?: number;
} = $props();

	const rows = $derived(view.scores.length);
	const cols = $derived(view.scores[0]?.length ?? 0);
	const total = $derived(rows * cols);
	const maxAbs = $derived(maxAbsOf(view.scores));

	/**
	 * 已经处理到第几行——**整行一起**处理（掩码的天然单位是一个 token 行：
	 * "这一行该看哪些列"的因果 / 滑窗规则是一次定下来的）。
	 * 一格一格地点过去会把整段播放拖得很慢（c128 的 `hca-attn` 掩码 88 格逐格点、
	 * 一格 60ms 要 5 秒多），而且看不出"整行规则"；一行 11 格一起出才是
	 * "一次算很多格子"。
	 */
	const doneRows = $derived(Math.min(rows, Math.round(progress * rows)));
	/** 已处理的格子数（整行一起，所以一次跳 `cols` 个） */
	const done = $derived(doneRows * cols);
	/** 当前正在处理的那一行 */
	const currentRow = $derived(Math.min(Math.max(0, rows - 1), doneRows));

	const cellSize = $derived(cellSizeProp ?? (compact ? 24 : 34));
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
		<!-- 表头 + 两行注解跟 transform / matmul 保持同款，而且顺序也一样
		     （名字 → 真实尺寸 → 来源 → 矩阵）：少一行并排时矩阵顶边就会偏。
		     掩码视图没有 `MatRef`、拿不到真实尺寸，塞一个不换行空格占位。 -->
		<div class="head">
			<span class="nm">{view.matrixName ?? view.name}</span>
			<span class="sz">[{rows} × {cols}]</span>
		</div>
		<div class="real">{'\u00a0'}</div>
		<div class="src"><Emph text={view.label ?? '\u00a0'} /></div>
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

			<!-- 当前正在处理的那一行（整行一起处理，高亮也跟着整行走） -->
			{#if done < total}
				<rect
					x="0"
					y={currentRow * cellSize}
					width={w}
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
