<script lang="ts">
	/**
	 * 矩阵网格：每个格子写真实数字，支持行列高亮与"未揭示"格子。
	 *
	 * 这是矩阵乘过程动画的基本单元——A 的当前行、B 的当前列会被高亮，
	 * C 里还没算出来的格子显示为空白。
	 */
	import { cellColor, cellTextColor, fmtCell, maxAbsOf } from './color';

	let {
		data,
		cellSize = 28,
		showValues = true,
		highlightRow,
		highlightCol,
		highlightCell,
		reveal,
		onCellClick
	}: {
		/** null 表示该格尚未算出 */
		data: (number | null)[][];
		cellSize?: number;
		showValues?: boolean;
		highlightRow?: number;
		highlightCol?: number;
		highlightCell?: [number, number] | null;
		/** 0..1：按行优先逐格揭示；不传则全部显示 */
		reveal?: number;
		onCellClick?: (i: number, j: number) => void;
	} = $props();

	const rows = $derived(data.length);
	const cols = $derived(data[0]?.length ?? 0);

	/** 实际渲染的数据：揭示模式下未轮到的格子置为 null */
	const shown = $derived.by(() => {
		if (reveal === undefined) return data;
		const total = rows * cols;
		const done = Math.round(reveal * total);
		return data.map((row, i) => row.map((v, j) => (i * cols + j < done ? v : null)));
	});

	/** 揭示模式下当前正在处理的格子 */
	const revealCell = $derived.by<[number, number] | null>(() => {
		if (reveal === undefined) return null;
		const total = rows * cols;
		const done = Math.round(reveal * total);
		if (done >= total) return null;
		return [Math.floor(done / cols), done % cols];
	});

	const effectiveHighlight = $derived(highlightCell ?? revealCell);

	const maxAbs = $derived(maxAbsOf(shown));

	const w = $derived(cols * cellSize);
	const h = $derived(rows * cellSize);
	const fontSize = $derived(Math.max(7, Math.min(11, cellSize * 0.36)));
	const showText = $derived(showValues && cellSize >= 20);

	const fill = (v: number | null) => cellColor(v, maxAbs);
	const textColor = (v: number | null) => cellTextColor(v, maxAbs);
	const fmt = fmtCell;
</script>

<svg width={w} height={h} class="block" class:clickable={!!onCellClick}>
	{#each shown as row, i (i)}
		{#each row as v, j (j)}
			<rect
				x={j * cellSize}
				y={i * cellSize}
				width={cellSize}
				height={cellSize}
				fill={fill(v)}
				stroke="#e2e8f0"
				stroke-width="0.6"
				onclick={() => onCellClick?.(i, j)}
				role={onCellClick ? 'button' : undefined}
			/>
			{#if showText && v !== null}
				<text
					x={j * cellSize + cellSize / 2}
					y={i * cellSize + cellSize / 2}
					text-anchor="middle"
					dominant-baseline="central"
					font-size={fontSize}
					fill={textColor(v)}>{fmt(v)}</text
				>
			{/if}
		{/each}
	{/each}

	<!-- 高亮的行（来自 A 的当前行） -->
	{#if highlightRow !== undefined && highlightRow >= 0 && highlightRow < rows}
		<rect
			x="0"
			y={highlightRow * cellSize}
			width={w}
			height={cellSize}
			fill="#f59e0b"
			opacity="0.22"
			stroke="#f59e0b"
			stroke-width="2"
			pointer-events="none"
		/>
	{/if}

	<!-- 高亮的列（来自 B 的当前列） -->
	{#if highlightCol !== undefined && highlightCol >= 0 && highlightCol < cols}
		<rect
			x={highlightCol * cellSize}
			y="0"
			width={cellSize}
			height={h}
			fill="#0ea5e9"
			opacity="0.22"
			stroke="#0ea5e9"
			stroke-width="2"
			pointer-events="none"
		/>
	{/if}

	<!-- 当前正在算的那一格 -->
	{#if effectiveHighlight}
		<rect
			x={effectiveHighlight[1] * cellSize}
			y={effectiveHighlight[0] * cellSize}
			width={cellSize}
			height={cellSize}
			fill="none"
			stroke="#0f172a"
			stroke-width="2.5"
			pointer-events="none"
		/>
	{/if}
</svg>

<style>
	.clickable :global(rect) {
		cursor: pointer;
	}
</style>
