<script lang="ts">
	/**
	 * 矩阵网格：每个格子写真实数字，支持行列高亮与"未揭示"格子。
	 *
	 * 这是矩阵乘过程动画的基本单元——A 的当前行、B 的当前列会被高亮，
	 * C 里还没算出来的格子显示为空白。
	 */
	import { cellColor, cellTextColor, fmtCell, maxAbsOf } from './color';
	import type { MatrixBand } from '$lib/core/types';

	let {
		data,
		cellSize = 28,
		showValues = true,
		highlightRow,
		highlightCol,
		highlightCell,
		highlightCols,
		highlightRows,
		bands,
		reveal,
		revealOrder = 'row',
		revealGroup = 0,
		groupRows = 0,
		busyCol: busyColProp,
		busyRows,
		scanRows,
		onCellClick
	}: {
		/** null 表示该格尚未算出 */
		data: (number | null)[][];
		cellSize?: number;
		showValues?: boolean;
		highlightRow?: number;
		highlightCol?: number;
		highlightCell?: [number, number] | null;
		/** 框出这几列（如"输出里被逆 RoPE 转的那几维"），整列加一层底色 + 外框 */
		highlightCols?: number[];
		/**
		 * 框出这一**段行** `[起始, 结束]`（闭区间）——整段加底色 + 外框。
		 *
		 * 用在"整块矩阵都画出来，但只有其中几行参与这一步"的场合：
		 * 如压缩器里"8 个 token 全画出来，框出的这 4 行才是这一条压缩条目看的窗口"。
		 */
		highlightRows?: [number, number];
		/**
		 * 分组框（见 `MatRef.bands`）：把某一段行 / 列用**浅色虚线**圈起来，每段一个颜色。
		 *
		 * 和 `highlightRows` / `highlightCols` 的区别：那两个是"这一段参与这一步"的琥珀色实框，
		 * 全局只有一种颜色；`bands` 是"同一个矩阵里的两类东西"，**多段、多色**——
		 * 如 K / V 里"滑窗的精确 token"（蓝）与"压缩条目"（琥珀）。
		 */
		bands?: MatrixBand[];
		/**
		 * 0..1：逐格揭示；不传则全部显示。
		 * 顺序由 `revealOrder` 决定（默认行优先）。
		 */
		reveal?: number;
		/**
		 * 逐格揭示的顺序：`'row'`（默认）一行一行扫；`'col'` 一列一列扫。
		 *
		 * `'col'` 用在"结果要按列归约"的场合（组内加权平均）：一整列算完，
		 * 右边的 `v̄` 就出对应的那个数；正在累加的那一列会整列加淡底。
		 */
		revealOrder?: 'row' | 'col';
		/**
		 * 逐列揭示时**每多少个"揭示单位"算完一整列**（= 上游那个矩阵的行数）。
		 *
		 * 给了它就**整列一起出现**（不是一格格长出来）：这一列在上游算完了才轮到它，
		 * 换算式 `done = ⌊round(reveal × 列数 × group) / group⌋ × 行数`——
		 * 与上游的 `round(reveal × 上游行数 × 列数)` 恰好同步（见 `ValueRow.revealGroup`）。
		 */
		revealGroup?: number;
		/**
		 * **逐组揭示**（和上游那个视图写同一个"一组几行"）：上游每算完"一组的一列"
		 * 才出**一格**——`done = ⌊round(reveal × 行 × 列 × groupRows) / groupRows⌋`。
		 *
		 * 与 `revealGroup` 的区别：`revealGroup` 是"一整列一起出现"（结果按列长），
		 * `groupRows` 是"一格一格出现、按自己的行序"（结果按行长——先出完第 1 条，再第 2 条）。
		 * 这里的 `行 × 列 × groupRows` 就是**上游的总格数**（上游行数 = 组数 × groupRows、
		 * 上游列数 = 本矩阵列数、组数 = 本矩阵行数）。
		 */
		groupRows?: number;
		/**
		 * 正在累加的那一列（整列加淡底）。
		 *
		 * 不传就按 `reveal` + `revealOrder` 自己推（`matrix` 视图走这条路）；
		 * 调用方自己控制揭示进度时（如 `MatmulView` 的逐元素模式）显式传进来。
		 */
		busyCol?: number;
		/**
		 * 淡底那一段**只覆盖这几行**（闭区间）。
		 *
		 * 逐组揭示时用：正在加的是"某一组的某一列"，不是整列——不限制的话
		 * 另外那几组也跟着变黄，看起来像在加它们。不传就覆盖整个矩阵高度。
		 */
		busyRows?: [number, number];
		/**
		 * 扫描行块：正在被"扫描拼接"的那一段行（浅蓝色行框，随动画移动，闭区间）。
		 *
		 * 与 `highlightRow`（琥珀，表示"正在计算这一行"）区分：这里是"这一整段行正在被拼进
		 * 另一块矩阵"（如 `-sape` 页窗口槽位打分逐行填时，源矩阵上被拷贝的那一段行）。
		 */
		scanRows?: [number, number] | null;
		onCellClick?: (i: number, j: number) => void;
	} = $props();

	const rows = $derived(data.length);
	const cols = $derived(data[0]?.length ?? 0);

	/** 第 (i,j) 格在揭示序列里的序号（行优先 / 列优先） */
	const seqOf = (i: number, j: number) => (revealOrder === 'col' ? j * rows + i : i * cols + j);

	/**
	 * 已经揭示了几格。
	 *
	 * 三种节奏（互斥）：
	 *   - 普通：`round(reveal × 总格数)`；
	 *   - `revealGroup`：**整列一起出现**——`⌊round(reveal × 列数 × group) / group⌋ × 行数`，
	 *     和上游按同一套算式对齐（上游是 `round(reveal × 上游行数 × 列数)`，`列数 × group` 正是它）；
	 *   - `groupRows`：**上游每算完一组的一列就出一格**——
	 *     `⌊round(reveal × 行 × 列 × groupRows) / groupRows⌋`，而 `行 × 列 × groupRows` 正是
	 *     上游的总格数（见 `groupRows` 的说明）。于是"上游第 g 组第 c 列算完" ⟺
	 *     "本矩阵出了第 g 行第 c 格"——先出完第 1 条，再出第 2 条。
	 */
	const revealedCount = $derived(
		reveal === undefined
			? rows * cols
			: groupRows > 0
				? Math.floor(Math.round(reveal * rows * cols * groupRows) / groupRows)
				: revealGroup > 1
					? Math.floor(Math.round(reveal * cols * revealGroup) / revealGroup) * rows
					: Math.round(reveal * rows * cols)
	);

	/** 实际渲染的数据：揭示模式下未轮到的格子置为 null */
	const shown = $derived.by(() => {
		if (reveal === undefined) return data;
		const done = revealedCount;
		return data.map((row, i) => row.map((v, j) => (seqOf(i, j) < done ? v : null)));
	});

	/** 揭示模式下当前正在处理的格子（按 `revealOrder` 的顺序取第 `done` 个） */
	const revealCell = $derived.by<[number, number] | null>(() => {
		if (reveal === undefined) return null;
		const total = rows * cols;
		const done = revealedCount;
		if (done >= total) return null;
		return revealOrder === 'col' ? [done % rows, Math.floor(done / rows)] : [Math.floor(done / cols), done % cols];
	});

	/** 列优先揭示时"正在累加的那一列"——整列加淡底，和右边逐个出现的数对上 */
	const busyCol = $derived(
		busyColProp ?? (revealOrder === 'col' && revealCell ? revealCell[1] : null)
	);

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
				class="cell"
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

	<!-- 分组框：浅色虚线（在其它高亮**下面**，所以先画）——"这一段是滑窗、那一段是压缩" -->
	{#each bands ?? [] as b (b.axis + ':' + b.from + '-' + b.to + ':' + b.tone)}
		{@const bx = b.axis === 'col' ? b.from * cellSize : 0}
		{@const by = b.axis === 'row' ? b.from * cellSize : 0}
		{@const bw = b.axis === 'col' ? (b.to - b.from + 1) * cellSize : w}
		{@const bh = b.axis === 'row' ? (b.to - b.from + 1) * cellSize : h}
		<rect x={bx} y={by} width={bw} height={bh} fill={b.tone} opacity="0.12" pointer-events="none" />
		<rect
			x={bx}
			y={by}
			width={bw}
			height={bh}
			fill="none"
			stroke={b.tone}
			stroke-width="1.8"
			stroke-dasharray="4 3"
			pointer-events="none"
			data-band={b.axis}
		/>
	{/each}

	<!-- 列优先揭示：正在累加的那一列整列加淡底（"这一列还没加完"） -->
	{#if busyCol !== null && busyCol !== undefined}
		{@const by = busyRows ? busyRows[0] * cellSize : 0}
		{@const bhh = busyRows ? (busyRows[1] - busyRows[0] + 1) * cellSize : h}
		<rect
			x={busyCol * cellSize}
			y={by}
			width={cellSize}
			height={bhh}
			fill="#f59e0b"
			opacity="0.16"
			pointer-events="none"
		/>
	{/if}

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

	<!-- 扫描行块（浅蓝）：这一段行正在被拼进另一块矩阵（随动画移动） -->
	{#if scanRows && scanRows[0] >= 0 && scanRows[1] < rows}
		<rect
			x="0"
			y={scanRows[0] * cellSize}
			width={w}
			height={(scanRows[1] - scanRows[0] + 1) * cellSize}
			fill="#38bdf8"
			opacity="0.18"
			stroke="#38bdf8"
			stroke-width="2"
			pointer-events="none"
			data-scan-rows="1"
		/>
	{/if}

	<!-- 框出的列：整列一层淡底 + 一个外框（用来指认"就是这几维"） -->
	{#if highlightCols && highlightCols.length}
		{@const cs = [...highlightCols].sort((a, b) => a - b)}
		<rect
			x={cs[0] * cellSize}
			y="0"
			width={(cs[cs.length - 1] - cs[0] + 1) * cellSize}
			height={h}
			fill="#f59e0b"
			opacity="0.14"
			pointer-events="none"
		/>
		<!-- 外框带 `data-highlight-cols`：探针/校验就靠它数"框了几处" -->
		<rect
			x={cs[0] * cellSize}
			y="0"
			width={(cs[cs.length - 1] - cs[0] + 1) * cellSize}
			height={h}
			fill="none"
			stroke="#f59e0b"
			stroke-width="2.5"
			pointer-events="none"
			data-highlight-cols="1"
		/>
	{/if}

	<!-- 框出的行段：整段加一层淡底 + 一个外框（"只有这几行参与这一步"） -->
	{#if highlightRows}
		{@const rs = highlightRows}
		<rect
			x="0"
			y={rs[0] * cellSize}
			width={w}
			height={(rs[1] - rs[0] + 1) * cellSize}
			fill="#f59e0b"
			opacity="0.14"
			pointer-events="none"
		/>
		<rect
			x="0"
			y={rs[0] * cellSize}
			width={w}
			height={(rs[1] - rs[0] + 1) * cellSize}
			fill="none"
			stroke="#f59e0b"
			stroke-width="2.5"
			pointer-events="none"
			data-highlight-rows="1"
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
