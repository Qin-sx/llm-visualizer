<script lang="ts">
	/**
	 * 一行数值：元素少时每个都带数字，元素多时退化为紧凑色条 + 首尾数值。
	 * 矩阵乘过程视图里用它展示"取哪一行 / 取哪一列 / 逐项乘积"。
	 */
	let {
		data,
		labels,
		title,
		max = 16,
		scaleMax,
		reveal,
		revealGroup = 1
	}: {
		data: number[];
		labels?: string[];
		title?: string;
		max?: number;
		scaleMax?: number;
		/**
		 * 0..1：逐个揭示；不传则整行显示。
		 *
		 * 用在"这一行是左边那块逐列算出来的"场合（组内加权平均的 `v̄`）：
		 * 第 c 个数等到第 c 列算完才出现。**格子照旧全部占位**（只是没数、留白底），
		 * 否则一边长一边换行，整页会跟着抖。
		 */
		reveal?: number;
		/**
		 * 每多少个"揭示单位"才出一个数——就是**左边那块矩阵的行数**。
		 *
		 * 两边共用同一个 `progress`（视图上标 `sync`），矩阵按列优先逐格揭示；
		 * 一列 `revealGroup` 格全算完，这里才出对应的那个数。不给出这个换算比，
		 * 两边的 `round` 会各走各的，对不上。
		 */
		revealGroup?: number;
	} = $props();

	const finite = $derived(data.filter(Number.isFinite));
	const maxAbs = $derived(scaleMax ?? Math.max(1e-9, ...finite.map((v) => Math.abs(v))));
	const showAll = $derived(data.length <= max);
	/**
	 * 已经揭示到第几个数。
	 *
	 * 和 `MatrixGrid` 的揭示进度**同一套算式**：矩阵那边是
	 * `done = round(reveal × 行 × 列)`、第 c 列算完当且仅当 `done ≥ (c+1) × 行`；
	 * 这边 `floor(done / 行)` 正好等于"已经算完的列数"。两边因此严格同步。
	 */
	const shownCount = $derived(
		reveal === undefined
			? data.length
			: Math.floor(Math.round(reveal * data.length * revealGroup) / revealGroup)
	);

	function color(v: number): string {
		if (!Number.isFinite(v)) return '#f1f5f9'; // ∅：留白
		const t = Math.max(-1, Math.min(1, v / maxAbs));
		const target = t >= 0 ? [79, 70, 229] : [225, 29, 72];
		const a = Math.abs(t);
		const r = Math.round(255 + (target[0] - 255) * a);
		const g = Math.round(255 + (target[1] - 255) * a);
		const b = Math.round(255 + (target[2] - 255) * a);
		return `rgb(${r},${g},${b})`;
	}

	const fmt = (v: number) => {
		if (!Number.isFinite(v)) return '∅';
		return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2);
	};
	const isDark = (v: number) => Number.isFinite(v) && Math.abs(v / maxAbs) > 0.55;
</script>

<div class="value-row">
	{#if title}<div class="title">{title}</div>{/if}

	{#if showAll}
		<div class="cells">
			{#each data as v, i (i)}
				{@const on = i < shownCount}
				<div
					class="cell"
					class:pending={!on}
					style={on ? `background:${color(v)}` : null}
					title={labels?.[i] ?? `#${i}`}
				>
					<span class="v" class:light={on && isDark(v)} class:muted={!Number.isFinite(v)}
						>{on ? fmt(v) : ''}</span
					>
				</div>
			{/each}
		</div>
	{:else}
		<div class="strip">
			{#each data as v, i (i)}
				<i style={`background:${color(v)};opacity:${i < shownCount ? 1 : 0.12}`}></i>
			{/each}
		</div>
		<div class="note">
			{data.length} 维 · 前 4 维 [{data.slice(0, 4).map(fmt).join(', ')} …]
		</div>
	{/if}
</div>

<style>
	.value-row {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
		min-width: 0;
	}
	.title {
		font-size: 0.68rem;
		color: #64748b;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.cells {
		display: flex;
		flex-wrap: wrap;
		gap: 2px;
	}
	.cell {
		min-width: 2.1rem;
		height: 1.35rem;
		border-radius: 2px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.62rem;
	}
	/* 还没算到的格子：占位但留白（不换行、不抖） */
	.cell.pending {
		background: #f8fafc;
		box-shadow: inset 0 0 0 1px #e2e8f0;
	}
	.v {
		color: #334155;
	}
	.v.light {
		color: #ffffff;
	}
	.v.muted {
		color: #cbd5e1;
	}
	.strip {
		display: flex;
		gap: 1px;
		height: 1.1rem;
	}
	.strip i {
		flex: 1 1 0;
		min-width: 1px;
		border-radius: 1px;
	}
	.note {
		font-size: 0.62rem;
		color: #94a3b8;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
</style>
