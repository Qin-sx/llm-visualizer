<script lang="ts">
	import type { Step, TensorView } from '$lib/core/types';
	import { animationUnits, isAnimatable, progressOf, stepDurationMs } from '$lib/core/steps';
	import { untrack } from 'svelte';
	import Dataflow from './Dataflow.svelte';
	import Formula from './Formula.svelte';
	import Tensor from './Tensor.svelte';
	import Emph from './Emph.svelte';

	let { step, progress = 0 }: { step: Step; progress?: number } = $props();

	/** 动画单元：并行行（`parallel`）里的多个视图合成一个单元一起播，其余各自一个 */
	const units = $derived(animationUnits(step));
	const animCount = $derived(units.length);

	// ── 页级播放控制：**一个**按钮驱动本页所有动画 ──────────
	const SPEEDS = [0.5, 1, 2];
	/** 1× 速度下播完本页所需时间：由本页动画工作量算出来（与底部总播放器同一套） */
	const BASE_MS = $derived(stepDurationMs(step));

	let localMode = $state(false);
	let localProgress = $state(0);
	let playing = $state(false);
	let speed = $state(1);
	let raf = 0;
	let lastTs = 0;

	/** 有效进度：本页接管后用本地进度，否则跟随底部播放条 */
	const eff = $derived(localMode ? localProgress : progress);

	// 底部播放条一旦开始推进，本页就交还控制权——否则页级"接管"状态会一直压着全局进度
	let lastSeen: number | undefined;
	$effect(() => {
		const p = progress;
		if (lastSeen === undefined) {
			lastSeen = p; // 首次只记录基线
			return;
		}
		if (p === lastSeen) return;
		lastSeen = p;
		untrack(() => {
			if (localMode) {
				localMode = false;
				playing = false;
			}
		});
	});

	function tick(ts: number) {
		if (!playing) return;
		const dt = lastTs ? (ts - lastTs) / 1000 : 0;
		lastTs = ts;
		localProgress = Math.min(1, localProgress + (dt * 1000 * speed) / BASE_MS);
		if (localProgress >= 1) {
			playing = false;
			raf = 0;
			return;
		}
		raf = requestAnimationFrame(tick);
	}

	function togglePlay() {
		localMode = true;
		if (playing) {
			playing = false;
			return;
		}
		if (localProgress >= 1) localProgress = 0;
		playing = true;
		lastTs = 0;
		raf = requestAnimationFrame(tick);
	}

	function reset() {
		localMode = true;
		playing = false;
		localProgress = 0;
	}

	function followGlobal() {
		localMode = false;
		playing = false;
	}

	$effect(() => {
		return () => {
			playing = false;
			if (raf) cancelAnimationFrame(raf);
		};
	});

	/** 本页有多个可动画视图时，进度切成等份依次播放；当前在第几段 */
	const segIndex = $derived(
		animCount > 0 ? Math.min(animCount - 1, Math.floor(eff * animCount)) : 0
	);

	/**
	 * 本页进度 `eff`（0..1）下某个视图的实际进度。
	 * 分段与"并行视图按工作量反比缩放"的规则都在 `core/steps.ts`，与 verify 共用一份。
	 */
	function subProgress(t: TensorView): number {
		if (!isAnimatable(t)) return 0;
		return progressOf(t, units, eff);
	}

	/**
	 * 布局：先按 `row` 切成横向的"行"，行内再按 `group` 切成纵向堆叠的"列"。
	 * 没有 `row` 的视图各自独占一行。
	 */
	const layout = $derived.by(() => {
		const rows: {
			row?: number;
			parallel: boolean;
			cols: { group?: string; items: TensorView[] }[];
		}[] = [];
		for (const t of step.tensors) {
			let r = rows[rows.length - 1];
			if (t.row === undefined || !r || r.row !== t.row) {
				r = { row: t.row, parallel: !!t.parallel, cols: [] };
				rows.push(r);
			}
			let c = r.cols[r.cols.length - 1];
			if (t.group === undefined || !c || c.group !== t.group) {
				c = { group: t.group, items: [] };
				r.cols.push(c);
			}
			c.items.push(t);
		}
		return rows;
	});
</script>

{#key step.id}
	<div class="step-panel">
		<div class="head">
			<span class="kind">{step.kind}</span>
			<span class="text-sm font-medium text-slate-800"><Emph text={step.label} /></span>
		</div>

		{#if step.formula}
			<div class="formula-row">
				<Formula tex={step.formula} />
			</div>
		{/if}

		{#if animCount > 0}
			<div class="ctrl-bar">
				<button class="play" onclick={togglePlay}>{playing ? '⏸ 暂停' : '▶ 播放'}</button>
				<button class="mini" onclick={reset} title="重置">↺</button>

				<span class="spd">速度</span>
				{#each SPEEDS as s (s)}
					<button class="chip" class:active={speed === s} onclick={() => (speed = s)}>{s}×</button>
				{/each}

				<div class="track"><div class="fill" style={`width:${eff * 100}%`}></div></div>

				{#if animCount > 1}
					<span class="segs">第 {segIndex + 1} / {animCount} 段</span>
				{/if}
				{#if localMode}
					<button class="mini" onclick={followGlobal} title="交还给底部总播放器">↩</button>
				{/if}
			</div>
		{/if}

		<div class="tensors">
			{#each layout as r, ri (ri)}
				{#if r.cols.length > 1}
					<!-- 横向并排：流水线（→）或并行分支（无箭头） -->
					<div class="hrow" class:parallel={r.parallel}>
						{#each r.cols as c, ci (ci)}
							{#if ci > 0 && !r.parallel}<span class="harrow">→</span>{/if}
							<div class="hcol">
								{#each c.items as t (t.name)}
									<Tensor view={t} progress={subProgress(t)} compact />
								{/each}
							</div>
						{/each}
					</div>
			{:else}
				{#each r.cols[0].items as t (t.name)}
					<Tensor view={t} progress={subProgress(t)} />
				{/each}
			{/if}
		{/each}
		</div>

		<!-- 参考数据流图：静态地图，放在本页最下面（不参与动画计时） -->
		{#if step.diagram}
			<Dataflow spec={step.diagram} />
		{/if}
	</div>
{/key}

<style>
	.step-panel {
		animation: fade-in 0.35s ease-out;
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}
	.head {
		display: flex;
		align-items: baseline;
		gap: 0.5rem;
	}
	.kind {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.6rem;
		letter-spacing: 0.06em;
		padding: 0.1rem 0.35rem;
		border-radius: 0.25rem;
		background: #eef2ff;
		color: #4338ca;
		border: 1px solid #c7d2fe;
	}
	.formula-row {
		background: #f8fafc;
		border: 1px solid #e2e8f0;
		border-radius: 0.375rem;
		padding: 0.5rem 0.75rem;
		overflow-x: auto;
	}

	/* 页级播放控制：一个按钮驱动本页所有动画 */
	.ctrl-bar {
		display: flex;
		align-items: center;
		gap: 0.35rem;
		flex-wrap: wrap;
		padding: 0.4rem 0.5rem;
		background: #ffffff;
		border: 1px solid #e2e8f0;
		border-radius: 0.5rem;
	}
	.play {
		padding: 0.28rem 0.7rem;
		border-radius: 0.375rem;
		border: 1px solid #4f46e5;
		background: #4f46e5;
		color: #fff;
		font-size: 0.75rem;
		cursor: pointer;
		white-space: nowrap;
	}
	.play:hover {
		background: #4338ca;
	}
	.mini {
		padding: 0.22rem 0.45rem;
		border-radius: 0.3rem;
		border: 1px solid #e2e8f0;
		background: #f8fafc;
		color: #64748b;
		font-size: 0.72rem;
		cursor: pointer;
	}
	.mini:hover {
		background: #f1f5f9;
	}
	.spd {
		font-size: 0.68rem;
		color: #94a3b8;
		margin-left: 0.25rem;
	}
	.chip {
		padding: 0.18rem 0.42rem;
		border-radius: 999px;
		border: 1px solid #e2e8f0;
		background: #fff;
		color: #64748b;
		font-size: 0.68rem;
		cursor: pointer;
	}
	.chip.active {
		background: #eef2ff;
		border-color: #c7d2fe;
		color: #4338ca;
	}
	.track {
		flex: 1 1 6rem;
		min-width: 4rem;
		height: 0.3rem;
		border-radius: 999px;
		background: #e2e8f0;
		overflow: hidden;
	}
	.fill {
		height: 100%;
		background: #6366f1;
		transition: width 0.08s linear;
	}
	.segs {
		font-size: 0.66rem;
		color: #94a3b8;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		white-space: nowrap;
	}

	.tensors {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
		gap: 1.4rem 1rem;
		align-items: start;
	}
	/* 矩阵乘 / 掩码等视图要占满整行——矩阵必须按 2×2 排布，不能被挤到换行 */
	.tensors :global(.tensor.wide) {
		grid-column: 1 / -1;
	}

	/* 横向流水线 / 并行分支 */
	.hrow {
		grid-column: 1 / -1;
		display: flex;
		align-items: flex-start;
		gap: 0.55rem;
		overflow-x: auto;
		padding-bottom: 0.4rem;
	}
	/* 并行分支之间不画箭头，用更大的间距表达"同时进行" */
	.hrow.parallel {
		gap: 1.6rem;
	}
	.hcol {
		display: flex;
		flex-direction: column;
		gap: 0.8rem;
		flex-shrink: 0;
	}
	.harrow {
		align-self: center;
		color: #cbd5e1;
		font-size: 1.1rem;
		font-weight: 700;
		flex-shrink: 0;
	}

	@keyframes fade-in {
		from {
			opacity: 0;
			transform: translateY(4px);
		}
		to {
			opacity: 1;
			transform: none;
		}
	}
</style>
