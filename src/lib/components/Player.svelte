<script lang="ts">
	/**
	 * 底部播放条：**播放整体流程**——沿时间轴依次走过所有步骤，
	 * 每一步的动画随进度自动播放。
	 *
	 * 与页级控制条（StepPanel）的分工：
	 *   - 这里：整段连播 + 跨步骤跳转
	 *   - 页级：只播当前这一页的动画，可单独调速
	 */
	let {
		playing,
		progress,
		stepIndex,
		stepCount,
		speed,
		onToggle,
		onPrev,
		onNext,
		onRestart,
		onSeek,
		onSpeed
	}: {
		playing: boolean;
		progress: number;
		stepIndex: number;
		stepCount: number;
		speed: number;
		onToggle: () => void;
		onPrev: () => void;
		onNext: () => void;
		onRestart: () => void;
		onSeek: (ratio: number) => void;
		onSpeed: (v: number) => void;
	} = $props();

	const speeds = [0.5, 1, 1.5, 2];
</script>

<div class="player">
	<button class="btn" onclick={onRestart} title="从头播放整体流程">⏮</button>
	<button class="btn" onclick={onPrev} title="上一步">⏪</button>
	<button class="btn primary" onclick={onToggle}>{playing ? '⏸ 暂停' : '▶ 播放整体流程'}</button>
	<button class="btn" onclick={onNext} title="下一步">⏩</button>

	<span class="step">第 <b>{stepIndex + 1}</b> / {stepCount} 步</span>

	<div class="track">
		<input
			type="range"
			min="0"
			max="1000"
			value={Math.round(progress * 1000)}
			oninput={(e) => onSeek(Number((e.currentTarget as HTMLInputElement).value) / 1000)}
		/>
		<div class="meta">
			<span>拖动可跳到任意位置</span>
			<span class="hint">单页调速请用页面上的 ▶</span>
		</div>
	</div>

	<div class="speeds">
		{#each speeds as s (s)}
			<button class="chip" class:active={speed === s} onclick={() => onSpeed(s)}>{s}×</button>
		{/each}
	</div>
</div>

<style>
	.player {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.55rem 0.9rem;
		background: #ffffff;
		border: 1px solid #e2e8f0;
		border-radius: 0.6rem;
		box-shadow: 0 1px 2px rgb(15 23 42 / 0.04);
	}
	.btn {
		padding: 0.3rem 0.6rem;
		border-radius: 0.375rem;
		border: 1px solid #e2e8f0;
		background: #f8fafc;
		color: #334155;
		font-size: 0.8rem;
		cursor: pointer;
		white-space: nowrap;
	}
	.btn:hover {
		background: #f1f5f9;
	}
	.btn.primary {
		background: #4f46e5;
		border-color: #4f46e5;
		color: #ffffff;
		min-width: 8.2rem;
	}
	.btn.primary:hover {
		background: #4338ca;
	}
	.step {
		font-size: 0.74rem;
		color: #64748b;
		white-space: nowrap;
	}
	.step b {
		color: #3730a3;
	}
	.track {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		min-width: 7rem;
	}
	input[type='range'] {
		width: 100%;
		accent-color: #4f46e5;
	}
	.meta {
		display: flex;
		justify-content: space-between;
		font-size: 0.65rem;
		color: #94a3b8;
	}
	.hint {
		font-style: italic;
	}
	.speeds {
		display: flex;
		gap: 0.25rem;
	}
	.chip {
		padding: 0.2rem 0.45rem;
		font-size: 0.7rem;
		border-radius: 999px;
		border: 1px solid #e2e8f0;
		background: #ffffff;
		color: #64748b;
		cursor: pointer;
	}
	.chip.active {
		background: #eef2ff;
		border-color: #c7d2fe;
		color: #4338ca;
	}
</style>
