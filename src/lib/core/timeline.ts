/**
 * 声明式时间轴播放器。
 *
 * 做法：用一条 GSAP timeline 只驱动一个代理对象 `{ t }` 从 0 走到总时长，
 * 再在 onUpdate 里把 t 换算成「第几步 + 步内进度」。这样：
 *
 *   - 播放 / 暂停 / 变速  → tl.play / tl.pause / tl.timeScale
 *   - 单步前进 / 后退     → tl.time(starts[i])
 *   - 拖动进度条          → tl.time(ratio * total)
 *   - 倒放                → tl.timeScale(-1)
 *
 * 视觉渲染是声明式的：组件只读 `stepIndex` / `stepProgress`。
 */
import { gsap } from 'gsap';
import type { Step } from './types';
import { stepDurationMs } from './steps';

export interface Playhead {
	stepIndex: number;
	/** 当前步内的进度 0..1 */
	stepProgress: number;
	time: number;
	duration: number;
}

export interface Player {
	play(): void;
	pause(): void;
	toggle(): void;
	next(): void;
	prev(): void;
	seekRatio(r: number): void;
	/** 直接跳到第 i 步的开头 */
	seekStep(i: number): void;
	/**
	 * 跳到第 i 步内部的某个比例（`0..1`）。
	 *
	 * 页级"本页播放"接管过之后要**交还**控制权时用：先把总播放器挪到"现在显示的位置"，
	 * 再交还——否则交还的瞬间画面会跳回总播放器原来停着的地方。
	 */
	seekStepRatio(i: number, r: number): void;
	setSpeed(s: number): void;
	restart(): void;
	destroy(): void;
	readonly total: number;
}

export function createPlayer(
	steps: Step[],
	onUpdate: (p: Playhead) => void,
	onPlayState?: (playing: boolean) => void
): Player {
	/**
	 * 每步的**终点**时刻（累计和），以及由它错位得到的**起点**时刻。
	 *
	 * `starts[i]` 必须**就是** `ends[i-1]` 这同一个数（用 `slice` 错位，而不是各自
	 * 累加一遍），这样"跳到第 i 步开头"与"第 i-1 步的结束边界"才落在同一个浮点数上。
	 */
	const ends: number[] = [];
	let acc = 0;
	for (const s of steps) {
		acc += stepDurationMs(s) / 1000; // 秒
		ends.push(acc);
	}
	const total = acc || 1;
	const starts = [0, ...ends.slice(0, -1)];

	/**
	 * 边界比较的容差。
	 *
	 * 必须留这个容差：GSAP 内部用 `progress = time / duration` 存进度，渲染时再乘回来，
	 * 一来一回会掉精度——`tl.time(x)` 之后再读 `tl.time()` 可能得到比 `x` 小一丁点的值，
	 * 于是"刚好落在段边界上"的时刻被判成还在上一段里，`next()` 再也翻不过去。
	 * 容差取 1µs：远大于这个 ULP 误差，又小到看不出来。
	 */
	const EPS = 1e-6;

	const proxy = { t: 0 };
	let cur = 0;

	const emit = () => {
		// 用 `tl.time()` 而不是补间插值出来的 `proxy.t`：前者更接近我们 seek 时设进去的数
		const t = tl.time();
		let idx = steps.length - 1;
		for (let i = 0; i < steps.length; i++) {
			if (t < ends[i] - EPS) {
				idx = i;
				break;
			}
		}
		cur = idx;
		const dur = (steps[idx] ? stepDurationMs(steps[idx]) : 1000) / 1000;
		const p = Math.min(1, Math.max(0, (t - starts[idx]) / dur));
		onUpdate({ stepIndex: idx, stepProgress: p, time: t, duration: total });
	};

	const tl = gsap.timeline({ paused: true, onUpdate: emit });
	tl.to(proxy, { t: total, duration: total, ease: 'none' });
	tl.eventCallback('onComplete', () => onPlayState?.(false));

	const atEnd = () => tl.time() >= total - EPS;

	const play = () => {
		if (atEnd()) {
			tl.time(0);
			emit();
		}
		tl.play();
		onPlayState?.(true);
	};

	const pause = () => {
		tl.pause();
		onPlayState?.(false);
	};

	return {
		play,
		pause,
		toggle() {
			if (tl.isActive()) pause();
			else play();
		},
		next() {
			tl.pause();
			tl.time(starts[Math.min(steps.length - 1, cur + 1)]);
		},
		prev() {
			tl.pause();
			tl.time(starts[Math.max(0, cur - 1)]);
		},
		seekRatio(r) {
			tl.pause();
			tl.time(Math.min(1, Math.max(0, r)) * total);
		},
		seekStep(i) {
			tl.pause();
			tl.time(starts[Math.min(steps.length - 1, Math.max(0, i))]);
		},
		seekStepRatio(i, r) {
			tl.pause();
			const idx = Math.min(steps.length - 1, Math.max(0, i));
			const dur = stepDurationMs(steps[idx]) / 1000;
			tl.time(starts[idx] + Math.min(1, Math.max(0, r)) * dur);
		},
		setSpeed(s) {
			tl.timeScale(s);
		},
		restart() {
			tl.pause();
			tl.timeScale(1);
			tl.time(0);
			emit();
		},
		destroy() {
			tl.kill();
		},
		get total() {
			return total;
		}
	};
}
