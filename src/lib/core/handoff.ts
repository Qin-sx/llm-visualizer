/**
 * 跨步骤的"矩阵流转"动画。
 *
 * 问题：步骤之间是断开的——`attn-score` 用的 `Q_h` 其实就是 `attn-q` 算出来的 `Q`，
 * 但切换步骤后看不出这层关系。
 *
 * 做法：给矩阵块标上 `data-handoff="q"` 这样的稳定 id（来自 `MatRef.handoffId`）。
 * 每次步骤切换后，把当前所有带 id 的块的位置记下来；下次某个 id 再出现时，
 * 就生成一个半透明"幽灵"从旧位置飞到新位置。
 *
 * 关键点：位置表是**持久**的，不是只记上一步——这样即使中间隔了几步
 * （比如 Q 在 `attn-q` 算出、到 `attn-score` 才用到），也能正确衔接。
 */
import { gsap } from 'gsap';

/** id → 上一次出现时的位置 */
const lastRects = new Map<string, DOMRect>();

const MIN_MOVE = 6; // 位移小于这个像素数就不做动画（避免原地重绘时乱飞）

function flyGhost(from: DOMRect, to: DOMRect, text: string) {
	const ghost = document.createElement('div');
	ghost.textContent = text;
	ghost.style.cssText = [
		'position:fixed',
		'z-index:9999',
		'pointer-events:none',
		'box-sizing:border-box',
		'border:2px solid #4f46e5',
		'background:rgba(79,70,229,0.14)',
		'border-radius:8px',
		'display:flex',
		'align-items:center',
		'justify-content:center',
		'font-family:ui-monospace,SFMono-Regular,Menlo,monospace',
		'font-weight:700',
		'font-size:0.9rem',
		'color:#3730a3',
		'box-shadow:0 8px 24px rgba(79,70,229,0.25)'
	].join(';');
	document.body.appendChild(ghost);

	gsap.fromTo(
		ghost,
		{ left: from.left, top: from.top, width: from.width, height: from.height, opacity: 0 },
		{
			left: to.left,
			top: to.top,
			width: to.width,
			height: to.height,
			opacity: 0.85,
			duration: 0.45,
			ease: 'power2.out',
			onComplete: () => {
				gsap.to(ghost, {
					opacity: 0,
					duration: 0.35,
					ease: 'power2.in',
					onComplete: () => ghost.remove()
				});
			}
		}
	);
}

/**
 * 采集当前 DOM 里所有 `[data-handoff]` 块的位置，并对"位置变了"的那些做飞行动画。
 * 应在**步骤切换后**（DOM 已更新）调用。
 */
export function animateHandoffs() {
	if (typeof document === 'undefined') return;

	const current = new Map<string, DOMRect & { name: string }>();
	document.querySelectorAll<HTMLElement>('[data-handoff]').forEach((el) => {
		const id = el.dataset.handoff;
		if (!id) return;
		const rect = el.getBoundingClientRect();
		if (rect.width === 0) return;
		current.set(id, Object.assign(rect, { name: el.dataset.handoffName ?? id.toUpperCase() }));
	});

	for (const [id, to] of current) {
		const from = lastRects.get(id);
		if (!from) continue;
		const moved =
			Math.abs(from.left - to.left) > MIN_MOVE || Math.abs(from.top - to.top) > MIN_MOVE;
		if (moved) flyGhost(from, to, to.name);
	}

	for (const [id, rect] of current) lastRects.set(id, rect);
}

/** 重置（切换模型/插件时调用，避免跨上下文乱飞） */
export function resetHandoffs() {
	lastRects.clear();
}
