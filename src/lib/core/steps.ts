/**
 * 步骤的"动画分段"规则。
 *
 * `StepPanel`（渲染）与 `scripts/verify.ts`（校验）共用这一份，避免两边各写一套、悄悄漂移。
 */
import type { ConcatView, Step, SumView, TensorView, TransformView } from './types';

/** 本步里需要动画的视图（矩阵乘 / 掩码 / 拼接 / 变换 / 加权求和 / 查表 / 缓存结构 / 逐格揭示的矩阵与分布） */
export function isAnimatable(t: TensorView): boolean {
	return (
		t.kind === 'matmul' ||
		t.kind === 'mask' ||
		t.kind === 'concat' ||
		t.kind === 'transform' ||
		t.kind === 'sum' ||
		t.kind === 'lookup' ||
		t.kind === 'kvcache' ||
		(t.kind === 'matrix' && !!t.animated) ||
		(t.kind === 'bars' && !!t.animated)
	);
}

/**
 * 把本步的可动画视图切成"动画单元"，页级进度按单元数等分：
 *   - **并行行**（`parallel: true` 且 `row` 相同的多个视图）算**一个**单元 → 一起播；
 *   - 其余视图各自一个单元 → 按等分依次播。
 */
export function animationUnits(step: Step): TensorView[][] {
	const units: TensorView[][] = [];
	const byRow = new Map<number, TensorView[]>();
	for (const t of step.tensors) {
		if (!isAnimatable(t)) continue;
		if (t.row !== undefined && t.parallel) {
			const shared = byRow.get(t.row);
			if (shared) {
				shared.push(t);
				continue;
			}
			const group = [t];
			byRow.set(t.row, group);
			units.push(group);
			continue;
		}
		units.push([t]);
	}
	return units;
}

/**
 * 一个视图的"动画工作量"——要逐格 / 逐行揭示多少步。
 *
 * 用来让并行播放的多个视图**每格耗时一致**：两个专家分到的 token 数不同
 * （2 个 vs 6 个），输出格子数就差 3 倍，直接共用同一个进度会让小的那个快 3 倍填完。
 */
/** 一个形状有多少个格子 */
const cellsOf = (shape: number[]) => Math.max(1, shape[0] * (shape[1] ?? 1));

/**
 * 一个视图的"动画工作量"——**统一以"揭示了多少个矩阵格子"为单位**。
 *
 * 这一点很关键：`matmul` 数格子、`transform` 若数**行数**，单位就不一致，
 * "逐行揭示"的页（MoE 里那几步 softmax / SiLU / 相加）每格会快 2~3 倍，
 * 后面的 step 看着明显更快。
 */
/** 一个视图分成几"段"、每段要揭示多少格（并行播放时的进度分配、页面时长都基于它） */
export function phaseCells(t: TensorView): number[] {
	switch (t.kind) {
		case 'matmul':
			return [cellsOf(t.out.shape)];
		case 'mask':
			return [cellsOf([t.scores.length, t.scores[0]?.length ?? 0])];
		case 'concat':
			return t.then ? [cellsOf(t.result.shape), cellsOf(t.then.result.shape)] : [cellsOf(t.result.shape)];
		case 'transform': {
			const out: number[] = [];
			if (t.preMask) out.push(cellsOf(t.input.shape));
			// 逐行段：output 与"逐行接一段"的 then.result 是同时揭示的，算作一段
			out.push(cellsOf(t.output.shape) + (t.then && !t.then.b ? cellsOf(t.then.result.shape) : 0));
			if (t.then?.b) out.push(cellsOf(t.then.result.shape));
			// 第三段（`tail`）：接着 then.result 再做一次逐行变换（如逆 RoPE）
			if (t.tail) out.push(cellsOf(t.tail.result.shape));
			return out;
		}
		case 'sum':
			// 两段：先逐行算出 result，再逐行叠加追加项得到最终结果
			return t.then
				? [cellsOf(t.result.shape), cellsOf(t.then.result.shape)]
				: [cellsOf(t.result.shape)];
		case 'matrix':
			return [t.data.length * (t.data[0]?.length ?? 0)];
		case 'lookup':
			// 逐行取：一行 = 一个 key 的向量
			return [Math.max(1, t.rows.length)];
		case 'kvcache':
			// 逐个 token 揭示（缓存是一行一行填起来的）
			return [Math.max(1, t.blocks[0]?.data?.length ?? t.blocks.length)];
		case 'bars':
			return [Math.max(1, t.data.length)];
		default:
			return [1];
	}
}

/** 一个视图的动画总工作量（格子数） */
export function workOf(t: TensorView): number {
	return Math.max(1, phaseCells(t).reduce((a, b) => a + b, 0));
}

/** 每段最少播多久——格子很少的小矩阵乘不至于一闪而过 */
export const MIN_PHASE_MS = 2400;
/** 每格耗时 */
export const MS_PER_CELL = 60;

/** 一个视图的动画总时长：各段分别取 `max(MIN_PHASE_MS, 格数 × MS_PER_CELL)` 再相加 */
export function viewDurationMs(t: TensorView): number {
	return phaseCells(t).reduce(
		(sum, cells) => sum + Math.max(MIN_PHASE_MS, cells * MS_PER_CELL),
		0
	);
}

/**
 * 链式变换各段的**时长**（毫秒，已含每段下限）。
 *
 * 页级时长（`stepDurationMs`）与视图内部的时间分配（`TransformView` 的各段边界）
 * 共用这一份，避免两边算法漂移。
 *
 * 注意：`phaseCells` 在"没有掩码段"时会少一项，所以这里必须**按段是否存在**依次取，
 * 不能按下标硬套——无掩码的变换（SiLU、softmax→top-k）会把"逐行段"当成"掩码段"，
 * `rows` 变成 0，逐行揭示被判成"已完成"，动画一上来就是终态。
 * 链尾多一段（`tail`）同理：有没有它决定 `cells` 末尾有没有第四项。
 */
export function transformPhaseMs(t: TransformView): {
	mask: number;
	rows: number;
	mm: number;
	tail: number;
} {
	const cells = phaseCells(t);
	let i = 0;
	const mask = t.preMask ? msOf(cells[i++]) : 0;
	const rows = msOf(cells[i++]);
	const mm = t.then?.b ? msOf(cells[i++]) : 0;
	const tail = t.tail ? msOf(cells[i++]) : 0;
	return { mask, rows, mm, tail };
}

/** 链式相加各段的时长：加权求和段 + 可选的"叠加"段 */
export function sumPhaseMs(t: SumView): { sum: number; add: number } {
	const cells = phaseCells(t);
	return { sum: msOf(cells[0] ?? 0), add: t.then ? msOf(cells[1] ?? 0) : 0 };
}

/** 链式拼接各段的时长：拼接段 + 可选的矩阵乘段 */
export function concatPhaseMs(t: ConcatView): { cols: number; mm: number } {
	const [cols = 0, mm = 0] = phaseCells(t);
	return { cols: msOf(cols), mm: msOf(mm) };
}

/** 一段的时长：至少 MIN_PHASE_MS，再按格数加 */
function msOf(cells: number): number {
	return cells === 0 ? 0 : Math.max(MIN_PHASE_MS, cells * MS_PER_CELL);
}

/**
 * 整页进度 `eff`（0..1）下，某个视图的实际进度。
 *
 * - 按"动画单元"切分：第 i 个单元在 `[i/M, (i+1)/M]` 区间跑完 0→1。
 * - **并行单元**里各视图工作量不同时，按工作量**反比**缩放进度，保证每格耗时一样；
 *   代价是工作量小的那个先完成、然后空转等大的那个（这是有意的：
 *   "两个专家同时开始算，2 个 token 的那个先算完"）。
 */
export function progressOf(t: TensorView, units: TensorView[][], eff: number): number {
	const i = units.findIndex((u) => u.includes(t));
	if (i < 0) return 0;
	const unit = units[i];
	const raw =
		units.length <= 1 ? eff : Math.min(1, Math.max(0, eff * units.length - i));
	if (unit.length <= 1) return raw;
	const maxWork = Math.max(...unit.map(workOf));
	return Math.min(1, (raw * maxWork) / workOf(t));
}

// ── 一步该播多久：由各段的工作量算出来，而不是插件手填 ──────────
/**
 * 一步的"动画工作量"：并行单元里同时播，所以取单元内的最大值再相加
 * （而不是把所有视图的工作量加起来——那样并行的那几个会被重复计算）。
 */
export function stepWork(step: Step): number {
	return animationUnits(step).reduce((sum, unit) => sum + Math.max(...unit.map(workOf)), 0);
}

/**
 * 一步在时间轴上的时长（毫秒）。
 *
 * 规则：**每个"动画段"至少 MIN_PHASE_MS，再按格数加时间**——
 * 只按格数线性算的话，格子少的页（`moe-route`）会比格子多的页（`attn-q`）
 * 快一倍，页与页之间忽快忽慢；加了下限之后两者接近，而格子明显更多的链式页
 * 仍然按格数拿到更长时间——毕竟要揭示的格子确实多一倍。
 *
 * 页级播放（`StepPanel`）与底部总播放器（`createPlayer`）都用这个值，两边速度一致。
 */
export function stepDurationMs(step: Step): number {
	return animationUnits(step).reduce(
		(sum, unit) => sum + Math.max(...unit.map(viewDurationMs)),
		0
	);
}

// ── 矩阵格子尺寸：渲染与校验共用 ────────────────────────────
/**
 * 格子小到写不下数字的阈值。
 *
 * 本项目的第一硬约束是"**每个矩阵都能逐格显示真实数字**"（`d_model = 8`、
 * `vocab_size = 12` 都是为它让的路）。所以"还能不能显示数字"这件事
 * 必须只有一份判定：`MatmulView` 用它排版，`verify.ts` 用它断言。
 */
export const MIN_TEXT_CELL = 20;

/**
 * 矩阵乘视图的格子边长。
 *
 * 宽度预算按"矩阵乘要能并排放下 A/B/C"取（横向并排时更紧），
 * 再夹在 `[15, compact ? 22 : 34]` 之间。
 */
export function matmulCellSize(maxDim: number, compact: boolean): number {
	const budget = compact ? 200 : 290;
	const cap = compact ? 22 : 34;
	return Math.max(15, Math.min(cap, Math.floor(budget / Math.max(maxDim, 1))));
}

/**
 * 变换视图的格子边长。
 *
 * 默认 34（数字读得清）。但**链尾还接了一段**（`tail`，如逆 RoPE）时，同一条链上会多出
 * "中段箭头 + 末块"两块矩阵，整行会顶出容器；于是按列数反推一个放得下的尺寸。
 * 下限仍是 `MIN_TEXT_CELL`——"每个矩阵都能逐格显示真实数字"这条硬约束不能因为
 * 链变长就破掉（`verify` 对全部变换视图都断言这一点）。
 */
export function transformCellSize(cols: number, compact: boolean, hasTail = false): number {
	if (compact) return 24;
	if (!hasTail) return 34;
	return Math.max(MIN_TEXT_CELL, Math.min(24, Math.floor(290 / Math.max(1, cols))));
}
