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
		t.kind === 'ewise' ||
		t.kind === 'mask' ||
		t.kind === 'concat' ||
		t.kind === 'transform' ||
		t.kind === 'sum' ||
		t.kind === 'lookup' ||
		t.kind === 'kvcache' ||
		(t.kind === 'matrix' && !!t.animated) ||
		(t.kind === 'row' && !!t.animated) ||
		(t.kind === 'bars' && !!t.animated)
	);
}

/**
 * 把本步的可动画视图切成"动画单元"，页级进度按单元数等分：
 *   - **同一行**且标了 `parallel`（并行分支）或 `sync`（必须同步）的多个视图算**一个**单元
 *     → 一起播；区别在 `progressOf` 里（`parallel` 按工作量反比缩放、`sync` 用同一个进度）；
 *   - 其余视图各自一个单元 → 按等分依次播。
 */
export function animationUnits(step: Step): TensorView[][] {
	const units: TensorView[][] = [];
	const byRow = new Map<number, TensorView[]>();
	for (const t of step.tensors) {
		if (!isAnimatable(t)) continue;
		if (t.row !== undefined && (t.parallel || t.sync)) {
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
		case 'ewise':
			// 逐组揭示时按"组的一列"计数：一次出 `groupRows` 个格子（`w ⊙ v` 那一步
			// 一组 4 个 token 的同一列是一起算的），动画单位是**组**而不是格——按格数算
			// 会把时长压死在 `MIN_PHASE_MS` 上（32 格 8 组：8×60ms ≪ 下限）。
			return t.groupRows ? [Math.ceil(cellsOf(t.out.shape) / t.groupRows)] : [cellsOf(t.out.shape)];
		case 'mask':
			return [cellsOf([t.scores.length, t.scores[0]?.length ?? 0])];
		case 'concat':
			return t.then ? [cellsOf(t.result.shape), cellsOf(t.then.result.shape)] : [cellsOf(t.result.shape)];
		case 'transform': {
			const out: number[] = [];
			if (t.preMask) out.push(cellsOf(t.input.shape));
			// 逐行段：`output` 一段
			out.push(cellsOf(t.output.shape));
			// 第二段是**逐行**（没有 `b`）时**另起一段**：`X ──op1──▶ Y ──op2──▶ Z` 里
			// Y 和 Z 是**先后**算出来的（先算出 Y 才谈得上对它做 op2），不能挤在同一段里
			// 一起揭示。
			if (t.then && !t.then.b) out.push(cellsOf(t.then.result.shape));
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
		case 'row':
			// 逐个元素揭示（`v̄` 那样"左边每列算完、这里出一个数"）
			return [Math.max(1, t.data.length)];
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

/** 每段最少播多久——格子很少的小矩阵乘不至于一闪而过。 */
export const MIN_PHASE_MS = 1200;
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
	/** 第二段是逐行（`then` 无 `b`）时，它自己那一段 */
	rows2: number;
	mm: number;
	tail: number;
} {
	const cells = phaseCells(t);
	let i = 0;
	const mask = t.preMask ? msOf(cells[i++]) : 0;
	const rows = msOf(cells[i++]);
	const rows2 = t.then && !t.then.b ? msOf(cells[i++]) : 0;
	const mm = t.then?.b ? msOf(cells[i++]) : 0;
	const tail = t.tail ? msOf(cells[i++]) : 0;
	return { mask, rows, rows2, mm, tail };
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
 * - **`sync` 单元**里所有视图用**同一个**进度（不做反比缩放）：这样"第 c 列算完"
 *   和"第 c 个数出现"才会严格对上（组内加权平均那一步）。
 */
export function progressOf(t: TensorView, units: TensorView[][], eff: number): number {
	const i = units.findIndex((u) => u.includes(t));
	if (i < 0) return 0;
	const unit = units[i];
	const raw =
		units.length <= 1 ? eff : Math.min(1, Math.max(0, eff * units.length - i));
	if (unit.length <= 1) return raw;
	if (unit.some((v) => v.sync)) return raw;
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
	// **一步再短也不能是 0**：时间轴是按"每步终点"切段的，某一步时长为 0 会让它和上一步
	// 的终点**落在同一个数上**，于是 `t < ends[i]` 永远不成立——那一步就再也跳不进去、
	// 后退也退不回来。
	return Math.max(
		MIN_PHASE_MS,
		animationUnits(step).reduce((sum, unit) => sum + Math.max(...unit.map(viewDurationMs)), 0)
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
 * 预算按"**A / B / C 三块并排的总列数**"取——不是按单块最宽的那一块。
 * 三个矩阵是左右排着的，决定整行宽度的本来就是 `a.cols + b.cols + out.cols`。
 *
 * 为什么必须按总列数算：`compressor` 的 gate 投影输出是 `[v | s]`（重叠窗口下 2·coff·head_dim = 16 列），
 * 只看 `max(out.cols, a.cols) = 16` 会算出 18px，**掉到 `MIN_TEXT_CELL` 以下**，
 * 那一页只能退化成"画个 transform、把 × W_gate 写在箭头里"——矩阵根本没画出来。
 * 按总列数算，A/B/C 的宽度都算进预算，16 列的输出也能拿到 22px。
 */
export function matmulCellSize(totalCols: number, compact: boolean): number {
	const budget = compact ? 700 : 900;
	const cap = compact ? 22 : 34;
	return Math.max(MIN_TEXT_CELL, Math.min(cap, Math.floor(budget / Math.max(totalCols, 1))));
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
	// 并排（compact）时用 22：和 `matmulCellSize` 的 compact 上限一致——
	// 并排的两列里一边是矩阵乘、一边是变换时，格子一样大，行高才一样、矩阵才能对齐
	if (compact) return 22;
	if (!hasTail) return 34;
	return Math.max(MIN_TEXT_CELL, Math.min(24, Math.floor(290 / Math.max(1, cols))));
}

/**
 * `fillWidth` 那一行放大之后，格子边长的上限。
 *
 * 上面两个函数给的是**默认**边长（一行装得下就行）；标了 `fillWidth` 的行会
 * 由 `StepPanel` 量出可用宽度再放大，这个常量是"别放大过头"的刹车：
 * 再大就"字小格子空"（`MatrixGrid` 的字号本身也有上限），
 * 而且矩阵的高度 = 行数 × 边长，8 行的矩阵边长 88px 就是 704px 高。
 */
export const MAX_TEXT_CELL = 88;

/**
 * 把格子边长缩放到"**这一行的内容刚好占满可用宽度**"（只给 `fillWidth` 的行用）。
 *
 * 一行内容的宽度 ≈ 固定开销（块内边距、块间距、算子名与箭头、注解与算式框的最小宽度）
 *  + 格子列数 × 边长。固定开销**算不准**（注解、标签、算式框的 `max-content` 都掺在里面），
 * 所以干脆不把它单独拆出来，只用**比例**迭代：
 *
 *   `next = applied × avail / content`
 *
 * 代入 `content = fixed + applied × N` 可知不动点正是 `(avail − fixed) / N`，
 * 且单调收敛（`StepPanel` 每次量完再渲染，一两轮就稳定）。
 * 下限仍是 `MIN_TEXT_CELL`——"每个矩阵都能逐格显示真实数字"这条硬约束不能破。
 */
export function fitCellSize(applied: number, content: number, avail: number): number {
	if (!(applied > 0) || !(content > 0) || !(avail > 0)) return applied;
	const next = Math.floor((applied * avail) / content);
	return Math.max(MIN_TEXT_CELL, Math.min(MAX_TEXT_CELL, next));
}
