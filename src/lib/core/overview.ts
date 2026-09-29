/**
 * "流程总览"：把整个 attention + MoE 拆成两类小组，横向串成一条连贯的流。
 *
 *   - **矩阵乘组**：左操作数在左下、右操作数在上、结果在右下
 *   - **单目运算组**：输入在左、输出在右（如 `S_h → P_h`，掩码 + softmax）
 *
 * 同一个矩阵在多个小组里出现时，用曲线把相邻两次出现连起来，
 * 于是"上一步的 Q 就是这一步的 Q"一眼可见。
 *
 * 数据完全从 `Step[]` 推导（只需给单目运算的产物标一个 `overview` 显示名）。
 */
import type { Mat, MatRef, Step, TensorView } from './types';
import { isAnimatable } from './steps';

export interface MatmulItem {
	kind: 'matmul';
	id: string;
	stepId: string;
	/** 左操作数（左下） */
	a: MatRef;
	/** 右操作数（右上） */
	b: MatRef;
	/** 结果（右下） */
	out: MatRef;
	/** 前置运算名，如 ['掩码', 'softmax']（标在节点上方） */
	ops?: string[];
	/** 并行分支个数（>1 时标 ×N） */
	parallel?: number;
}

export interface UnaryItem {
	kind: 'unary';
	id: string;
	stepId: string;
	/** 输入（左） */
	input: MatRef;
	/** 输出（右） */
	output: MatRef;
	/** 运算名，如 ['掩码', 'softmax'] */
	ops: string[];
	/** 并行分支个数（>1 时标 ×N） */
	parallel?: number;
}

export type OverviewItem = MatmulItem | UnaryItem;

/** 某个节点里出现的所有矩阵 key（用于高亮判断与连线） */
export function itemKeys(it: OverviewItem): string[] {
	return it.kind === 'matmul'
		? [refKey(it.a), refKey(it.b), refKey(it.out)]
		: [refKey(it.input), refKey(it.output)];
}

/** 统一 key：跨步骤复用的矩阵用 handoffId，其余用名字 */
export function refKey(ref: { name: string; handoffId?: string }): string {
	return ref.handoffId ?? ref.name;
}

/**
 * 从 `Step[]` 推导总览节点——**一步一个节点**。
 *
 * "一个矩阵乘 / 一个单目运算各出一个节点"会让节点数超过步骤数、看着比详情页还碎，
 * 所以一步只出一个节点：
 *   - 这一步最后一段是矩阵乘（含链式变换 / 链式拼接的第二段）→ 出**矩阵乘节点**；
 *   - 否则出**单目节点**；
 *   - 前面几段的运算名挂在节点的 `ops` 上（如 `掩码 → softmax` 标在 `O_h = P_h·V_h` 上方）；
 *   - 并行分支（一个 token 同时过多个专家）只画第一个 + `×N` 角标。
 */
export function buildOverviewItems(steps: Step[]): OverviewItem[] {
	const items: OverviewItem[] = [];

	for (const step of steps) {
		const anim = step.tensors.filter(isAnimatableView);
		if (!anim.length) continue;

		const ops: string[] = [];
		/** 最后一段的"形状"：矩阵乘 or 单目 */
		let node: OverviewItem | null = null;
		/** 并行分支个数 */
		let parallel = 1;

		for (const t of anim) {
			// 并行行（一个 token 同时过多个专家）：只让第一个视图出节点，其余只计数
			if (t.parallel && node) {
				parallel++;
				continue;
			}
			switch (t.kind) {
				case 'matmul': {
					node = { kind: 'matmul', id: `${step.id}:${t.name}`, stepId: step.id, a: t.a, b: t.b, out: t.out };
					break;
				}
				case 'transform': {
					if (t.preMask) ops.push('掩码');
					ops.push(t.op);
					if (t.then?.b) {
						node = { kind: 'matmul', id: `${step.id}:${t.name}:then`, stepId: step.id, a: t.output, b: t.then.b, out: t.then.result };
						// 链尾还有一段逐行变换（`tail`，如逆 RoPE）：节点仍是这次矩阵乘，
						// 只把它的名字带上，免得总览里"这一步干了什么"漏掉半截
						if (t.tail) ops.push(t.tail.op);
					} else if (t.then) {
						ops.push(t.then.op);
						if (t.tail) ops.push(t.tail.op);
						// 链尾有 `tail` 时，整条链的**最终产物**是 tail.result
						node = { kind: 'unary', id: `${step.id}:${t.name}`, stepId: step.id, input: t.input, output: t.tail?.result ?? t.then.result, ops: [...ops] };
					} else {
						node = { kind: 'unary', id: `${step.id}:${t.name}`, stepId: step.id, input: t.input, output: t.output, ops: [...ops] };
					}
					break;
				}
				case 'concat': {
					ops.push('拼接');
					if (t.then) {
						node = { kind: 'matmul', id: `${step.id}:${t.name}:then`, stepId: step.id, a: t.result, b: t.then.b, out: t.then.result };
					} else {
						const first = t.parts[0];
						node = first
							? { kind: 'unary', id: `${step.id}:${t.name}`, stepId: step.id, input: first, output: t.result, ops: [...ops] }
							: null;
					}
					break;
				}
				case 'sum': {
					ops.push('加权求和');
					if (t.then) ops.push('叠加共享专家');
					const first = t.terms[0];
					if (first) {
						node = {
							kind: 'unary',
							id: `${step.id}:${t.name}`,
							stepId: step.id,
							input: first,
							output: t.then?.result ?? t.result,
							ops: [...ops]
						};
					}
					break;
				}
				case 'lookup': {
					ops.push('查表');
					node = {
						kind: 'unary',
						id: `${step.id}:${t.name}`,
						stepId: step.id,
						input: t.table,
						output: t.result,
						ops: [...ops]
					};
					break;
				}
				case 'kvcache':
					// 缓存结构不是一次运算，总览里不画节点（它在步骤列表里自有位置）
					break;
				default:
					break;
			}
		}

		if (!node) continue;
		// 单目节点的 ops 是它自己的（已含运算名）；矩阵乘节点的 ops 是"前置运算"
		const item: OverviewItem =
			node.kind === 'matmul' ? { ...node, ops: [...ops] } : node;
		items.push(parallel > 1 ? { ...item, parallel } : item);
	}

	return items;
}

/** 总览节点上挂的额外信息（前置运算名 / 并行个数） */
export interface OverviewItemExtra {
	/** 前置或本步的运算名，如 ['掩码', 'softmax']；矩阵乘节点上表示"前置运算" */
	ops: string[];
	/** 并行分支个数（>1 时界面上标 ×N） */
	parallel?: number;
}

/** 只认识"需要动画"的视图——和渲染层同一套判定（`core/steps.ts`） */
function isAnimatableView(t: TensorView): boolean {
	return isAnimatable(t);
}

/**
 * 一个视图里所有"带 label 的矩阵引用"。
 *
 * 不同视图类型的挂载位置不同（矩阵乘是 a/b/out，变换是 input/output，拼接是 parts/result），
 * 集中在这里列出——新增视图类型时只要补一个 case，步骤号解析就不会漏。
 */
export function matRefsOf(t: TensorView): MatRef[] {
	switch (t.kind) {
		case 'matmul':
			return [t.a, t.b, t.out];
		case 'transform':
			return [
				t.input,
				t.output,
				...(t.then ? [...(t.then.b ? [t.then.b] : []), t.then.result] : []),
				...(t.tail ? [t.tail.result] : [])
			];
		case 'concat':
			return [...t.parts, t.result, ...(t.then ? [t.then.b, t.then.result] : [])];
		case 'sum':
			return [...t.terms, t.result, ...(t.then ? [...t.then.terms, t.then.result] : [])];
		case 'lookup':
			return [t.table, t.result];
		default:
			// `kvcache` 等没有矩阵引用的视图：不用解析步骤号占位符
			return [];
	}
}