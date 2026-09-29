/**
 * DeepSeek V4 混合注意力的**参考数据流图**——`swa` / `csa` / `hca` 三层共用一张。
 *
 * 三种层型只差"有没有那条支路"：
 *   - 纯滑窗层没有压缩支路（`W_gate` → 组内加权平均 → 压缩 KV），也没有 indexer；
 *   - HCA 有压缩支路、没有 indexer（条目少，全看就行）；
 *   - CSA 两样都有。
 *
 * 所以这里**只画一张图**，缺的那条支路按 `gone` + 淡显处理——切换层型时
 * "同一张图变了颜色"，差在哪一眼可见（与 MLA 的吸收合并版同一套做法）。
 *
 * 位置是**手写**的（`col` / `row`）：这张图我比自动布局更清楚线该怎么绕。
 * 主链自左向右：`X → Q/KV → 压缩 → 一次注意力 → 逆 RoPE → 输出`，缓存挂在下面。
 */
import type { DataflowEdge, DataflowNode, DataflowSpec } from '$lib/core/types';

/** 三种层型 */
export type V4Variant = 'swa' | 'csa' | 'hca';

const NODES: DataflowNode[] = [
	{ id: 'x', label: 'X', sub: '[16 × 8] 输入', col: 0, row: 1, tone: 'input' },

	// ── Q 链：低秩下投影 → 归一化 → 上投影 → 每头 RMSNorm → RoPE ──
	{ id: 'cq', label: 'c_Q', sub: '[16 × 4] 低秩', col: 1, row: 0, tone: 'latent' },
	{ id: 'cqn', label: 'c_Q~', sub: 'RMSNorm', col: 2, row: 0, tone: 'latent' },
	{ id: 'q', label: 'q', sub: '2 头 · 每头 2 + 2', col: 3, row: 0, tone: 'head' },

	// ── KV 链：只有一个头，而且 K = V ──
	{ id: 'kvraw', label: 'k_raw', sub: '[16 × 4] 单头', col: 1, row: 2, tone: 'latent' },
	{ id: 'kvnorm', label: 'k~', sub: 'RMSNorm', col: 2, row: 2, tone: 'latent' },
	{ id: 'kv', label: 'k = v', sub: '键即值', col: 3, row: 2, tone: 'head' },

	// ── 压缩支路：gate 投影 → 组内逐通道 softmax → 加权平均 → 压缩 KV ──
	{ id: 'gate', label: '[v | s]', sub: 'W_gate 投影', col: 1, row: 3, tone: 'latent' },
	{ id: 'w', label: 'w', sub: '按列 softmax', col: 2, row: 3, tone: 'latent' },
	{ id: 'entries', label: 'v̄', sub: '压缩条目 ×4', col: 3, row: 3, tone: 'latent' },
	{ id: 'ckv', label: '压缩 KV', sub: 'K = V', col: 4, row: 3, tone: 'cache' },

	// ── indexer：用自己那套更窄的压缩 KV 打分、挑 top-k ──
	{ id: 'itop', label: 'top-k', sub: '挑 2 条', col: 4, row: 2, tone: 'latent' },

	// ── 分数（真正的矩阵乘）→ 掩码 + softmax + P·V → 逆 RoPE → 输出 ──
	{ id: 'score', label: 'S', sub: 'Q·Kᵀ/√d', col: 4, row: 0, tone: 'result' },
	{ id: 'sink', label: 'sink', sub: '分母上的一项 · 无位置', col: 4, row: 1, tone: 'weight' },
	{ id: 'attn', label: '注意力', sub: '掩码 + softmax + P·V', col: 5, row: 1, tone: 'result' },
	{ id: 'inv', label: '逆 RoPE', sub: '→ 相对量', col: 6, row: 1, tone: 'rope' },
	{ id: 'out', label: 'u', sub: '分组低秩', col: 7, row: 1, tone: 'result' },

	{ id: 'cache', label: 'KV Cache', sub: '滑窗 + 压缩', col: 5, row: 4, tone: 'cache' }
];

const EDGES: DataflowEdge[] = [
	{ from: 'x', to: 'cq', label: 'W_Q^A' },
	{ from: 'x', to: 'kvraw', label: 'W_KV' },
	{ from: 'cq', to: 'cqn', label: 'RMSNorm' },
	{ from: 'cqn', to: 'q', label: 'W_Q^B+RoPE' },
	{ from: 'kvraw', to: 'kvnorm', label: 'RMSNorm' },
	{ from: 'kvnorm', to: 'kv', label: 'RoPE' },

	// 压缩支路（`x → gate` 从最左边一路绕下去，所以给它一个偏移量）
	{ from: 'x', to: 'gate', label: 'W_gate', routeOffset: 34 },
	{ from: 'gate', to: 'w', label: 'softmax + ape' },
	{ from: 'gate', to: 'entries', label: '值', routeOffset: 30 },
	{ from: 'w', to: 'entries', label: '权重' },
	{ from: 'entries', to: 'ckv', label: 'RMSNorm + RoPE' },

	// indexer（`c_Q~` 要跨过 `q` 那一行，给它一个偏移量）
	{ from: 'cqn', to: 'itop', label: 'W_q^I + k^I', routeOffset: 44 },

	// 分数：真实列（滑窗 token + 压缩条目）汇进来做一次矩阵乘；sink **不进**这个乘
	{ from: 'q', to: 'score' },
	{ from: 'kv', to: 'score', routeOffset: 20 },
	{ from: 'ckv', to: 'score', label: '压缩条目', routeOffset: 44 },
	{ from: 'itop', to: 'score', label: '选中的', routeOffset: 30 },

	// 一次注意力：分数 + sink 列 → 掩码 → softmax → P·V
	{ from: 'score', to: 'attn' },
	{ from: 'sink', to: 'attn', routeOffset: 24 },
	{ from: 'attn', to: 'inv' },
	{ from: 'inv', to: 'out' },

	// 进缓存的两块
	{ from: 'kv', to: 'cache', label: '滑窗', routeOffset: 30 },
	{ from: 'ckv', to: 'cache', label: '压缩', routeOffset: 46 }
];

/** 这一层没有的节点 → 灰色淡显（与 MLA 吸收合并版同一套标注语言） */
const GONE: Record<V4Variant, Record<string, string>> = {
	swa: {
		gate: '纯滑窗层没有压缩支路',
		w: '纯滑窗层没有压缩支路',
		entries: '纯滑窗层没有压缩支路',
		ckv: '纯滑窗层没有压缩支路',
		itop: '纯滑窗层没有 indexer'
	},
	csa: {},
	hca: { itop: 'HCA 没有 indexer（条目少，全看就行）' }
};

const HINTS: Record<V4Variant, string> = {
	swa: '实线 = 本层真正走的路径。灰色虚线 + 淡显 = 这一层**没有**的支路：压缩（`W_gate` → 组内加权平均 → 压缩 KV）与 indexer。所以纯滑窗层只看得到最近那 W 个 token（`swa_window`），更早的历史彻底丢了——翻到 C4 / C128 那两层对比最直观。',
	csa: '实线 = 本层真正走的路径。三条输入主线（Q、KV、压缩）汇到**同一个** softmax，所以滑窗 KV 与压缩 KV 不是"各算一遍再合并"。`top-k` 那一步决定哪些压缩条目能进注意力——被拒的那些在分数矩阵里是 −∞。',
	hca: '实线 = 本层真正走的路径。灰色淡显的 `top-k` 是 HCA **没有**的那一步：压缩比大、条目本来就少，全部直接进 softmax，省掉"打分 + 挑选"这套开销。'
};

const goneIdsOf = (v: V4Variant) => new Set(Object.keys(GONE[v]));

/**
 * 每种层型的图**只构造一次**并缓存。
 *
 * 同一层的所有步骤必须共用同一对 `nodes` / `edges` 对象（与 MLA 的做法一致）：
 * 组件据此判断"要不要重画"，`verify` 也据此断言"各步共用同一张图"。
 */
const CACHE = new Map<V4Variant, { nodes: DataflowNode[]; edges: DataflowEdge[] }>();

function build(variant: V4Variant) {
	const hit = CACHE.get(variant);
	if (hit) return hit;
	const gone = goneIdsOf(variant);
	const nodes = NODES.map((n) =>
		gone.has(n.id) ? { ...n, tone: 'gone' as const, faded: true, sub: GONE[variant][n.id] } : n
	);
	const edges = EDGES.map((e) =>
		gone.has(e.from) || gone.has(e.to) ? { ...e, tone: 'gone' as const, faded: true } : e
	);
	const built = { nodes, edges };
	CACHE.set(variant, built);
	return built;
}

/**
 * 取这张图、把本步正在算的节点标出来。
 *
 * 高亮里**自动滤掉本层没有的节点**——同一步的 active 列表对三种层型是同一份，
 * 纯滑窗层不该出现"高亮着一条不存在的支路"。
 */
export function v4Diagram(variant: V4Variant, active: string[]): DataflowSpec {
	const gone = goneIdsOf(variant);
	const { nodes, edges } = build(variant);
	return { nodes, edges, active: active.filter((id) => !gone.has(id)), hint: HINTS[variant] };
}
