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

/** 三种 V4 层型 + 三种 V4.1 CSA2 角色 */
export type V4Variant = 'swa' | 'csa' | 'hca' | 'v41-full' | 'v41-reuse' | 'v41-reindex';

/** V4.1 CSA2 角色：是否真的属于 V4.1 的共享层（用于来源标注） */
function isV41(v: V4Variant): boolean {
	return v === 'v41-full' || v === 'v41-reuse' || v === 'v41-reindex';
}

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
	hca: { itop: 'HCA 没有 indexer（条目少，全看就行）' },
	// V4.1 CSA2：Full 层自己压缩 + 自己跑 indexer（与 csa 同，只是提示换成 CSA2 口径）
	'v41-full': {},
	// V4.1 Reuse：不压缩、不重选——压缩支路与 indexer 都读共享池（来源层号由 `src` 动态填）
	'v41-reuse': {
		gate: '本层不压缩，直接读',
		w: '本层不压缩，直接读',
		entries: '本层不压缩，直接读',
		ckv: '本层不压缩，直接读',
		itop: '本层不重选，复用'
	},
	// V4.1 Reindex：复用压缩 KV，但自己跑 indexer（池内重打分）
	'v41-reindex': {
		gate: '本层不压缩，直接读',
		w: '本层不压缩，直接读',
		entries: '本层不压缩，直接读',
		ckv: '本层不压缩，直接读'
	}
};

const HINTS: Record<V4Variant, string> = {
	swa: '实线 = 本层真正走的路径。灰色虚线 + 淡显 = 这一层**没有**的支路：压缩（`W_gate` → 组内加权平均 → 压缩 KV）与 indexer。所以纯滑窗层只看得到最近那 W 个 token（`swa_window`），更早的历史彻底丢了——翻到 C4 / C128 那两层对比最直观。',
	csa: '实线 = 本层真正走的路径。三条输入主线（Q、KV、压缩）汇到**同一个** softmax，所以滑窗 KV 与压缩 KV 不是"各算一遍再合并"。`top-k` 那一步决定哪些压缩条目能进注意力——被拒的那些在分数矩阵里是 ∅。',
	hca: '实线 = 本层真正走的路径。灰色淡显的 `top-k` 是 HCA **没有**的那一步：压缩比大、条目本来就少，全部直接进 softmax，省掉"打分 + 挑选"这套开销。',
	'v41-full': '实线 = 本层真正走的路径。这是 V4.1 的 **Full 层（源层）**：自己压缩出 KV、自己跑 indexer，并把压缩 KV / top-k **写进共享池**，供后面的 Reuse / Reindex 层直接读。',
	'v41-reuse': '实线 = 本层真正走的路径。**天蓝虚线 + 实底**的压缩支路与 `top-k` = **共享矩阵**：本层**不自己算**，压缩 KV 与选择结果都读共享池（来源层号见节点标注），本层只做 Q 与一次注意力。',
	'v41-reindex': '实线 = 本层真正走的路径。**天蓝虚线 + 实底**的压缩支路 = 本层**复用共享池里的 KV**（来源层号见标注）；但 `top-k` 是实线——本层**用自己的 Q 在候选池里重打分**（Reindex），不沿用源层的选择。'
};

const goneIdsOf = (v: V4Variant) => new Set(Object.keys(GONE[v]));

/**
 * V4.1 CSA2 角色里"本层不压缩"的说明，按来源层号补全。
 * Reuse / Reindex 层读共享池，ckv 节点的淡显说明要写清"来自第 N 层"。
 */
function v41Sub(id: string, variant: V4Variant, src?: number, srcAction?: string): string | undefined {
	const text = GONE[variant][id];
	if (text === undefined) return undefined;
	if (id === 'ckv' && src !== undefined)
		return `← 第 ${src} 层${srcAction ?? '压缩'}（本层不压缩）`;
	return text;
}

/**
 * 每种层型的图**只构造一次**并缓存。
 *
 * 同一层的所有步骤必须共用同一对 `nodes` / `edges` 对象（与 MLA 的做法一致）：
 * 组件据此判断"要不要重画"，`verify` 也据此断言"各步共用同一张图"。
 * 注意：V4.1 的 Reuse / Reindex 层**来源层号不同**（如层 3 读层 2、层 7 读层 5），
 * 而图的缓存是按 variant 命名的——来源层号不参与缓存，靠调用侧保证"同一步同一来源"。
 */
const CACHE = new Map<V4Variant, { nodes: DataflowNode[]; edges: DataflowEdge[] }>();

function build(variant: V4Variant, src?: number, srcAction?: string) {
	const hit = CACHE.get(variant);
	if (hit && src === undefined) return hit;
	const gone = goneIdsOf(variant);
	// V4.1 CSA2：压缩支路是"本层不计算、但读共享池"——**存在且在使用**（来源在别处），
	// 用 `shared` tone、不淡显（区别于 V4 纯滑窗层"真没有"的 gone 灰）。
	// V4 三种层型：gone 节点是真不存在，保持灰色淡显。
	const isV41Shared = variant === 'v41-reuse' || variant === 'v41-reindex';
	const nodeTone = isV41Shared ? ('shared' as const) : ('gone' as const);
	const nodes = NODES.map((n) => {
		if (n.id === 'cache' && isV41Shared && src !== undefined) {
			// KV Cache 的压缩段来自共享池——sub 里标来源动作（层 2 压缩 / 层 5 CED 投影）
			return { ...n, sub: `滑窗（本层）+ 第 ${src} 层${srcAction ?? '压缩'}` };
		}
		if (!gone.has(n.id)) return n;
		const sub = v41Sub(n.id, variant, src, srcAction);
		return { ...n, tone: nodeTone, faded: !isV41Shared, sub: sub ?? GONE[variant][n.id] };
	});
	const edges = EDGES.map((e) =>
		gone.has(e.from) || gone.has(e.to) ? { ...e, tone: nodeTone, faded: !isV41Shared } : e
	);
	const built = { nodes, edges };
	if (src === undefined) CACHE.set(variant, built);
	return built;
}

/**
 * 取这张图、把本步正在算的节点标出来。
 *
 * 高亮里**自动滤掉本层没有的节点**——同一步的 active 列表对三种层型是同一份，
 * 纯滑窗层不该出现"高亮着一条不存在的支路"。
 *
 * 但 V4.1 的 **shared 节点不过滤**：它们"来自别处、本层在用"，score/attn 页确实会高亮
 * `ckv` / `itop`（本层的注意力真读它们），不该像 V4 纯滑窗层那样被滤掉。
 *
 * `src` / `srcAction`（V4.1 Reuse / Reindex 的来源层号与动作）只影响 ckv / cache 的来源说明，不影响高亮过滤。
 */
export function v4Diagram(
	variant: V4Variant,
	active: string[],
	src?: number,
	srcAction?: string
): DataflowSpec {
	const gone = goneIdsOf(variant);
	const { nodes, edges } = build(variant, src, srcAction);
	// V4.1 shared 节点：不被过滤（见上）；V4 gone 节点：照旧过滤
	const filterGone = (id: string) =>
		!gone.has(id) || variant === 'v41-reuse' || variant === 'v41-reindex';
	return { nodes, edges, active: active.filter(filterGone), hint: HINTS[variant] };
}
