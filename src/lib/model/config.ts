/**
 * v0 精确尺寸配置。
 *
 * 尺寸选择的首要约束：**每个矩阵都要能逐格显示真实数字**。
 * 因此 d_model 取 8、权重取 8×8（64 格），一个矩阵在屏幕上约 280px 宽。
 * 真实 DeepSeek R1 的维度见 `REAL_R1`，界面上会标注缩小了多少倍。
 *
 * 缩放原则：**保留结构性特征，只缩绝对维度**。
 */
import type { ModelConfigLike } from '$lib/core/types';

export const CONFIG: ModelConfigLike & { seed: number } = {
	// ── 序列 ──────────────────────────────
	seq_len: 6, // 6 个 token
	// 词表：真实 129280，这里缩到 12。上限由"矩阵必须能逐格显示数字"倒推——
	// LM Head 的 logits 是一行 vocab 个数，列数超过 14 就会把格子压到 20px 以下、写不下数字
	// （与 d_model 取 8 是同一个约束）。
	vocab_size: 12,
	causal: true, // DeepSeek R1 是 decoder-only，注意力需施加因果掩码

	// ── 主干 ──────────────────────────────
	num_layers: 4, // 4 层
	d_model: 8, // 隐藏维度：8 维，一屏放得下
	first_k_dense: 2, // 前 2 层是 dense FFN，后 2 层是 MoE

	// ── Attention（最简单的 MHA）──────────
	num_heads: 2, // 2 个头
	head_dim: 4, // 每头 4 维；2 × 4 = 8 = d_model

	// ── MLA（Multi-head Latent Attention）─────────────────
	// 维度按同一套原则缩放：**保留结构性特征，只缩绝对维度**。
	// 三个必须保住的特征：
	//   ① kv_lora_rank < q_lora_rank < d_model（真实 512 < 1536 < 7168）
	//   ② kv_lora_rank ≪ num_heads × qk_nope_head_dim，否则"压缩"看不出来
	//   ③ qk_rope_head_dim 取 2 而不是 1 —— RoPE 必须能旋转"一对"
	q_lora_rank: 3, // Q 的低秩维度（真实 1536）
	kv_lora_rank: 2, // KV 的共享潜向量维度（真实 512）★ MLA 的核心
	qk_nope_head_dim: 3, // 每头"不含位置编码"的 Q/K 维度（真实 128）
	qk_rope_head_dim: 2, // 每头位置编码占的维度（真实 64）——不可压缩，必须单独缓存
	v_head_dim: 3, // 每头 V 维度（真实 128，与 qk_nope_head_dim 相等）

	// ── Dense FFN（前 2 层）───────────────
	dense_intermediate: 8,

	// ── MoE（后 2 层）─────────────────────
	// 路由专家的中间维度**小于** d_model（R1 真实也是 2048 < 7168）：
	// 细粒度专家的特点，dense FFN 才是真的升维
	moe_intermediate: 4, // 每个专家的中间维度
	num_routed_experts: 4, // 4 个路由专家
	num_shared_experts: 1, // 1 个共享专家
	top_k: 2, // 每个 token 激活 2 个路由专家

	seed: 42
};

/**
 * DeepSeek R1 真实 config（`model/ds-r1/config.json`）。
 * 仅用于界面上标注"真实 size 是多少、缩小了几倍"。
 */
export const REAL_R1 = {
	num_layers: 61,
	d_model: 7168,
	num_heads: 128,
	head_dim: 128, // qk_nope_head_dim
	rope_head_dim: 64, // qk_rope_head_dim
	v_head_dim: 128,
	/** MLA 的低秩维度：KV 潜向量 512、Q 潜向量 1536 */
	kv_lora_rank: 512,
	q_lora_rank: 1536,
	dense_intermediate: 18432,
	moe_intermediate: 2048,
	num_routed_experts: 256,
	num_shared_experts: 1,
	top_k: 8,
	vocab_size: 129280
} as const;

/** 标注缩放倍数，如 "缩小 896×" */
export function scaleLabel(real: number, shown: number): string {
	if (shown <= 0) return '';
	const f = real / shown;
	if (f < 1.5) return '与真实同量级';
	if (f < 10) return `缩小 ${f.toFixed(1)}×`;
	return `缩小 ${Math.round(f)}×`;
}

export type LayerKind = 'dense' | 'moe';

/** 某层的 FFN 类型：前 `first_k_dense` 层是 dense FFN，其余是 MoE（V4 全 MoE） */
export function layerKind(cfg: ModelConfigLike, layer: number): LayerKind {
	return layer < cfg.first_k_dense ? 'dense' : 'moe';
}

/**
 * 默认插件组合——这是**配置**而非核心逻辑，所以允许出现具体插件名。
 * 对应 R1 的结构：前 N 层 dense，其余 MoE。
 */
export const DEFAULT_FFN_BY_KIND: Record<LayerKind, string> = {
	dense: 'dense-ffn',
	moe: 'deepseek-moe'
};

// ── DeepSeek V4-Flash ─────────────────────────────────────
/**
 * V4-Flash 的缩放版。结构取官方 4 层 parity harness（`compress_ratios=[0,0,4,128]`），
 * 所以层选择器还是 4 个 chip。
 *
 * 与 R1 版的两处结构性差别：
 *   ① **序列更长**（8 而不是 6）—— 压缩要有历史可压；
 *   ② **全 MoE**（没有 dense 层），前两层是纯 SWA 层。
 *
 * **三个"128/4"是怎么缩的**（用户定的）：真实的压缩比 `4` → `2`、`128` → `4`，
 * 滑窗 `128` → `4`。这样 8 个 token 下：CSA 压出 4 条、HCA 压出 2 条
 * （**方向不能反**：CSA 压得轻、条目多；HCA 压得狠、条目少）。
 *
 * 序列取 8 而不是 16 是**布局约束**，不是随便缩的：注意力分数矩阵要画成真正的
 * `matmul` 视图（能逐格写数字），它的列数 = 序列长度 + 压缩条目数 + 1(sink)，
 * 而 `MatmulView` 的格子边长是 `290/列数`、低于 20px 就写不下数字：
 *
 * | 序列 | SWA 列数/格子 | CSA 列数/格子 | HCA 列数/格子 |
 * | --- | --- | --- | --- |
 * | 16 | 17 / 17px ✗ | 25 / 15px ✗ | 21 / 15px ✗ |
 * | 8 | 9 / 32px ✓ | 13 / 22px ✓ | 11 / 26px ✓ |
 *
 * 16 的时候**三种层型都写不下数字**（连纯滑窗的 17 列也只有 17px），所以必须降到 8。
 */
export const V4_FLASH_CFG: ModelConfigLike & { seed: number } = {
	seq_len: 8, // 8 个 token：压缩比 2 → 4 条压缩条目，比 4 → 2 条
	vocab_size: 12,
	causal: true,

	num_layers: 4, // 官方 parity harness 的层数
	d_model: 8,
	first_k_dense: 0, // 全 MoE，没有 dense 层

	// ── 混合注意力（V4 的核心）──────────────
	// 每头 4 维 = 不含位置编码 2 + 位置编码 2（真实 448 + 64 = 512）。
	// 头维压得很小是**布局约束**：压缩器的 gate 要投影出 2·coff·head_dim 段
	// （重叠窗口下是 4·head_dim），再宽就写不下数字了。
	num_heads: 2,
	head_dim: 4,
	qk_nope_head_dim: 2,
	qk_rope_head_dim: 2,
	v_head_dim: 4, // K = V 共享，所以 V 的宽度就是 head_dim
	q_lora_rank: 4, // Q 的低秩潜向量（真实 1024）

	swa_window: 4, // 最近 4 个 token 用精确 KV（真实 128）——必须 < 序列长度，否则滑窗就退化成纯因果了
	compress_ratios: [0, 0, 2, 4], // 纯 SWA / 纯 SWA / CSA(真实 4) / HCA(真实 128)
	rope_theta: 10000, // 纯滑窗层用主 RoPE
	compress_rope_theta: 40000, // 压缩层改用压缩版 YaRN RoPE（真实 base 40000）

	index_n_heads: 2, // indexer 的头数（真实 64）
	index_head_dim: 2, // indexer 每头维度（真实 128）——比注意力的 head_dim=4 更窄，q^I 和 Q 一眼可分
	index_topk: 2, // 4 条压缩条目里挑 2 条（真实 512）

	o_groups: 2, // 输出投影分成 2 组各自降到低秩（真实 8 组）
	o_lora_rank: 4, // 每组降到 4 维再拼回来（真实 1024）

	// ── MoE（V4 全 MoE，复用 deepseek-moe 插件）──
	dense_intermediate: 8, // V4 没有 dense 层，这个字段用不上（类型要求）
	moe_intermediate: 4,
	num_routed_experts: 4,
	num_shared_experts: 1,
	top_k: 2,

	// ── mHC 混合残差流（真实 4 条，展示缩成 2 条）──────
	// 真实 `hc_mult = 4`（结构），但 4 条流并排太挤、页面复杂——展示只画 2 条，
	// 每个 mHC 页面都标注"真实 4 条流，此处只演示 2 条"。门控形状随 hc 缩放
	// （`mix = (2+hc)·hc = 8`，`fn: [8, 2×d_model]`，真实 `[24, 4×4096]`）。
	hc_mult: 2,
	hc_sinkhorn_iters: 20,
	hc_eps: 1e-6,

	seed: 42
};

/** DeepSeek V4-Flash 真实 config（`model/ds-v4-flash/config.json`） */
export const REAL_V4_FLASH = {
	num_layers: 43,
	d_model: 4096,
	num_heads: 64,
	head_dim: 512, // qk_nope_head_dim 448 + qk_rope_head_dim 64
	rope_head_dim: 64,
	v_head_dim: 512, // K = V 共享
	q_lora_rank: 1024,
	swa_window: 128,
	compress_rope_theta: 40000,
	index_topk: 512,
	index_n_heads: 64,
	index_head_dim: 128,
	o_groups: 8,
	o_lora_rank: 1024,
	moe_intermediate: 2048,
	num_routed_experts: 256,
	num_shared_experts: 1,
	top_k: 6,
	vocab_size: 129280,
	// mHC：4 条并行残差流、Sinkhorn 迭代 20 次、eps 1e-6（config.json 与 sglang 默认一致）
	hc_mult: 4,
	hc_sinkhorn_iters: 20,
	hc_eps: 1e-6
} as const;

/** 真实维度（界面上标注"真实 size 是多少、缩小了几倍"用） */
export type RealDims = typeof REAL_R1 | typeof REAL_V4_FLASH;

/**
 * 一个**代表模型**：缩放配置 + 真实维度 + 逐层结构。
 *
 * 注意力与 FFN 都按层定下来，UI 上就不再是"自由组合插件"——
 * 而是"选一个代表模型 + 选一层"。这样非法组合（如 MLA 配 V4）根本不存在。
 */
export interface ModelSpec {
	id: string;
	name: string;
	desc: string;
	cfg: ModelConfigLike & { seed: number };
	real: RealDims;
	/** 逐层的 FFN 插件 id */
	ffnByLayer: string[];
	/** 逐层的注意力插件 id；`null` = 这一层的注意力由用户在 `attnChoices` 里选 */
	attnByLayer: (string | null)[];
	/** 用户可挑的注意力语义（空 = 逐层固定，不给选） */
	attnChoices: string[];
	/** 层选择器上每层的标注（如 `dense` / `MoE` / `SWA` / `C4` / `C128`） */
	layerLabels: string[];
	/**
	 * 层故事里的残差形态：
	 * `none` = 没有残差阶段；`plain` = 普通残差（x + 子层输出，每层两次）；`mhc` = V4 的 mHC
	 * （`hc_mult` 条并行残差流 + 每子层 pre/post 混合 + 末尾 `hc_head` 压回）。
	 */
	residual: 'none' | 'plain' | 'mhc';
}

/** R1：前 2 层 dense FFN、其余 MoE；注意力可选 MHA / MLA */
export const R1_MODEL: ModelSpec = {
	id: 'r1',
	name: 'DeepSeek R1',
	desc: '61 层、MLA 注意力 + 细粒度 MoE。这里缩成 4 层（前 2 层 dense FFN、其余 MoE），注意力可选 MHA 或 MLA。',
	cfg: CONFIG,
	real: REAL_R1,
	ffnByLayer: ['dense-ffn', 'dense-ffn', 'deepseek-moe', 'deepseek-moe'],
	attnByLayer: [null, null, null, null],
	attnChoices: ['mha', 'mla'],
	layerLabels: ['dense', 'dense', 'MoE', 'MoE'],
	residual: 'plain'
};

/** V4-Flash：全 MoE；注意力逐层固定（纯 SWA / CSA / HCA） */
export const V4_FLASH_MODEL: ModelSpec = {
	id: 'v4-flash',
	name: 'DeepSeek V4-Flash',
	desc: '43 层全 MoE + 混合注意力：每层的注意力由 compress_ratios 决定——纯 SWA、CSA（4 倍压缩 + indexer 稀疏检索）或 HCA（128 倍压缩，稠密）。这里缩成官方的 4 层 parity 结构。',
	cfg: V4_FLASH_CFG,
	real: REAL_V4_FLASH,
	ffnByLayer: ['deepseek-moe', 'deepseek-moe', 'deepseek-moe', 'deepseek-moe'],
	attnByLayer: ['swa', 'swa', 'csa', 'hca'],
	attnChoices: [],
	layerLabels: ['SWA', 'SWA', 'C4', 'C128'],
	// V4 的残差是 mHC（4 条并行流）
	residual: 'mhc'
};

export const MODELS: ModelSpec[] = [R1_MODEL, V4_FLASH_MODEL];

export function getModel(id: string): ModelSpec {
	return MODELS.find((m) => m.id === id) ?? R1_MODEL;
}

/**
 * 这个模型会用到的插件（不含 `ffnId` 覆盖用的备选）。
 *
 * 用途有二：`initWeights` 据此判断"这个插件要不要初始化"，UI / 校验据此判断
 * "某个插件在这个模型下根本不存在"。
 */
export function pluginsOf(spec: ModelSpec): string[] {
	return [
		...new Set([
			...spec.ffnByLayer,
			...spec.attnByLayer.filter((a): a is string => !!a),
			...spec.attnChoices
		])
	];
}

/**
 * 某一层的注意力插件候选。
 *
 * 逐层固定的（V4）只有它自己那一个；`null` 的（R1）是用户可挑的那几个——
 * 权重按**候选**初始化，所以换选择不需要重新初始化。
 */
export function attnIdsFor(spec: ModelSpec, layer: number): string[] {
	const fixed = spec.attnByLayer[layer];
	return fixed ? [fixed] : spec.attnChoices;
}

/**
 * FFN 插件候选：所有层都按同一套初始化。
 *
 * 与注意力不同，FFN 的权重形状不依赖层号（也不依赖 V4 那些字段），
 * 而且流程允许用 `ffnId` 整网覆盖（校验脚本就会这么做），所以这里放全量。
 */
export function ffnIdsOf(spec: ModelSpec): string[] {
	return [...new Set([...spec.ffnByLayer, ...Object.values(DEFAULT_FFN_BY_KIND)])];
}

/** 某层的注意力插件 id：逐层固定的直接用，`null` 的由用户在 `attnChoices` 里选 */
export function attnIdFor(spec: ModelSpec, layer: number, choice: string): string {
	return spec.attnByLayer[layer] ?? choice;
}

/** 某层的 FFN 插件 id */
export function ffnIdFor(spec: ModelSpec, layer: number): string {
	return spec.ffnByLayer[layer] ?? DEFAULT_FFN_BY_KIND[layerKind(spec.cfg, layer)];
}
