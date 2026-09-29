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

/** R1 的结构特征：前 first_k_dense 层是 dense FFN，其余是 MoE */
export function layerKind(layer: number): LayerKind {
	return layer < CONFIG.first_k_dense ? 'dense' : 'moe';
}

/**
 * 默认插件组合——这是**配置**而非核心逻辑，所以允许出现具体插件名。
 * 对应 R1 的结构：前 N 层 dense，其余 MoE。
 */
export const DEFAULT_FFN_BY_KIND: Record<LayerKind, string> = {
	dense: 'dense-ffn',
	moe: 'deepseek-moe'
};
