/**
 * 参考模型前向。
 *
 * 本文件**不知道任何具体算子**——它只从注册表取插件、调用 `compute`、把 out 串起来。
 * 每一层用哪个插件由**代表模型**（`ModelSpec`）定下来：注意力逐层可能不同（V4 的
 * SWA / CSA / HCA 就是逐层变的），FFN 也一样。
 *
 * 骨架遵循真实 Transformer（与 DeepSeek R1 / V4 一致）：每个子层前做 RMSNorm，
 * 子层输出以残差加回。**这两件事不是可选项**——少了归一化，激活尺度不可控、
 * 注意力分数会全部趋近于 0；少了残差，token 自身信息在注意力后被平均掉，
 * 表征会完全坍缩（两两余弦相似度趋近 1）。
 */
import { matAdd, matmul, mulberry32, rmsNorm, type Mat } from '$lib/core/mat';
import { getSemantics, type ResolvedPlugin } from '$lib/core/registry';
import type { LayerCtx, SlotTrace } from '$lib/core/types';
import { ffnIdFor, layerKind, type LayerKind, type ModelSpec } from './config';
import type { Weights } from './init';

export interface LayerTrace {
	layer: number;
	kind: LayerKind;
	attnId: string;
	ffnId: string;
	attention: SlotTrace;
	ffn: SlotTrace;
	/** 层输入 x（**未归一化**）——残差加回的那条"主残差流"本身 */
	xIn: Mat;
	/** 注意力输出加回之后（`x + attn_out`） */
	attnResidual: Mat;
	/** FFN 输出加回之后（`x₁ + ffn_out`，= 下一层的输入 / LM Head 的输入） */
	ffnResidual: Mat;
}

export interface ModelTrace {
	tokenIds: number[];
	embed: Mat;
	layers: LayerTrace[];
	/** 最后一层之后的隐藏态 [seq, d_model]——LM Head 的输入 */
	hidden: Mat;
	logits: Mat;
}

/** 逐层解析注意力插件（`null` 表示这一层由用户选） */
export type AttnResolver = (layer: number) => ResolvedPlugin;
/** 逐层解析 FFN 插件 */
export type FfnResolver = (layer: number) => string;

/**
 * 注意力插槽用**解析后的插件**（语义 × 实现）——这样 `forward` 不用关心
 * "这一步的 compute 来自语义还是实现"，也不需要认识任何算子名。
 */
export function forward(
	weights: Weights,
	spec: ModelSpec,
	resolveAttn: AttnResolver,
	resolveFfn: FfnResolver = (l) => ffnIdFor(spec, l)
): ModelTrace {
	const cfg = spec.cfg;

	// 固定 seed 的随机 token，保证每次刷新看到同一份输入
	const rnd = mulberry32(cfg.seed + 7);
	const tokenIds = Array.from({ length: cfg.seq_len }, () => Math.floor(rnd() * cfg.vocab_size));

	const embed = tokenIds.map((id) => weights.embed[id].slice());
	let x: Mat = embed;

	const layers: LayerTrace[] = [];

	for (let l = 0; l < cfg.num_layers; l++) {
		const kind = layerKind(cfg, l);
		const attn = resolveAttn(l);
		// 层输入：残差流本身。子层看的是它的归一化版，加回的是它原样——
		// 所以"残差加回"是 `x + sublayer(rmsNorm(x))`，不是 `norm(x) + sublayer(...)`。
		const xIn = x;

		// ── Attention（前置 RMSNorm + 残差）──────────────
		const attnIn = rmsNorm(x);
		const attnCtx: LayerCtx = { layer: l, cfg, w: weights.slots[attn.weightKey]?.[l] };
		const attnTrace = attn.compute(attnIn, attnCtx);
		x = matAdd(x, attnTrace.out);
		const attnResidual = x;

		// ── FFN / MoE（前置 RMSNorm + 残差）─────────────
		const ffnId = resolveFfn(l);
		const ffnSpec = getSemantics(ffnId);
		const ffnIn = rmsNorm(x);
		const ffnCtx: LayerCtx = { layer: l, cfg, w: weights.slots[ffnId]?.[l] };
		const ffnTrace = ffnSpec.compute(ffnIn, ffnCtx);
		x = matAdd(x, ffnTrace.out);
		const ffnResidual = x;

		layers.push({
			layer: l,
			kind,
			attnId: attn.semantics.id,
			ffnId,
			attention: attnTrace,
			ffn: ffnTrace,
			xIn,
			attnResidual,
			ffnResidual
		});
	}

	// 词表投影：`[seq, d_model] · [d_model, vocab] → [seq, vocab]`
	//
	// 注意 `lmHead` 的存储形状就是 `[d_model, vocab]`，**不要再转置**——
	// 多转一次维度就对不上（8 ≠ 12），`matmul` 取到 `undefined` 会让整张 logits 变成 NaN。
	// `verify` 会断言 logits 全部有限。
	const logits = matmul(x, weights.lmHead);

	return { tokenIds, embed, layers, hidden: x, logits };
}
