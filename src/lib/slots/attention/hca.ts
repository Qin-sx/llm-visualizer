/**
 * 高压缩注意力（HCA，`compress_ratio = 128`）。
 *
 * 和 CSA 是同一套骨架（滑窗 + 压缩历史），差别只有两处：
 *   - **压缩比大得多**：每 128 个 token 才压出 1 条，条目很少但每条覆盖很长一段。
 *   - **没有 indexer**：条目本来就少，干脆全部参与注意力（稠密），
 *     省掉"打分 + top-k"这一整套开销。
 *
 * 所以 CSA 和 HCA 不是"两种注意力"，而是同一个混合注意力的两档配置：
 * 压缩比小 → 条目多 → 需要检索；压缩比大 → 条目少 → 全看就行。
 */
import type { ModelConfigLike, SemanticsSpec } from '$lib/core/types';
import { compressorSteps } from './compressor';
import {
	hybridCompute,
	hybridHeadSteps,
	hybridTailSteps,
	makeHybridWeights,
	variantOf,
	type HybridPlan,
	type HybridTrace,
	type HybridWeights
} from './hybrid';

/** 本层的压缩比由模型的 `compress_ratios[layer]` 决定（官方 4 层 harness 里是第 4 层） */
function planFor(cfg: ModelConfigLike, layer: number): HybridPlan {
	const ratio = cfg.compress_ratios?.[layer];
	if (!ratio) {
		throw new Error(`[hca] 第 ${layer + 1} 层的 compress_ratios 不是压缩层（HCA 需要 ratio > 0）`);
	}
	return { ratio, overlap: false, indexer: false };
}

export const hca: SemanticsSpec<HybridTrace> = {
	id: 'hca',
	slot: 'attention',
	name: '高压缩注意力 (HCA)',
	desc: '滑窗 + 128 倍压缩历史，稠密参与：压缩条目本来就少，所以没有 indexer，全部进同一次 softmax。与 CSA 是同一套骨架的两档配置。',
	formula:
		'\\text{HCA}=\\underbrace{\\text{SWA}}_{\\text{最近 }W\\text{ 个精确 KV}}+\\underbrace{\\text{compress}_{128}}_{\\text{条目少，全看}}',
	defaultDepth: 'L2',
	maxDepth: 'L2',

	makeWeights(rnd, cfg, layer): HybridWeights {
		return makeHybridWeights(rnd, cfg, planFor(cfg, layer));
	},

	compute(x, ctx) {
		return hybridCompute(x, ctx.w as HybridWeights, planFor(ctx.cfg, ctx.layer), ctx.cfg);
	},

	steps(trace, ctx) {
		const w = ctx.w as HybridWeights;
		const plan = planFor(ctx.cfg, ctx.layer);
		const prefix = 'hca';
		return [
			...hybridHeadSteps(trace, w, ctx.cfg, plan, prefix),
			...(trace.compress
				? compressorSteps(
						trace.compress,
						w.comp!,
						ctx.cfg,
						prefix,
						'注意力的压缩 KV',
						variantOf(plan)
					)
				: []),
			...hybridTailSteps(trace, w, ctx.cfg, plan, prefix)
		];
	}
};
