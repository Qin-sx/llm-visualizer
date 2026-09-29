/**
 * 压缩稀疏注意力（CSA，`compress_ratio = 4`）——DeepSeek V4 里最复杂的那类层。
 *
 * 它在纯滑窗之上加两样东西：
 *   1. **4 倍压缩历史**：每 4 个 token 用可学习的加权平均压成 1 条 KV，
 *      而且用**重叠窗口**（每条看"上一组 4 个 + 自己 4 个"），组与组的边界因此是渐变的。
 *   2. **indexer 稀疏检索**：每个 query 用一套更窄的压缩 KV 给所有压缩条目打分，
 *      只挑 top-k 条真正参与注意力。
 *
 * 于是这一层的注意力同时看到"最近 128 个 token 的精确 KV"和"历史上若干条压缩 KV"，
 * 而且这两类**进的是同一个 softmax**（见 `hybrid.ts` 的说明）。
 */
import type { ModelConfigLike, SemanticsSpec } from '$lib/core/types';
import { compressorSteps, indexerSteps } from './compressor';
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

/**
 * 本层的压缩比由模型的 `compress_ratios[layer]` 决定（官方 4 层 harness 里是第 3 层）。
 * 配不上就早报错——悄悄退回一个默认值只会画出一张和权重对不上的图。
 */
function planFor(cfg: ModelConfigLike, layer: number): HybridPlan {
	const ratio = cfg.compress_ratios?.[layer];
	if (!ratio) {
		throw new Error(`[csa] 第 ${layer + 1} 层的 compress_ratios 不是压缩层（CSA 需要 ratio > 0）`);
	}
	return { ratio, overlap: true, indexer: true };
}

export const csa: SemanticsSpec<HybridTrace> = {
	id: 'csa',
	slot: 'attention',
	name: '压缩稀疏注意力 (CSA)',
	desc: '滑窗 + 4 倍压缩历史 + indexer 稀疏检索：压缩 KV 由可学习的加权平均得到（重叠窗口让组边界渐变），indexer 每个 query 只挑 top-k 条压缩条目进注意力。',
	formula:
		'\\text{CSA}=\\underbrace{\\text{SWA}}_{\\text{最近 }W\\text{ 个精确 KV}}+\\underbrace{\\text{compress}_{4}+\\text{top-}k}_{\\text{远处压缩历史}}',
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
		const prefix = 'csa';
		const variant = variantOf(plan);
		return [
			...hybridHeadSteps(trace, w, ctx.cfg, plan, prefix),
			// 注意力的压缩 KV（宽 head_dim，负责"被看"）
			...(trace.compress
				? compressorSteps(trace.compress, w.comp!, ctx.cfg, prefix, '注意力的压缩 KV', variant)
				: []),
			// indexer 自己的压缩 KV（更窄，只负责"排序"）+ 打分与 top-k
			...(trace.indexer ? indexerSteps(trace.indexer, w.idx!, ctx.cfg, prefix, variant) : []),
			...hybridTailSteps(trace, w, ctx.cfg, plan, prefix)
		];
	}
};
