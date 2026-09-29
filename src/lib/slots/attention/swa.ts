/**
 * 滑动窗口注意力（SWA）——DeepSeek V4 里 `compress_ratio = 0` 的层：
 * 只看最近 `swa_window` 个 token 的精确 KV。Q/KV、K = V、attn_sink、逆 RoPE、
 * 分组低秩输出投影都和 CSA / HCA 完全一样（见 `hybrid.ts`）。
 */
import type { SemanticsSpec } from '$lib/core/types';
import {
	hybridCompute,
	hybridHeadSteps,
	hybridTailSteps,
	makeHybridWeights,
	type HybridPlan,
	type HybridTrace,
	type HybridWeights
} from './hybrid';

/** 纯滑窗层：没有压缩分支，也没有 indexer */
const PLAN: HybridPlan = { ratio: 0, overlap: false, indexer: false };

export const swa: SemanticsSpec<HybridTrace> = {
	id: 'swa',
	slot: 'attention',
	name: '滑动窗口注意力 (SWA)',
	desc: '只看最近 W 个 token 的精确 KV，缓存因此有上限（不随序列变长）。代价是更早的历史完全看不到——V4 用 CSA / HCA 的压缩分支把那一段捡回来。',
	formula:
		'M_{ij}=1\\iff j\\le i\\ \\wedge\\ i-j<W,\\qquad O=\\mathrm{softmax}\\!\\left(\\frac{QK^{\\top}}{\\sqrt{d_h}}+M\\right)V',
	defaultDepth: 'L2',
	maxDepth: 'L2',

	makeWeights(rnd, cfg): HybridWeights {
		return makeHybridWeights(rnd, cfg, PLAN);
	},

	compute(x, ctx) {
		return hybridCompute(x, ctx.w as HybridWeights, PLAN, ctx.cfg);
	},

	steps(trace, ctx) {
		const w = ctx.w as HybridWeights;
		const prefix = 'swa';
		return [
			...hybridHeadSteps(trace, w, ctx.cfg, PLAN, prefix),
			...hybridTailSteps(trace, w, ctx.cfg, PLAN, prefix)
		];
	}
};
