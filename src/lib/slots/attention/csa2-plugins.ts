/**
 * CSA2 的三个角色插件（DeepSeek V4.1）——`csa2-full` / `csa2-reindex` / `csa2-reuse`。
 *
 * 三者**共享同一份注意力实现**（`csa2.ts` 的 compute / steps），角色由 `cfg` + 层号
 * 推导（`csa2PlanOf`）——Full / Reindex / Reuse 是"谁算谁共享"，不是三种算法，
 * 所以是三个薄壳，和 V4 的 swa / csa / hca 三壳套 hybrid 是同一套组织方式。
 *
 * 层分配（`V4_1_FLASH_MODEL.attnByLayer`，层号按 UI 1 开头）：
 *   `csa2-full`：第 3 层（ratio-2 编码器组 Full）、第 6 层（ratio-1 全局 KV Full + 候选源）
 *   `csa2-reindex`：第 7 层（复用第 6 层的池，用自己的 q^I 在候选池里重打分）
 *   `csa2-reuse`：第 4/5/8 层（不压缩不检索，直接读池 + 复用最近的 top-k）
 */
import { register } from '$lib/core/registry';
import type { SemanticsSpec } from '$lib/core/types';
import { csa2Compute, csa2PlanOf, csa2Steps, makeCsa2Weights, type Csa2Trace } from './csa2';

function shell(
	id: string,
	name: string,
	desc: string,
	formula: string
): SemanticsSpec<Csa2Trace> {
	return {
		id,
		slot: 'attention',
		name,
		desc,
		formula,
		defaultDepth: 'L2',
		maxDepth: 'L2',

		makeWeights(rnd, cfg, layer) {
			return makeCsa2Weights(rnd, cfg, csa2PlanOf(cfg, layer));
		},

		compute(x, ctx) {
			return csa2Compute(x, ctx);
		},

		steps(trace, ctx) {
			return csa2Steps(trace, ctx);
		}
	};
}

export const csa2Full: SemanticsSpec<Csa2Trace> = shell(
	'csa2-full',
	'CSA2 Full',
	'自己算 main KV 与 indexer K、扫全量建索引（candidate source 还建候选池）。只有 4 个 kv_source 层（演示第 3 / 6 层）是它——其余层都读它的池。',
	'C^l=HW^{KV}_{l},\\qquad I=\\mathrm{TopK}\\!\\left(Q_{\\mathrm{index}}^l K_{\\mathrm{index}}^\\top\\right)'
);

export const csa2Reindex: SemanticsSpec<Csa2Trace> = shell(
	'csa2-reindex',
	'CSA2 Reindex',
	'复用上游的 main KV 与 indexer K，但用自己的 indexer Q 在候选池里重新打分、挑 top-k——缓存共享，选择可以逐层不同（避免"一层选错、整组跟着错"）。',
	'I_l=\\mathrm{TopK}\\!\\left(Q_{\\mathrm{index}}^l K_{\\mathrm{index}}^\\top\\right)\\ \\text{（候选池内）}'
);

export const csa2Reuse: SemanticsSpec<Csa2Trace> = shell(
	'csa2-reuse',
	'CSA2 Reuse',
	'同时复用 main KV、indexer K 与最近一次 Full / Reindex 的 top-k——连 indexer 都不跑，最省的一档。',
	'O=\\mathrm{softmax}\\!\\left(\\frac{QK^{\\top}}{\\sqrt{d_h}}\\right)V,\\qquad \\text{（K、top-k 来自共享池）}'
);

register(csa2Full);
register(csa2Reindex);
register(csa2Reuse);
