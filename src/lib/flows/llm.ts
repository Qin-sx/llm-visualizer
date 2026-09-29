/**
 * LLM（decoder-only）流程。
 *
 * 阶段顺序：`Embedding → Attention → FFN/MoE → LM Head`。
 * 前后两个是**模型级**的（整个模型各一次），中间两个展开的是**正在检视的那一层**。
 *
 * 新增一条流程（diffusion / 多模态 / …）：仿照这里新写一个 `buildXxxFlow` 并
 * `registerFlow()` 即可。
 *
 * 算子（`mha` / `deepseek-moe` / …）依然由注册表提供——**本文件不含任何算子名**，
 * 只按插槽取当前选中的插件。
 */
import { buildFlow, type BuiltFlow, type StageOutput, type StageToggle } from '$lib/core/flow';
import { getSemantics, registerFlow, resolvePlugin, type ResolvedPlugin } from '$lib/core/registry';
import { softmaxRow } from '$lib/core/mat';
import type { LayerCtx, SemanticsSpec } from '$lib/core/types';
import { CONFIG, DEFAULT_FFN_BY_KIND, type LayerKind } from '$lib/model/config';
import { forward, type ModelTrace } from '$lib/model/forward';
import { initWeights } from '$lib/model/init';
import { embeddingLookup, nextTokenHead, type HeadData } from '$lib/stages';

export interface LlmFlowOptions {
	attnId: string;
	/** 注意力的**执行方式**（如 MLA 的矩阵吸收合并）；不给就用语义插件自带的实现 */
	attnImplId?: string | null;
	/** `'auto'` = 按 R1 结构（前 N 层 dense，其余 MoE）；否则整网统一用该插件 */
	ffnId: string;
	layer?: number;
	seed?: number;
	/**
	 * 关掉的阶段 id（见 `LLM_STAGE_TOGGLES`）；不在里面 = 开着。不传 = 全开。
	 *
	 * 关掉的阶段**根本不进流程**——步骤、流程图节点、总览都跟着消失，
	 * 而不是"渲染出来但藏起来"。
	 */
	off?: string[];
}

/**
 * 这条流程有哪几个阶段可以开关、默认开不开。
 *
 * 默认只开 Attention 与 FFN/MoE（这条流程的主体）；Embedding 与 LM Head
 * 各只有一步，想看"输入从哪来 / 输出怎么变成下一个 token"时再打开。
 */
export const LLM_STAGE_TOGGLES: StageToggle[] = [
	{ id: 'embedding-lookup', title: 'Embedding', defaultOn: false },
	{ id: 'attention', title: 'Attention', defaultOn: true },
	{ id: 'ffn-moe', title: 'FFN / MoE', defaultOn: true },
	{ id: 'lm-head', title: 'LM Head', defaultOn: false }
];

export interface LlmFlowResult extends BuiltFlow {
	trace: ModelTrace;
	/** 当前检视的层号 */
	layer: number;
	/** 当前生效的注意力插件（语义 × 实现） */
	attn: ResolvedPlugin;
	ffnSpec: SemanticsSpec<any>;
}

/** LM Head 阶段的输入：最后一层的最后一个位置 */
function headDataOf(trace: ModelTrace, lmHead: number[][]): HeadData {
	const last = trace.hidden[trace.hidden.length - 1];
	const logits = trace.logits[trace.logits.length - 1];
	return {
		last,
		table: lmHead,
		logits,
		probs: softmaxRow(logits),
		topN: 4
	};
}

export function buildLlmFlow(opts: LlmFlowOptions): LlmFlowResult {
	const weights = initWeights(opts.seed ?? CONFIG.seed);
	// 语义 × 实现 → 当前生效的注意力插件（吸收合并这类"怎么算"的选择在这一层生效）
	const attn = resolvePlugin(opts.attnId, opts.attnImplId);

	const resolveFfn = (_l: number, kind: LayerKind) =>
		opts.ffnId === 'auto' ? DEFAULT_FFN_BY_KIND[kind] : opts.ffnId;

	const trace = forward(weights, attn, resolveFfn);

	const layer = opts.layer ?? CONFIG.first_k_dense;
	const lt = trace.layers[layer];
	const ffnSpec = getSemantics(lt.ffnId);

	const ctxFor = (key: string): LayerCtx => ({
		layer,
		cfg: CONFIG,
		w: weights.slots[key]?.[layer]
	});

	// 关掉的阶段**根本不进流程**：步骤、流程图节点、总览都跟着消失
	const off = new Set(opts.off ?? []);
	const outputs: StageOutput[] = [];

	if (!off.has('embedding-lookup')) {
		outputs.push({
			stage: embeddingLookup,
			steps: embeddingLookup.steps(
				{ tokenIds: trace.tokenIds, table: weights.embed, rows: trace.embed },
				CONFIG
			)
		});
	}
	if (!off.has('attention')) {
		// 层级的阶段描述在运行时填上当前插件名——流程图因此永远显示真实插件，
		// 而不是写死的名字。选了执行方式时也一并显示（`MLA · 矩阵吸收合并`）
		outputs.push({
			stage: {
				id: 'attention',
				slot: 'attention',
				scope: 'layer',
				depth: 'L2',
				title: 'Attention',
				sub: attn.name,
				desc: attn.desc
			},
			steps: attn.steps(lt.attention, ctxFor(attn.weightKey))
		});
	}
	if (!off.has('ffn-moe')) {
		outputs.push({
			stage: {
				id: 'ffn-moe',
				slot: 'ffn-moe',
				scope: 'layer',
				depth: 'L2',
				title: 'FFN / MoE',
				sub: ffnSpec.name,
				desc: ffnSpec.desc
			},
			steps: ffnSpec.steps(lt.ffn, ctxFor(ffnSpec.id))
		});
	}
	if (!off.has('lm-head')) {
		outputs.push({
			stage: nextTokenHead,
			steps: nextTokenHead.steps(headDataOf(trace, weights.lmHead), CONFIG)
		});
	}

	// 四个阶段全关掉时流程是空的——这不算错误（UI 会给提示），
	// 但 `buildFlow([])` 出来的 `steps` 是空数组，调用方要能处理
	const built = buildFlow(outputs);

	return { ...built, trace, layer, attn, ffnSpec };
}

registerFlow({
	id: 'llm',
	name: 'LLM（decoder-only）',
	desc: 'Embedding → 逐层 Attention + FFN/MoE → LM Head',
	stageToggles: LLM_STAGE_TOGGLES,
	build: buildLlmFlow
});
