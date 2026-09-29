/**
 * LLM（decoder-only）流程。
 *
 * 阶段顺序：`Embedding → Attention → FFN/MoE → LM Head`。
 * 前后两个是**模型级**的（整个模型各一次），中间两个展开的是**正在检视的那一层**。
 *
 * 新增一条流程（diffusion / 多模态 / …）：仿照这里新写一个 `buildXxxFlow` 并
 * `registerFlow()` 即可。
 *
 * 算子（`mha` / `csa` / `deepseek-moe` / …）依然由注册表提供——**本文件不含任何算子名**，
 * 只按代表模型（`ModelSpec`）逐层取插件。V4 的注意力是逐层变的（SWA / CSA / HCA），
 * 所以这里按层解析而不是"整网一个插件"。
 */
import { buildFlow, type BuiltFlow, type StageOutput, type StageToggle } from '$lib/core/flow';
import { getSemantics, registerFlow, resolvePlugin, type ResolvedPlugin } from '$lib/core/registry';
import { softmaxRow } from '$lib/core/mat';
import type { LayerCtx, SemanticsSpec } from '$lib/core/types';
import { attnIdFor, ffnIdFor, getModel, type ModelSpec } from '$lib/model/config';
import { forward, type ModelTrace } from '$lib/model/forward';
import { initWeights } from '$lib/model/init';
import { embeddingLookup, nextTokenHead, type HeadData } from '$lib/stages';

export interface LlmFlowOptions {
	/** 代表模型 id（见 `MODELS`）；不传用 R1 */
	modelId?: string;
	/** 逐层固定的层用不到它；`null` 的那些层（R1）用它 */
	attnId: string;
	/** 注意力的**执行方式**（如 MLA 的矩阵吸收合并）；不给就用语义插件自带的实现 */
	attnImplId?: string | null;
	/** `'auto'` = 按模型结构；否则整网统一用该插件 */
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
	/** 当前检视那一层的注意力插件（语义 × 实现） */
	attn: ResolvedPlugin;
	/** 当前检视那一层的 FFN 插件 */
	ffnSpec: SemanticsSpec<any>;
	/** 代表模型 */
	model: ModelSpec;
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
	const model = getModel(opts.modelId ?? 'r1');
	const cfg = model.cfg;
	const weights = initWeights(model, opts.seed);

	/**
	 * 逐层解析注意力：逐层固定的（V4）直接用模型给的 id；
	 * `null` 的（R1）用用户在 `attnChoices` 里选的。
	 *
	 * 语义 × 实现 → 当前生效的插件，所以"吸收合并这类'怎么算'的选择"在这一层生效。
	 */
	const resolved = new Map<number, ResolvedPlugin>();
	const resolveAttn = (layer: number) => {
		const id = attnIdFor(model, layer, opts.attnId);
		const key = `${id}|${opts.attnImplId ?? ''}`;
		const cached = resolved.get(layer);
		if (cached && `${cached.semantics.id}|${cached.implementation?.id ?? ''}` === key) return cached;
		const plugin = resolvePlugin(id, opts.attnImplId);
		resolved.set(layer, plugin);
		return plugin;
	};

	const trace = forward(
		weights,
		model,
		resolveAttn,
		// `'auto'` = 按模型结构（R1 前 2 层 dense、其余 MoE；V4 全 MoE）
		(l) => (opts.ffnId === 'auto' ? ffnIdFor(model, l) : opts.ffnId)
	);

	const layer = opts.layer ?? 0;
	const lt = trace.layers[layer];
	const attn = resolveAttn(layer);
	const ffnSpec = getSemantics(lt.ffnId);

	const ctxFor = (key: string): LayerCtx => ({
		layer,
		cfg,
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
				cfg
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
			steps: nextTokenHead.steps(headDataOf(trace, weights.lmHead), cfg)
		});
	}

	// 四个阶段全关掉时流程是空的——这不算错误（UI 会给提示），
	// 但 `buildFlow([])` 出来的 `steps` 是空数组，调用方要能处理
	const built = buildFlow(outputs);

	return { ...built, trace, layer, attn, ffnSpec, model };
}

registerFlow({
	id: 'llm',
	name: 'LLM（decoder-only）',
	desc: 'Embedding → 逐层 Attention + FFN/MoE → LM Head',
	stageToggles: LLM_STAGE_TOGGLES,
	build: buildLlmFlow
});
