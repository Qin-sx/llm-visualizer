/**
 * LLM（decoder-only）流程。
 *
 * 阶段顺序：`Embedding → Attention → FFN/MoE → Residual → LM Head`。
 * 前后两个是**模型级**的（整个模型各一次），中间三个展开的是**正在检视的那一层**
 * （Residual 做"整层收尾"：两个子层的输出都加回主残差流）。
 *
 * 新增一条流程（diffusion / 多模态 / …）：仿照这里新写一个 `buildXxxFlow` 并
 * `registerFlow()` 即可。
 *
 * 算子（`mha` / `csa` / `deepseek-moe` / …）依然由注册表提供——**本文件不含任何算子名**，
 * 只按代表模型（`ModelSpec`）逐层取插件。V4 的注意力是逐层变的（SWA / CSA / HCA），
 * 所以这里按层解析而不是"整网一个插件"。
 */
import { buildFlow, type BuiltFlow, type StageInfo, type StageOutput, type StageToggle } from '$lib/core/flow';
import { getSemantics, registerFlow, resolvePlugin, type ResolvedPlugin } from '$lib/core/registry';
import { softmaxRow, type Mat } from '$lib/core/mat';
import { matRefsOf } from '$lib/core/overview';
import type { LayerCtx, SemanticsSpec, Step } from '$lib/core/types';
import { attnIdFor, ffnIdFor, getModel, type ModelSpec } from '$lib/model/config';
import { forward, type LayerTrace, type ModelTrace } from '$lib/model/forward';
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
 * 默认只开 Attention / FFN-MoE / Residual（这条流程的主体）；Embedding 与 LM Head
 * 各只有一步，想看"输入从哪来 / 输出怎么变成下一个 token"时再打开。
 *
 * Residual 只在 `model.residual` 的模型（R1）里真的产出步骤——
 * V4 的残差是 Hyper-Connections（还没做），`+page.svelte` 会按模型把开关藏掉。
 */
export const LLM_STAGE_TOGGLES: StageToggle[] = [
	{ id: 'embedding-lookup', title: 'Embedding', defaultOn: false },
	{ id: 'attention', title: 'Attention', defaultOn: true },
	{ id: 'ffn-moe', title: 'FFN / MoE', defaultOn: true },
	{ id: 'residual', title: 'Residual', defaultOn: true },
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

/**
 * 找到"画出 `out` 这个矩阵"的那一步——残差步骤要引用"注意力输出 / FFN 输出"，
 * 但产出它们的步骤 id 因插件而异（MHA 是 `attn-out`、MLA 是 `mla-out`、
 * MoE 是 `moe-shared-add`、dense FFN 是 `ffn-down`）。从步骤里扫出真实 id，
 * 流程层不写死插件步骤名。
 */
function producingStepId(out: Mat, steps: Step[]): string | null {
	for (const s of steps) {
		for (const t of s.tensors) {
			for (const r of matRefsOf(t)) {
				if (r.data === out) return s.id;
			}
		}
	}
	return null;
}

/**
 * 残差阶段的两步（R1 普通残差）：`x + O_attn → x₁`、`x₁ + O_ffn → x₂`。
 *
 * 两步拆在**真实位置**：`residual-attn` 紧跟 Attention、`residual-ffn` 紧跟 FFN
 * （总览里"注意力残差"就不会跑到 MoE 后面去了）。属于同一个 `residual` 阶段——
 * `buildFlow` 把同一阶段 id 的多个输出并成一个流程图节点（位置取最后一次，
 * 正好落在 FFN 之后当"整层收尾"），但每段区间各记一条，步骤高亮按真实位置。
 *
 * `attnOutId` / `ffnOutId` 是"画出该子层输出"的那一步的 id；对应阶段被关掉时
 * 传 `null`，标签就不带"第 N 步"引用（否则 `{{step:...}}` 会解析成 `?`）。
 */
function residualSteps(
	lt: LayerTrace,
	model: ModelSpec,
	attnOutId: string | null,
	ffnOutId: string | null
): Step[] {
	const S = model.cfg.seq_len;
	const D = model.cfg.d_model;
	const real: [number, number] = [S, model.real.d_model];
	const attnRef = attnOutId ? `← 第 {{step:${attnOutId}}} 步的注意力输出` : '← 注意力的输出';
	const ffnRef = ffnOutId ? `← 第 {{step:${ffnOutId}}} 步的 FFN 输出` : '← FFN 的输出';

	return [
		{
			id: 'residual-attn',
			kind: 'ADD',
			label:
				`残差加回：注意力输出 O 加回**未归一化**的层输入 x（前面 Q/K/V 都是从归一化后的 X 投影的；` +
				`这条"主残差流"在这里把 token 自身的信息捡回来）→ x₁`,
			formula: 'x_1=x+O_{\\mathrm{attn}}',
			tensors: [
				{
					name: 'residual-attn-add',
					kind: 'sum',
					shape: [S, D],
					terms: [
						{
							name: 'x',
							shape: [S, D],
							data: lt.xIn,
							realShape: real,
							label: '← 这一层的输入（未归一化，主残差流）'
						},
						{
							name: 'O_attn',
							shape: [S, D],
							data: lt.attention.out,
							realShape: real,
							label: attnRef
						}
					],
					result: {
						name: 'x₁',
						shape: [S, D],
						data: lt.attnResidual,
						realShape: real,
						label: '← 加回后（归一化前的那份输入还在，只是被更新了）'
					}
				}
			]
		},
		{
			id: 'residual-ffn',
			kind: 'ADD',
			label:
				`FFN 的输出同样加回：x₁ + O_ffn → x₂。x₂ 就是这一层的最终输出，也是下一层的输入` +
				`（最后一层的 x₂ 直接进 LM Head）`,
			formula: 'x_2=x_1+O_{\\mathrm{ffn}}',
			tensors: [
				{
					name: 'residual-ffn-add',
					kind: 'sum',
					shape: [S, D],
					terms: [
						{
							name: 'x₁',
							shape: [S, D],
							data: lt.attnResidual,
							realShape: real,
							label: '← 第 {{step:residual-attn}} 步加回后的残差流'
						},
						{
							name: 'O_ffn',
							shape: [S, D],
							data: lt.ffn.out,
							realShape: real,
							label: ffnRef
						}
					],
					result: {
						name: 'x₂',
						shape: [S, D],
						data: lt.ffnResidual,
						realShape: real,
						label: '← 这一层的输出（下一层 / LM Head 的输入）'
					}
				}
			]
		}
	];
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
	/** 画出"注意力输出 / FFN 输出"的那一步的 id（残差步骤要引用它；对应阶段关掉时为 null） */
	let attnOutStep: string | null = null;
	let ffnOutStep: string | null = null;

	// 先各自建好步骤（FFN 的产出步骤 id 要在残差标签里引用，得先算出来），
	// 再按真实顺序 push：embedding → attention → residual-attn → ffn → residual-ffn → lm-head
	let attnOutput: StageOutput | null = null;
	let ffnOutput: StageOutput | null = null;
	if (!off.has('attention')) {
		// 层级的阶段描述在运行时填上当前插件名——流程图因此永远显示真实插件，
		// 而不是写死的名字。选了执行方式时也一并显示（`MLA · 矩阵吸收合并`）
		const attnSteps = attn.steps(lt.attention, ctxFor(attn.weightKey));
		attnOutStep = producingStepId(lt.attention.out, attnSteps);
		attnOutput = {
			stage: {
				id: 'attention',
				slot: 'attention',
				scope: 'layer',
				depth: 'L2',
				title: 'Attention',
				sub: attn.name,
				desc: attn.desc
			},
			steps: attnSteps
		};
	}
	if (!off.has('ffn-moe')) {
		const ffnSteps = ffnSpec.steps(lt.ffn, ctxFor(ffnSpec.id));
		ffnOutStep = producingStepId(lt.ffn.out, ffnSteps);
		ffnOutput = {
			stage: {
				id: 'ffn-moe',
				slot: 'ffn-moe',
				scope: 'layer',
				depth: 'L2',
				title: 'FFN / MoE',
				sub: ffnSpec.name,
				desc: ffnSpec.desc
			},
			steps: ffnSteps
		};
	}
	/** 残差两页（R1 专属）：`residual-attn` 紧跟 Attention、`residual-ffn` 紧跟 FFN */
	const resSteps = model.residual && !off.has('residual') ? residualSteps(lt, model, attnOutStep, ffnOutStep) : null;
	const residualStage: StageInfo = {
		id: 'residual',
		slot: 'residual',
		scope: 'layer',
		depth: 'L2',
		title: 'Residual',
		desc: '每个子层的输出都加回主残差流（x + 子层输出）：注意力后一次、FFN 后一次。残差让 token 自身的信息不被子层"平均"掉'
	};

	if (!off.has('embedding-lookup')) {
		outputs.push({
			stage: embeddingLookup,
			steps: embeddingLookup.steps(
				{ tokenIds: trace.tokenIds, table: weights.embed, rows: trace.embed },
				cfg
			)
		});
	}
	if (attnOutput) outputs.push(attnOutput);
	if (resSteps) outputs.push({ stage: residualStage, steps: [resSteps[0]] });
	if (ffnOutput) outputs.push(ffnOutput);
	if (resSteps) outputs.push({ stage: residualStage, steps: [resSteps[1]] });
	if (!off.has('lm-head')) {
		outputs.push({
			stage: nextTokenHead,
			steps: nextTokenHead.steps(headDataOf(trace, weights.lmHead), cfg)
		});
	}

	// 所有阶段全关掉时流程是空的——这不算错误（UI 会给提示），
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
