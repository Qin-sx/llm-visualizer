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
import { matmul, softmaxRow, transpose, type Mat } from '$lib/core/mat';
import { matRefsOf } from '$lib/core/overview';
import type { LayerCtx, SemanticsSpec, Step, TensorView } from '$lib/core/types';
import { attnIdFor, ffnIdFor, getModel, REAL_V4_FLASH, type ModelSpec } from '$lib/model/config';
import { forward, type LayerTrace, type ModelTrace } from '$lib/model/forward';
import type { HcMixTrace, HcWriteTrace } from '$lib/model/hc';
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
 * Residual 只在 `model.residual !== 'none'` 的模型里产出步骤：R1 是普通残差两页、
 * V4 是 mHC 五页。`+page.svelte` 会按标记把开关藏掉。
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

/**
 * mHC（V4）残差阶段的五页：`pre-attn` / `post-attn` / `pre-ffn` /
 * `post-ffn` / `head`。与 R1 普通残差一样属于 `residual` 阶段，拆在真实位置（见
 * `buildLlmFlow` 的 push 顺序）。子层插件不用改——mHC 只发生在它们**外面**。
 */
function mhcSteps(
	lt: LayerTrace,
	trace: ModelTrace,
	model: ModelSpec,
	attnOutId: string | null,
	ffnOutId: string | null
): Step[] {
	const cfg = model.cfg;
	const hc = cfg.hc_mult!;
	const S = cfg.seq_len;
	const D = cfg.d_model;
	const R = model.real;
	const real: [number, number] = [S, R.d_model];
	const mhc = lt.mhc!;
	const hcHead = trace.hcHead!;
	const attnRef = attnOutId ? `← 第 {{step:${attnOutId}}} 步的注意力输出 y` : '← 注意力输出 y';
	const ffnRef = ffnOutId ? `← 第 {{step:${ffnOutId}}} 步的 FFN 输出 y` : '← FFN 输出 y';
	const mixN = (2 + hc) * hc;
	// 真实 hc_mult / mix（标注用：真实 4 条流，演示 2 条）
	const realHc = REAL_V4_FLASH.hc_mult;
	const realMix = (2 + realHc) * realHc;
	/** `fn` 的真实形状标注（界面上提示真实与演示各是多少） */
	const fnReal = `真实 [(2+${realHc})·${realHc}, ${realHc}×${R.d_model}] = [${realMix}, ${realHc * R.d_model}]（此处演示 [${mixN}, ${hc * D}]）`;
	const headFnReal = `真实 [${realHc}, ${realHc}×${R.d_model}] = [${realHc}, ${realHc * R.d_model}]（此处演示 [${hc}, ${hc * D}]）`;
	/** 真实是 `hc_mult` 条流，展示只画 `hc` 条——每个 mHC 页面都标注（用户定的简化） */
	const hcNote = `**真实 ${REAL_V4_FLASH.hc_mult} 条流，此处只演示 ${hc} 条**`;
	const streamNames = Array.from({ length: hc }, (_, k) => `r_${k}`).join('..');
	const initNote = lt.layer === 0 ? `（第 0 层的 ${hc} 条流是**同一份 embedding**，真实如此）` : '';

	/** 一条流的 MatRef（r_k） */
	const streamRef = (s: Mat, k: number, label: string) => ({
		name: `r_${k}`,
		shape: [S, D],
		data: s,
		realShape: real,
		label
	});

	/**
	 * pre-mix 那页的公共视图：`hc` 条流加权求和 → h_raw，再过 RMSNorm → 子层输入。
	 * 求和与归一化**同一行**（`row: 1`）：StepPanel 默认按"每列最后一块矩阵的顶边"对齐，
	 * sum 列的最后一块是 h_raw、变换列的最后一块是 X → 两者顶边自然对齐（**不要**用 `vcenter`，
	 * 那是对"内容跨度中心"，会把 X 挪到 r_1/r_2 之间去）。
	 * 每条流右侧挂它的 pre 门控列（`termSide`），矩阵缩到 24px 给并排让位。
	 */
	const preViews = (
		prefix: string,
		streams: Mat[],
		mix: HcMixTrace,
		fnName: string,
		outLabel: string,
		gatesId: string
	): TensorView[] => [
		{
			name: `${prefix}-sum`,
			kind: 'sum',
			shape: [S, D],
			row: 1,
			cellSize: 24,
			terms: streams.map((s, k) =>
				streamRef(
					s,
					k,
					`第 ${k} 条残差流${k === 0 && lt.layer === 0 ? '（第 0 层初始是同一份 embedding）' : ''}`
				)
			),
			// 逐行系数 = pre 门控（`weights[j][t]` 是第 j 项在第 t 行的系数）
			weights: Array.from({ length: hc }, (_, k) => mix.pre.map((row) => row[k])),
			// 每项右侧竖挂它的 pre 列：`r_k | pre[:,k]`（读门矩阵，逐行揭示和该项同步）
			termSide: Array.from({ length: hc }, (_, k) => ({
				name: `pre[:,${k}]`,
				shape: [S, 1],
				data: mix.pre.map((row) => [row[k]]),
				realShape: [S, 1],
				label: `← 第 {{step:${gatesId}}} 步的读门（σ+eps，逐 token）`
			})),
			result: {
				name: 'h_raw',
				shape: [S, D],
				data: mix.combined,
				realShape: real,
				label: '← Σ pre_k·r_k（未归一化）'
			}
		},
		{
			name: `${prefix}-norm`,
			kind: 'transform',
			shape: [S, D],
			row: 1,
			cellSize: 24,
			hideInput: true,
			op: 'RMSNorm',
			input: { name: 'h_raw', shape: [S, D], data: mix.combined, realShape: real },
			output: {
				name: 'X',
				shape: [S, D],
				data: mix.normed,
				realShape: real,
				label: outLabel
			}
		},
		{
			name: `${prefix}-note`,
			kind: 'shape',
			wide: true,
			shape: [S, mixN],
			note: `门控来源：mixes = concat(${streamNames}) @ ${fnName}ᵀ · rsqrt（整块 RMS 缩放）→ [S, ${mixN}]，前 ${hc} 列 σ 成 pre（+eps，**不归一化**）。**pre 是逐 token 的门控**：每列 pre[:,k] 挂在第 k 条流右边，每个数都是"那一行（token）"的 pre 值，按行乘、不是整矩阵乘一个数；pre 不是 softmax 路由，两者别混。${fnName} 真实 ${fnReal}`
		}
	];

	/**
	 * post 写回那页的公共视图：`hc` 条流**同结构**并行写回（r′_k = post_k·y + Σ_j comb_kj·r_j）。
	 *
	 * 和第 2 页（pre 混合）同款布局：每条流一个块，操作数各乘自己的门控、**门控列贴在操作数右边**
	 * （`termSide`：y|post_k、r_0|comb[k,0]、r_1|comb[k,1]）。两条流**共用同一份** y / r_0 / r_1
	 * （校验对"并行分支共用输入"放行），所以两边画的是同一份数据、各按自己的门控缩放。
	 */
	const postViews = (
		prefix: string,
		mix: HcMixTrace,
		write: HcWriteTrace,
		y: Mat,
		oldStreams: Mat[],
		subRef: string,
		gatesId: string
	): TensorView[] => [
		...Array.from({ length: hc }, (_, k) => ({
			name: `${prefix}-w${k}`,
			kind: 'sum' as const,
			shape: [S, D],
			row: 1,
			parallel: true,
			group: `w${k}`,
			// 门控列贴着操作数（第 2 页同款）：y|post_k、r_0|comb[k,0]、r_1|comb[k,1]
			termSide: [
				{
					name: `post_${k}`,
					shape: [S, 1],
					data: mix.post.map((row) => [row[k]]),
					realShape: [S, 1],
					label: `← 写门（第 {{step:${gatesId}}} 步，2σ ∈ (0,2)）`
				},
				{
					name: `comb[${k},0]`,
					shape: [S, 1],
					data: mix.comb.map((row) => [row[k * hc + 0]]),
					realShape: [S, 1],
					label: `← comb 第 ${k} 行第 0 列（第 {{step:${gatesId}}} 步，Sinkhorn）`
				},
				{
					name: `comb[${k},1]`,
					shape: [S, 1],
					data: mix.comb.map((row) => [row[k * hc + 1]]),
					realShape: [S, 1],
					label: `← comb 第 ${k} 行第 1 列（第 {{step:${gatesId}}} 步，Sinkhorn）`
				}
			],
			terms: [
				{
					name: 'y',
					shape: [S, D],
					data: y,
					realShape: real,
					label: `← ${subRef}（两条流共用同一份）`
				},
				{
					name: 'r_0',
					shape: [S, D],
					data: oldStreams[0],
					realShape: real,
					label: '← 第 0 条旧流（两条流共用同一份）'
				},
				{
					name: 'r_1',
					shape: [S, D],
					data: oldStreams[1],
					realShape: real,
					label: '← 第 1 条旧流（两条流共用同一份）'
				}
			],
			// 逐行系数 = 门控：y × post_k、r_0 × comb[k,0]、r_1 × comb[k,1]
			weights: [
				mix.post.map((row) => row[k]),
				mix.comb.map((row) => row[k * hc + 0]),
				mix.comb.map((row) => row[k * hc + 1])
			],
			result: {
				name: `r_${k}′`,
				shape: [S, D],
				data: write.newStreams[k],
				realShape: real,
				label: `← 写回后（post_${k}·y + Σ_j comb_${k}j·r_j）`
			}
		})),
		{
			name: `${prefix}-note`,
			kind: 'shape',
			wide: true,
			shape: [S, hc * hc],
			note: `两条流**同结构**（各 3 个操作数 × 门控，和第 2 页的 pre 混合同款布局）：r′_k = post_k·y + comb[k,0]·r_0 + comb[k,1]·r_1。y / r_0 / r_1 是**同一份**（两条流共用，各按自己的门控缩放）。comb 每 token 一个 ${hc}×${hc} 矩阵，Sinkhorn 归一化到行和 ≈ 列和 ≈ 1（"流形约束"，真实迭代 ${cfg.hc_sinkhorn_iters} 次）；post = 2σ ∈ (0,2)——写回能增强、不只衰减。`
		}
	];

	/** 门控计算页：concat(hc 条流) → hc_fnᵀ 投影（**真正的矩阵乘**）→ ×rsqrt → mixes → σ+eps → pre */
	const gatesStep = (
		id: string,
		streams: Mat[],
		mix: HcMixTrace,
		fnName: string,
		preRef: string,
		postRef: string
	): Step => {
		const concat: Mat = Array.from({ length: S }, (_, i) => streams.flatMap((s) => s[i]));
		// sglang 里 `mixes = F.linear(x_flat, hc_fn) * rsqrt`——F.linear 就是 x_flat @ hc_fnᵀ。
		// 先把"投影结果（未缩放）"算出来画成真矩阵乘，再 ×rsqrt 得到 mixes。
		const fnT: Mat = transpose(mix.fn); // [hc·D, mix]
		const raw: Mat = matmul(concat, fnT); // [S, mix]
		/** mixes 列三段：pre（前 hc 列）/ post（中 hc 列）/ comb（后 hc² 列），画框标注 */
		const gateBands = [
			{ axis: 'col' as const, from: 0, to: hc - 1, tone: '#38bdf8' }, // 蓝 = pre（读门）
			{ axis: 'col' as const, from: hc, to: 2 * hc - 1, tone: '#fbbf24' }, // 琥珀 = post（写门）
			{ axis: 'col' as const, from: 2 * hc, to: mixN - 1, tone: '#a78bfa' } // 紫 = comb（残差矩阵）
		];
		return {
			id,
			kind: 'PROJECT',
			label:
				`门控从哪来：${hc} 条流首尾拼成一行（concat）→ **真正的矩阵乘** ${fnName}ᵀ（逐格动画）→ 投影结果 → ` +
				`×rsqrt（整块 RMS 缩放）→ mixes [S, ${mixN}]（**框里按列分三段**：蓝=pre、琥珀=post、紫=comb）→ ` +
				`**三个门同时从 mixes 出来**：前 ${hc} 列 σ+eps → pre（读门）、中 ${hc} 列 2σ → post（写门）、` +
				`后 ${hc * hc} 列 Sinkhorn → comb（残差矩阵）。${hcNote}`,
			formula:
				'\\mathrm{pre}=\\sigma\\!\\left(\\mathrm{concat}(r_0,\\dots,r_3)\\,W_{\\mathrm{hc}}^{\\top}\\cdot\\mathrm{rsqrt}\\right)+\\epsilon',
			tensors: [
				{
					name: `${id}-mm`,
					kind: 'matmul',
					shape: [S, mixN],
					row: 1,
					// 与右侧 mixes 的变换同 cellSize（24）——同一行的矩阵大小一致、顶边才对得齐
					cellSize: 24,
					a: {
						name: 'concat',
						shape: [S, hc * D],
						data: concat,
						realShape: [S, realHc * R.d_model],
						label: `← ${hc} 条流首尾拼接（${S}×${hc * D}，真实 ${realHc}×${R.d_model}）`
					},
					b: {
						name: `${fnName}ᵀ`,
						shape: [hc * D, mixN],
						data: fnT,
						realShape: [realHc * R.d_model, realMix],
						label: `← ${fnName} 转置（checkpoint 名 hc_attn_fn / hc_ffn_fn 对应各自那份；${fnReal}）`
					},
					out: {
						name: '投影结果',
						shape: [S, mixN],
						data: raw,
						realShape: [S, realMix],
						label: '← 矩阵乘结果（**未缩放**）；下一步整块 RMS 缩放'
					}
				},
				{
					name: `${id}-scale`,
					kind: 'transform',
					shape: [S, mixN],
					row: 1,
					cellSize: 24,
					hideInput: true,
					op: '× rsqrt（整块 RMS 缩放）',
					input: {
						name: '投影结果',
						shape: [S, mixN],
						data: raw,
						realShape: [S, realMix]
					},
					output: {
						name: 'mixes',
						shape: [S, mixN],
						data: mix.mixes,
						realShape: [S, realMix],
						// 三色列框：蓝 = pre（前 hc 列）、琥珀 = post（中 hc 列）、紫 = comb（后 hc² 列）
						bands: gateBands,
						label: '← 门控 logits（投影 × rsqrt，[S, 8]）——**框里按列分三段**：蓝 = pre（读门）、琥珀 = post（写门）、紫 = comb（残差矩阵）'
					}
				},
				{
					name: `${id}-sigma`,
					kind: 'transform',
					shape: [S, hc],
					row: 2,
					parallel: true,
					cellSize: 24,
					hideInput: true,
					op: `σ + eps（前 ${hc} 列）`,
					input: {
						name: 'mixes',
						shape: [S, mixN],
						data: mix.mixes,
						realShape: [S, realMix]
					},
					output: {
						name: 'pre',
						shape: [S, hc],
						data: mix.pre,
						realShape: [S, realHc],
						label: `← 读门（逐 token，∈(ε,1+ε)）；下一页把每列挂到对应流的右边（第 {{step:${preRef}}} 步）`
					}
				},
				{
					name: `${id}-post`,
					kind: 'transform',
					shape: [S, hc],
					row: 2,
					parallel: true,
					cellSize: 24,
					hideInput: true,
					op: `2·σ（中 ${hc} 列）`,
					input: {
						name: 'mixes',
						shape: [S, mixN],
						data: mix.mixes,
						realShape: [S, realMix]
					},
					output: {
						name: 'post',
						shape: [S, hc],
						data: mix.post,
						realShape: [S, realHc],
						label: `← 写门（逐 token，∈(0,2)，×2 是"能增强不衰减"）；写回页用它（第 {{step:${postRef}}} 步）`
					}
				},
				{
					name: `${id}-comb`,
					kind: 'transform',
					shape: [S, hc * hc],
					row: 2,
					parallel: true,
					cellSize: 24,
					hideInput: true,
					op: `Sinkhorn（后 ${hc * hc} 列 → ${hc}×${hc}）`,
					input: {
						name: 'mixes',
						shape: [S, mixN],
						data: mix.mixes,
						realShape: [S, realMix]
					},
					output: {
						name: 'comb',
						shape: [S, hc * hc],
						data: mix.comb,
						realShape: [S, realHc * realHc],
						label: `← 残差矩阵（逐 token 一个 ${hc}×${hc}，Sinkhorn 后行和/列和 ≈ 1）；写回页用它（第 {{step:${postRef}}} 步）`
					}
				},
				{
					name: `${id}-note`,
					kind: 'shape',
					wide: true,
					shape: [S, mixN],
					note: `${fnName} ${fnReal}。×rsqrt（整块 RMS 缩放）先算，然后**三个门同时从 mixes 出来**（并行）：前 ${hc} 列（蓝）σ+eps → **pre**（∈(ε,1+ε)）、中 ${hc} 列（琥珀）2σ → **post**（∈(0,2)）、后 ${hc * hc} 列（紫）重排 ${hc}×${hc} + Sinkhorn（行/列交替归一 20 次）→ **comb**（双随机）。pre 到读门、post/comb 到写回。`
				}
			]
		};
	};

	return [
		// ── 1. 注意力前：先算门控（pre 从哪来） ──
		gatesStep('mhc-gates-attn', mhc.streamsIn, mhc.attnMix, 'W_hc^attn', 'mhc-pre-attn', 'mhc-post-attn'),
		// ── 2. 注意力前：4 条流 pre 混合成 1 条 ──
		{
			id: 'mhc-pre-attn',
			kind: 'ADD',
			label: `mHC 的"读"：${hc} 条并行残差流先用学出来的门控 pre 混合成 1 条（每条流右边那列就是它的 pre），再过 RMSNorm 喂给注意力${initNote}。${hcNote}`,
			formula: 'h=\\mathrm{RMSNorm}\\!\\left(\\textstyle\\sum_{k}\\mathrm{pre}_k\\,r_k\\right)',
			tensors: preViews('mhc-pre-attn', mhc.streamsIn, mhc.attnMix, 'W_hc^attn', '↓ 注意力的输入 X（就是注意力步骤里的 X）', 'mhc-gates-attn')
		},
		// ── 3. 注意力后：输出写回 hc 条流 ──
		{
			id: 'mhc-post-attn',
			kind: 'ADD',
			label: `mHC 的"写"：注意力输出 y 按写门 post 分发回每条流，旧流再按 comb 跨流重分配。${hc} 条流同时写回。${hcNote}`,
			formula: "r'_k=\\mathrm{post}_k\\,y+\\textstyle\\sum_{j}\\mathrm{comb}_{kj}\\,r_j",
			tensors: postViews('mhc-post-attn', mhc.attnMix, mhc.attnWrite, lt.attention.out, mhc.streamsIn, attnRef, 'mhc-gates-attn')
		},
		// ── 4. FFN 前：再算一次门控（换 hc_ffn_fn） ──
		gatesStep('mhc-gates-ffn', mhc.streamsAfterAttn, mhc.ffnMix, 'W_hc^ffn', 'mhc-pre-ffn', 'mhc-post-ffn'),
		// ── 5. FFN 前：再混合一次 ──
		{
			id: 'mhc-pre-ffn',
			kind: 'ADD',
			label: `FFN 前同样"读"一次：把注意力写回后的 ${hc} 条流按**另一组门控**（W_hc^ffn，checkpoint 名 hc_ffn_fn）混合成 1 条，过 RMSNorm 喂给 MoE。${hcNote}`,
			formula: 'h=\\mathrm{RMSNorm}\\!\\left(\\textstyle\\sum_{k}\\mathrm{pre}_k\\,r_k\\right)',
			tensors: preViews('mhc-pre-ffn', mhc.streamsAfterAttn, mhc.ffnMix, 'W_hc^ffn', '↓ FFN 的输入 X（就是 FFN 步骤里的 X）', 'mhc-gates-ffn')
		},
		// ── 6. FFN 后：输出写回 ──
		{
			id: 'mhc-post-ffn',
			kind: 'ADD',
			label: `FFN 输出同样写回：r″_k = post_k·y + Σ_j comb_kj·r_j。写回后的 ${hc} 条流就是下一层的输入。${hcNote}`,
			formula: "r''_k=\\mathrm{post}_k\\,y+\\textstyle\\sum_{j}\\mathrm{comb}_{kj}\\,r_j",
			tensors: postViews('mhc-post-ffn', mhc.ffnMix, mhc.ffnWrite, lt.ffn.out, mhc.streamsAfterAttn, ffnRef, 'mhc-gates-ffn')
		},
		// ── 7. 模型末尾：hc_head 压回 1 条 ──
		{
			id: 'mhc-head',
			kind: 'ADD',
			label: `模型末尾把 ${hc} 条流压回 1 条：hc_head 用学出来的门控（sigmoid + eps，**无 Sinkhorn**——只做"选加权平均"、不写回）${lt.layer === 0 ? '' : `（这里演示第 ${lt.layer + 1} 层写回后的流；真实在最后一层之后压回）`}。${hcNote}`,
			formula: 'y=\\textstyle\\sum_{k}\\mathrm{head\\_pre}_k\\, r_k',
			tensors: [
				{
					name: 'mhc-head-sum',
					kind: 'sum',
					shape: [S, D],
					terms: mhc.streamsAfterFfn.map((s, k) =>
						streamRef(s, k, `← 第 ${k} 条流（最后一次写回后）`)
					),
					weights: Array.from({ length: hc }, (_, k) => hcHead.pre.map((row) => row[k])),
					result: {
						name: 'y',
						shape: [S, D],
						data: hcHead.out,
						realShape: real,
						label: '↓ 最终表示，进 LM Head'
					}
				},
				{
					name: 'mhc-head-note',
					kind: 'shape',
					wide: true,
					shape: [S, hc],
					note: `hc_head 与 pre-mix 同构（拼一行 → 投影 → RMS 缩放 → σ+eps 门控 → 加权平均），但没有写回、没有 Sinkhorn。W_hc^head（checkpoint 名 hc_head_fn）真实 ${headFnReal}`
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
	/** 残差步骤：R1 两页（`residual-attn` / `residual-ffn`）；V4 五页（mHC，见 `mhcSteps`） */
	const plainRes = model.residual === 'plain' && !off.has('residual') ? residualSteps(lt, model, attnOutStep, ffnOutStep) : null;
	const mhcRes = model.residual === 'mhc' && !off.has('residual') ? mhcSteps(lt, trace, model, attnOutStep, ffnOutStep) : null;
	const residualStage: StageInfo = {
		id: 'residual',
		slot: 'residual',
		scope: 'layer',
		depth: 'L2',
		title: 'Residual',
		desc:
			model.residual === 'mhc'
				? 'mHC（V4）：残差流是 4 条并行流（此处演示 2 条），每个子层前 pre 混合成 1 条、算完 post 写回，模型末尾 hc_head 压回 1 条'
				: '每个子层的输出都加回主残差流（x + 子层输出）：注意力后一次、FFN 后一次。残差让 token 自身的信息不被子层"平均"掉'
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
	// 残差步骤拆在**真实位置**（buildFlow 会把同一阶段 id 的多个输出并成一个流程图节点）：
	//   plain：attn → residual-attn → ffn → residual-ffn
	//   mhc：mhc-gates-attn → mhc-pre-attn → attn → mhc-post-attn
	//        → mhc-gates-ffn → mhc-pre-ffn → ffn → mhc-post-ffn → mhc-head
	if (mhcRes) {
		outputs.push({ stage: residualStage, steps: [mhcRes[0]] }); // mhc-gates-attn（注意力前）
		outputs.push({ stage: residualStage, steps: [mhcRes[1]] }); // mhc-pre-attn
	}
	if (attnOutput) outputs.push(attnOutput);
	if (plainRes) outputs.push({ stage: residualStage, steps: [plainRes[0]] }); // 普通残差：紧跟注意力
	if (mhcRes) outputs.push({ stage: residualStage, steps: [mhcRes[2]] }); // mhc-post-attn
	if (ffnOutput) outputs.push(ffnOutput);
	if (plainRes) outputs.push({ stage: residualStage, steps: [plainRes[1]] }); // 普通残差：紧跟 FFN
	if (mhcRes) {
		outputs.push({ stage: residualStage, steps: [mhcRes[3]] }); // mhc-gates-ffn（FFN 前）
		outputs.push({ stage: residualStage, steps: [mhcRes[4]] }); // mhc-pre-ffn
		outputs.push({ stage: residualStage, steps: [mhcRes[5]] }); // mhc-post-ffn
		outputs.push({ stage: residualStage, steps: [mhcRes[6]] }); // mhc-head
	}
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
