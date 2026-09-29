/**
 * DeepSeekMoE：共享专家 + 细粒度路由专家。
 *
 * 结构特征（与 DeepSeek R1 真实 config 对应，只是维度缩小）：
 *   - 每个 token 只激活 top-k 个路由专家   → 总参数量 ≠ 计算量
 *   - 另有 1 个**始终激活**的共享专家      → 承载跨上下文共性知识
 */
import {
	matAdd,
	matmul,
	normalizeRows,
	randFfnPair,
	randMat,
	silu,
	softmaxRows,
	topK,
	zeros,
	type Mat
} from '$lib/core/mat';
import type { SemanticsSpec } from '$lib/core/types';
import { REAL_R1 } from '$lib/model/config';

interface FfnWeights {
	Wup: Mat;
	Wdown: Mat;
}

interface MoeWeights {
	Wrouter: Mat;
	experts: FfnWeights[];
	shared: FfnWeights;
}

export interface MoeTrace {
	out: Mat;
	x: Mat;
	/** [seq, num_routed_experts] */
	routerLogits: Mat;
	routerProbs: Mat;
	/** [seq][top_k] */
	topkIdx: number[][];
	topkW: number[][];
	/** [seq][top_k][d_model] */
	expertOut: number[][][];
	routedOut: Mat;
	sharedOut: Mat;
	/** 每个专家分到多少个 token（负载分布） */
	expertLoad: number[];
}

export const deepseekMoe: SemanticsSpec<MoeTrace> = {
	id: 'deepseek-moe',
	slot: 'ffn-moe',
	name: 'DeepSeekMoE（共享 + 路由专家）',
	desc: '每个 token 只激活少数路由专家（top-k），再叠加一个始终激活的共享专家。总参数量 ≠ 计算量。',
	formula:
		'y=\\sum_{i\\in\\mathrm{TopK}}\\tilde{g}_i\\,\\mathrm{Expert}_i(x)+\\mathrm{Expert}_{shared}(x)',
	defaultDepth: 'L2',
	maxDepth: 'L2',

	makeWeights(rnd, cfg) {
		return {
			// 路由权重按行归一化，避免"范数最大的专家通吃"（见 mat.ts 的 normalizeRows）
			Wrouter: normalizeRows(randMat(cfg.d_model, cfg.num_routed_experts, rnd)),
			experts: Array.from({ length: cfg.num_routed_experts }, () =>
				randFfnPair(rnd, cfg.d_model, cfg.moe_intermediate)
			),
			shared: randFfnPair(rnd, cfg.d_model, cfg.moe_intermediate)
		};
	},

	compute(x, ctx) {
		const { num_routed_experts: E, top_k: K, d_model: D } = ctx.cfg;
		const w = ctx.w as MoeWeights;

		const routerLogits = matmul(x, w.Wrouter);
		const routerProbs = softmaxRows(routerLogits);

		const S = x.length;
		const topkIdx: number[][] = [];
		const topkW: number[][] = [];
		const expertOut: number[][][] = [];
		const expertLoad = new Array<number>(E).fill(0);
		const routedOut = zeros(S, D);

		const runExpert = (e: number, vec: number[]): number[] => {
			const up = matmul([vec], w.experts[e].Wup)[0];
			const hid = up.map(silu);
			return matmul([hid], w.experts[e].Wdown)[0];
		};

		for (let t = 0; t < S; t++) {
			const idx = topK(routerProbs[t], K);
			const raw = idx.map((i) => routerProbs[t][i]);
			const sum = raw.reduce((a, b) => a + b, 0) || 1;
			const wts = raw.map((v) => v / sum); // top-k 权重归一化

			topkIdx.push(idx);
			topkW.push(wts);

			const outs: number[][] = [];
			for (let j = 0; j < K; j++) {
				const e = idx[j];
				expertLoad[e] += 1;
				const o = runExpert(e, x[t]);
				outs.push(o);
				for (let i = 0; i < D; i++) routedOut[t][i] += wts[j] * o[i];
			}
			expertOut.push(outs);
		}

		// 共享专家：所有 token 都过
		const sharedHidden = matmul(x, w.shared.Wup).map((row) => row.map(silu));
		const sharedOut = matmul(sharedHidden, w.shared.Wdown);

		return {
			out: matAdd(routedOut, sharedOut),
			x,
			routerLogits,
			routerProbs,
			topkIdx,
			topkW,
			expertOut,
			routedOut,
			sharedOut,
			expertLoad
		};
	},

	steps(trace, ctx) {
		const {
			seq_len: S,
			d_model: D,
			num_routed_experts: E,
			top_k: K,
			moe_intermediate: MI
		} = ctx.cfg;
		const w = ctx.w as MoeWeights;
		const R = REAL_R1;

		// 用 token 0 举例贯穿整条 MoE 流程
		const t0 = 0;
		/**
		 * token 0 选中的 K 个专家，以及**每个专家分到的全部 token**。
		 * 专家计算那一步不是"只看 token 0 走一遍"，而是把该专家的整个 batch
		 * 一起算——这才是 MoE 真实的执行方式（一次批量矩阵乘）。
		 */
		const batches = trace.topkIdx[t0].map((e) => ({
			e,
			tokens: Array.from({ length: S }, (_, t) => t).filter((t) =>
				trace.topkIdx[t].includes(e)
			)
		}));
		/**
		 * 每个专家这一批 token 的完整 FFN 中间量：
		 *   X → U = X·W_up → H = SiLU(U) → O = H·W_down
		 * 三个步骤（升维 / 激活 / 降维）共用同一份计算结果，保证前后一致。
		 */
		const expertBatch = batches.map(({ e, tokens }) => {
			const X = tokens.map((t) => trace.x[t]);
			const U = matmul(X, w.experts[e].Wup);
			const H = U.map((row) => row.map(silu));
			const O = matmul(H, w.experts[e].Wdown);
			return { e, tokens, X, U, H, O };
		});
		const loadLabel = trace.expertLoad.map((n, e) => `E${e}: ${n} token`);

		// top-k 选择的结果：每行只保留概率最高的 K 个（归一化后的门控权重），其余归零
		const g = trace.routerProbs.map((row, t) =>
			row.map((_, e) => {
				const j = trace.topkIdx[t].indexOf(e);
				return j >= 0 ? trace.topkW[t][j] : 0;
			})
		);

		return [
			// ── 1. 路由打分 ───────────────────────────────
			{
				id: 'moe-route',
				kind: 'ROUTE',
				label: `路由器给每个 token 对 ${E} 个专家各打一个分——就是一次 [${S}×${D}] @ [${D}×${E}] 的矩阵乘。真实模型是 ${R.d_model}→${R.num_routed_experts}`,
				formula: 'r = X\\,W_{router}',
				tensors: [
					{
						name: 'route-matmul',
						kind: 'matmul',
						shape: [S, E],
						a: {
							name: 'X',
							shape: [S, D],
							data: trace.x,
							realShape: [S, R.d_model],
							handoffId: 'o',
							label: '← 上一层注意力的输出'
						},
						b: {
							name: 'W_router',
							shape: [D, E],
							data: w.Wrouter,
							realShape: [R.d_model, R.num_routed_experts]
						},
						out: {
							name: 'r',
							shape: [S, E],
							data: trace.routerLogits,
							realShape: [S, R.num_routed_experts]
						}
					}
				]
			},
			// ── 2. softmax 成概率 → top-k 选择（一条链，p 只画一次） ──
			{
				id: 'moe-probs-select',
				kind: 'SOFTMAX',
				label: `先由路由分数逐行算出概率 p，再每行只留概率最高的 ${K} 个专家得到 g（其余不参与计算）`,
				formula:
					'p = \\mathrm{softmax}(r),\\quad \\mathrm{TopK}(p),\\quad g_i=\\frac{p_i}{\\sum_{j\\in\\mathrm{TopK}}p_j}',
				tensors: [
					{
						name: 'probs-select-anim',
						kind: 'transform',
						shape: [S, E],
						op: 'softmax',
						input: {
							name: 'r',
							shape: [S, E],
							data: trace.routerLogits,
							label: `← 第 {{step:moe-route}} 步的路由分数`
						},
						output: {
							name: 'p',
							shape: [S, E],
							data: trace.routerProbs,
							label: '← 每行归一化后（和为 1）'
						},
						// 第二段接着 p 继续，p 不再重画
						then: {
							op: 'top-k',
							result: {
								name: 'g',
								shape: [S, E],
								data: g,
								label: `← 每行只留 top-${K} 的权重，其余归零`
							}
						}
					},
					{
						name: 'expertLoad',
						label: `各专家分到的 token 数（共 ${E} 个专家，真实 ${R.num_routed_experts} 个）`,
						kind: 'bars',
						shape: [E],
						data: trace.expertLoad,
						labels: loadLabel
					}
				]
			},

			// ── 4. 专家升维：每个专家把分给它的**整批 token** 一起算 ──
			{
				id: 'moe-experts',
				kind: 'MATMUL',
				label: `token ${t0} 选中的专家各自一次处理分给它的整批 token（${batches
					.map((b) => `E${b.e}: ${b.tokens.length} 个`)
					.join('，')}）——一次批量矩阵乘，这就是"总参数大但计算量小"的来源`,
				formula: 'U^{(i)}=X^{(i)}W_{up}^{(i)}',
				tensors: expertBatch.map(({ e, tokens, X, U }) => ({
					name: `expert-up-E${e}`,
					row: 1,
					parallel: true,
					kind: 'matmul' as const,
					shape: [tokens.length, MI],
					a: {
						name: `X^(E${e})`,
						shape: [tokens.length, D],
						data: X,
						realShape: [tokens.length, R.d_model],
						label: `← 分给 E${e} 的 ${tokens.length} 个 token`
					},
					b: {
						name: `W_up^(E${e})`,
						shape: [D, MI],
						data: w.experts[e].Wup,
						realShape: [R.d_model, R.moe_intermediate]
					},
					out: {
						name: `U^(E${e})`,
						shape: [tokens.length, MI],
						data: U,
						realShape: [tokens.length, R.moe_intermediate]
					}
				}))
			},

			// ── 5. 专家激活：SiLU（逐元素，两个专家同时算） ──
			{
				id: 'moe-expert-act',
				kind: 'ACT',
				label: `专家内部先做逐元素 SiLU 激活（只影响每个元素自身，不跨维度混合）——两个专家同时算`,
				formula: 'H^{(i)}=\\mathrm{SiLU}\\!\\left(U^{(i)}\\right)=\\frac{U^{(i)}}{1+e^{-U^{(i)}}}',
				tensors: expertBatch.map(({ e, tokens, U, H }) => ({
					name: `expert-act-E${e}`,
					row: 1,
					parallel: true,
					kind: 'transform' as const,
					shape: [tokens.length, MI],
					op: 'silu',
					input: {
						name: `U^(E${e})`,
						shape: [tokens.length, MI],
						data: U,
						label: `← 第 {{step:moe-experts}} 步的升维结果`
					},
					output: {
						name: `H^(E${e})`,
						shape: [tokens.length, MI],
						data: H,
						label: '← 逐元素激活后'
					}
				}))
			},

			// ── 6. 专家降维：W_down 回到主干维度 ──────────
			{
				id: 'moe-expert-down',
				kind: 'PROJECT',
				label: `再用 W_down 降回 ${D} 维——专家是一个完整的小 FFN（升维 → 激活 → 降维），两个专家同时算`,
				formula: 'O^{(i)}=H^{(i)}W_{down}^{(i)}',
				tensors: expertBatch.map(({ e, tokens, H, O }) => ({
					name: `expert-down-E${e}`,
					row: 1,
					parallel: true,
					kind: 'matmul' as const,
					shape: [tokens.length, D],
					a: {
						name: `H^(E${e})`,
						shape: [tokens.length, MI],
						data: H,
						realShape: [tokens.length, R.moe_intermediate],
						label: `← 第 {{step:moe-expert-act}} 步激活后的结果`
					},
					b: {
						name: `W_down^(E${e})`,
						shape: [MI, D],
						data: w.experts[e].Wdown,
						realShape: [R.moe_intermediate, R.d_model]
					},
					out: {
						name: `O^(E${e})`,
						shape: [tokens.length, D],
						data: O,
						realShape: [tokens.length, R.d_model]
					}
				}))
			},

			// ── 7. 按门控权重求和，再叠加共享专家（一条链，routed 只画一次） ──
			{
				id: 'moe-shared-add',
				kind: 'ADD',
				label: '各专家输出按门控权重相加得到 routed，再叠加始终激活的共享专家——两次逐元素相加，形状不变',
				formula:
					'y = \\underbrace{\\textstyle\\sum_i \\tilde{g}_i E_i(x)}_{\\text{路由专家}} + \\underbrace{E_{shared}(x)}_{\\text{共享专家}}',
				tensors: [
					{
						name: 'moe-add',
						kind: 'sum',
						shape: [S, D],
						// 第一段：第 1、2 个选中专家的输出按各自的 g̃ 相加
						terms: [
							{
								name: 'O_1',
								shape: [S, D],
								data: trace.expertOut.map((outs) => outs[0]),
								realShape: [S, R.d_model],
								label: '← 每个 token 第 1 个选中专家的输出'
							},
							{
								name: 'O_2',
								shape: [S, D],
								data: trace.expertOut.map((outs) => outs[1]),
								realShape: [S, R.d_model],
								label: '← 第 2 个选中专家的输出'
							}
						],
						weights: [
							trace.topkW.map((ws) => ws[0]),
							trace.topkW.map((ws) => ws[1])
						],
						result: {
							name: 'routed',
							shape: [S, D],
							data: trace.routedOut,
							realShape: [S, R.d_model],
							label: '← 路由专家的加权和'
						},
						// 第二段：接着 routed 的结果，把共享专家加上去
						then: {
							terms: [
								{
									name: 'shared',
									shape: [S, D],
									data: trace.sharedOut,
									realShape: [S, R.d_model],
									label: `← 共享专家（${D}→${MI}→${D}），所有 token 都过`
								}
							],
							result: {
								name: 'y',
								shape: [S, D],
								data: trace.out,
								realShape: [S, R.d_model],
								label: '← MoE 输出，形状与输入一致'
							}
						}
					}
				]
			}
		];
	}
};
