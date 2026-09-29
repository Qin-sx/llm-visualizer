/**
 * 普通前馈网络（Dense FFN）——作为 MoE 的对照。
 *
 * 每个 token 都过同一个网络，计算量正比于 token 数，没有路由、没有稀疏。
 */
import { matmul, randFfnPair, silu, type Mat } from '$lib/core/mat';
import type { SemanticsSpec } from '$lib/core/types';
import { REAL_R1 } from '$lib/model/config';

interface FfnWeights {
	Wup: Mat;
	Wdown: Mat;
}

export interface DenseFfnTrace {
	out: Mat;
	x: Mat;
	/** [seq, intermediate] */
	up: Mat;
	/** SiLU 之后 */
	hidden: Mat;
}

export const denseFfn: SemanticsSpec<DenseFfnTrace> = {
	id: 'dense-ffn',
	slot: 'ffn-moe',
	name: 'Dense FFN',
	desc: '每个 token 都过同一个前馈网络：升维 → SiLU → 降维。计算量与 token 数成正比，没有路由。',
	formula: '\\mathrm{FFN}(x)=\\mathrm{SiLU}(xW_{up})\\,W_{down}',
	defaultDepth: 'L1',
	maxDepth: 'L2',

	makeWeights(rnd, cfg) {
		return randFfnPair(rnd, cfg.d_model, cfg.dense_intermediate);
	},

	compute(x, ctx) {
		const { Wup, Wdown } = ctx.w as FfnWeights;
		const up = matmul(x, Wup);
		const hidden = up.map((row) => row.map(silu));
		return { out: matmul(hidden, Wdown), x, up, hidden };
	},

	steps(trace, ctx) {
		const { seq_len: S, d_model: D, dense_intermediate: I } = ctx.cfg;
		const { Wup, Wdown } = ctx.w as FfnWeights;
		const R = REAL_R1;
		const preview = 6; // 数值行只展示前 6 维

		return [
			{
				id: 'ffn-up',
				kind: 'PROJECT',
				label: `升维：每个 token 都走同一个矩阵，没有选择。真实模型里是 ${R.d_model} → ${R.dense_intermediate}`,
				formula: 'U = X\\,W_{up}',
				tensors: [
					{
						name: 'up-matmul',
						kind: 'matmul',
						shape: [S, I],
						a: {
							name: 'X',
							shape: [S, D],
							data: trace.x,
							realShape: [S, R.d_model],
							handoffId: 'o',
							label: '← 上一层注意力的输出'
						},
						b: {
							name: 'W_up',
							shape: [D, I],
							data: Wup,
							realShape: [R.d_model, R.dense_intermediate]
						},
						out: {
							name: 'U',
							shape: [S, I],
							data: trace.up,
							realShape: [S, R.dense_intermediate]
						}
					}
				]
			},
			{
				id: 'ffn-act',
				kind: 'ACT',
				label: '逐元素 SiLU 激活（只影响每个元素自身，不跨维度混合）',
				formula: 'H = \\mathrm{SiLU}(U) = \\frac{U}{1+e^{-U}}',
				tensors: [
					{
						name: 'act-in',
						label: `激活前 U 的第 0 行（前 ${preview} 维）`,
						kind: 'row',
						shape: [I],
						data: trace.up[0].slice(0, preview)
					},
					{
						name: 'act-out',
						label: `激活后 H 的第 0 行（前 ${preview} 维）`,
						kind: 'row',
						shape: [I],
						data: trace.hidden[0].slice(0, preview)
					}
				]
			},
			{
				id: 'ffn-down',
				kind: 'PROJECT',
				label: `降维：回到主干维度`,
				formula: '\\mathrm{out} = H\\,W_{down}',
				tensors: [
					{
						name: 'down-matmul',
						kind: 'matmul',
						shape: [S, D],
						a: {
							name: 'H',
							shape: [S, I],
							data: trace.hidden,
							realShape: [S, R.dense_intermediate]
						},
						b: {
							name: 'W_down',
							shape: [I, D],
							data: Wdown,
							realShape: [R.dense_intermediate, R.d_model]
						},
						out: {
							name: 'out',
							shape: [S, D],
							data: trace.out,
							realShape: [S, R.d_model]
						}
					}
				]
			}
		];
	}
};
