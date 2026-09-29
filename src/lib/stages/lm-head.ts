/**
 * LM Head 阶段：最后一层的隐藏态 → 词表上的分数（logits）→ 概率 → 预测。
 *
 * 两个要点：
 *   1. 预测下一个 token **只需要最后一个位置**的隐藏态，所以这里只算一行
 *      （真实模型里也是一个 `[1, d_model] · [d_model, vocab]` 的矩阵乘）。
 *   2. 词表有 12 个（真实 129280），logits 是一行 12 个数——正好能逐格写出来。
 */
import type { StageSpec } from '$lib/core/flow';
import type { Mat, Vec } from '$lib/core/mat';
import { REAL_R1 } from '$lib/model/config';

export interface HeadData {
	/** 最后一个位置的隐藏态 [d_model] */
	last: Vec;
	/** 词表投影矩阵 [d_model × vocab] */
	table: Mat;
	/** 该位置的 logits [vocab] */
	logits: Vec;
	/** softmax 之后 [vocab] */
	probs: Vec;
	/** 展示概率最高的前几个 */
	topN: number;
}

export const nextTokenHead: StageSpec<HeadData> = {
	id: 'lm-head',
	slot: 'lm-head',
	scope: 'model',
	depth: 'L1',
	title: 'LM Head',
	sub: '隐藏态 → 词表分数 → 下一个 token',
	desc: '取最后一个位置的隐藏态，与词表矩阵相乘得到每个 token 的分数（logits），softmax 之后就是词表上的概率分布。',

	steps({ last, table, logits, probs, topN }, cfg) {
		const { vocab_size: V, d_model: D, seq_len: S } = cfg;
		const lastIdx = S - 1;
		const top = probs
			.map((p, i) => ({ p, i }))
			.sort((a, b) => b.p - a.p)
			.slice(0, topN);

		return [
			{
				id: 'lm-head-project',
				kind: 'PROJECT',
				label: `预测下一个 token 只需要最后一个位置（前面 ${lastIdx} 个位置的输出在这一步用不上）：把它与词表矩阵相乘，得到 ${V} 个 token 各自的分数（logits），softmax 之后就是概率——分数最高的那个就是模型的预测。`,
				formula: 'z=x_{L}\\,W_{\\mathrm{head}},\\qquad p=\\mathrm{softmax}(z)',
				tensors: [
					{
						name: 'head-matmul',
						kind: 'matmul',
						shape: [1, V],
						a: {
							name: `x_L[${lastIdx}]`,
							shape: [1, D],
							data: [last],
							realShape: [1, REAL_R1.d_model],
							label: `← 最后一层第 ${lastIdx} 个位置的隐藏态`
						},
						b: {
							name: 'W_head',
							shape: [D, V],
							data: table,
							realShape: [REAL_R1.d_model, REAL_R1.vocab_size],
							label: '← 词表矩阵：每一列对应一个 token'
						},
						out: {
							name: 'logits',
							shape: [1, V],
							data: [logits],
							realShape: [1, REAL_R1.vocab_size],
							label: '↓ 每个 token 一个分数，还没有归一化'
						}
					},
					{
						name: 'head-probs',
						label: `softmax 之后概率最高的 ${topN} 个（其余 ${V - topN} 个更低）`,
						kind: 'bars',
						shape: [V],
						data: top.map((t) => Number(t.p.toFixed(3))),
						labels: top.map((t) => `#${t.i}`),
						animated: true
					}
				]
			}
		];
	}
};
