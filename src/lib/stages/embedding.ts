/**
 * Embedding 阶段：token id → 向量。
 *
 * 这一步**不是矩阵乘**，是查表——`x_i = W_E[id_i]`。所以用 `lookup` 视图
 * （左边整张表、右边取出来的行），而不是把它写成 `onehot · W_E`。
 *
 * 阶段（stage）与算子插件（SemanticsSpec）的区别：算子插件是"每层各做一次"的，
 * 而 Embedding 整个模型只做一次。见 `core/flow.ts`。
 */
import type { StageSpec } from '$lib/core/flow';
import type { Mat, Vec } from '$lib/core/mat';
import { REAL_R1 } from '$lib/model/config';

export interface EmbeddingData {
	/** token id 序列 */
	tokenIds: number[];
	/** 词表矩阵 [vocab × d_model] */
	table: Mat;
	/** 查表得到的行 [seq × d_model] */
	rows: Vec[];
}

export const embeddingLookup: StageSpec<EmbeddingData> = {
	id: 'embedding-lookup',
	slot: 'embedding',
	scope: 'model',
	depth: 'L1',
	title: 'Embedding',
	sub: 'token id → 向量',
	desc: '文本先切成 token，每个 token 的 id 在词表矩阵里对应一行；查表把那一行取出来，就是整个模型的输入。',

	steps({ tokenIds, table, rows }, cfg) {
		const { vocab_size: V, d_model: D, seq_len: S } = cfg;

		return [
			{
				id: 'embed-lookup',
				kind: 'LOAD',
				label: `文本先切成 ${S} 个 token（教学词表共 ${V} 个，真实 129280）。每个 id 对应 W_E 里的一行，查表把它整行取出来就是输入——这一步「没有任何乘法」，只是按 id 取行。`,
				formula: 'x_i = W_E[\\mathrm{id}_i]',
				tensors: [
					{
						name: 'embed-lookup',
						kind: 'lookup',
						shape: [S, D],
						keyName: 'token id',
						table: {
							name: 'W_E',
							shape: [V, D],
							data: table,
							realShape: [REAL_R1.vocab_size, REAL_R1.d_model]
						},
						keys: tokenIds,
						rows,
						result: {
							name: 'x_0',
							shape: [S, D],
							data: rows,
							realShape: [S, REAL_R1.d_model],
							label: '↓ 整个模型的输入（每层先过 RMSNorm 再进注意力）'
						}
					}
				]
			}
		];
	}
};
