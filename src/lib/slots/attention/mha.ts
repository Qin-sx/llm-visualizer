/**
 * 最简单的注意力：Multi-Head Attention。
 *
 * 这里是**最简单的**参考实现：没有 KV 压缩（那是 MLA 的事），也没有分块（那是 FlashAttention 的事）。
 * 后续新增 mla.ts / flash 实现时，只需实现同一套 SemanticsSpec 接口并 register 即可。
 *
 * 步骤划分：
 *   1~3  Q / K / V 三个投影，各自一页（便于用跨步骤流转动画讲清"谁是谁"）
 *   4    attention 核心计算**合并到一页**：Q·Kᵀ → 掩码 → softmax → P·V → 输出投影
 *
 * 表示策略：矩阵直接画出来、逐格写真实数字、标注真实尺寸与缩小倍数；
 * 动画由 progress 驱动（纯函数），可本地播放、可拖动进度条。
 */
import {
	matmul,
	mergeHeads,
	randMat,
	softmaxRow,
	splitHeads,
	transpose,
	type Mat
} from '$lib/core/mat';
import type { SemanticsSpec } from '$lib/core/types';
import { REAL_R1 } from '$lib/model/config';

interface MhaWeights {
	Wq: Mat;
	Wk: Mat;
	Wv: Mat;
	Wo: Mat;
}

export interface MhaTrace {
	out: Mat;
	x: Mat;
	/** 投影后、按头拆之前：[seq, d_model] */
	qFlat: Mat;
	kFlat: Mat;
	vFlat: Mat;
	/** [head][seq][head_dim] */
	q: number[][][];
	k: number[][][];
	v: number[][][];
	/** [head][seq][seq] 原始分数（未掩码，保留有限值） */
	scores: number[][][];
	/** [head][seq][seq] 掩码后（j>i 处为 −Infinity） */
	maskedScores: number[][][];
	/** [head][seq][seq] softmax 之后，下三角结构 */
	probs: number[][][];
	headOut: number[][][];
	concat: Mat;
	/** [seq][seq] 因果掩码：1 = 允许，0 = 屏蔽 */
	mask: Mat;
	causal: boolean;
}

const NEG_INF = Number.NEGATIVE_INFINITY;

export const mha: SemanticsSpec<MhaTrace> = {
	id: 'mha',
	slot: 'attention',
	name: '多头注意力 (MHA)',
	desc: '每个头独立做 Q·Kᵀ → 掩码 → softmax → 加权求和 V，再拼接后过输出投影。最基础：无 KV 压缩、无稀疏选择。',
	formula: '\\mathrm{Attn}(Q,K,V)=\\mathrm{softmax}\\!\\left(\\frac{QK^{\\top}}{\\sqrt{d_h}}+M\\right)V',
	defaultDepth: 'L2',
	maxDepth: 'L2',

	makeWeights(rnd, cfg) {
		const d = cfg.d_model;
		return {
			Wq: randMat(d, d, rnd),
			Wk: randMat(d, d, rnd),
			Wv: randMat(d, d, rnd),
			Wo: randMat(d, d, rnd)
		};
	},

	compute(x, ctx) {
		const { num_heads: H, head_dim: dh, seq_len: S, causal } = ctx.cfg;
		const { Wq, Wk, Wv, Wo } = ctx.w as MhaWeights;

		const qFlat = matmul(x, Wq);
		const kFlat = matmul(x, Wk);
		const vFlat = matmul(x, Wv);

		const q = splitHeads(qFlat, H, dh);
		const k = splitHeads(kFlat, H, dh);
		const v = splitHeads(vFlat, H, dh);

		const scale = 1 / Math.sqrt(dh);
		const scores: number[][][] = [];
		const maskedScores: number[][][] = [];
		const probs: number[][][] = [];
		const headOut: number[][][] = [];

		for (let h = 0; h < H; h++) {
			const raw = matmul(q[h], transpose(k[h])).map((row) => row.map((z) => z * scale));
			const masked = causal
				? raw.map((row, i) => row.map((val, j) => (j <= i ? val : NEG_INF)))
				: raw;
			const p = masked.map(softmaxRow);
			scores.push(raw);
			maskedScores.push(masked);
			probs.push(p);
			headOut.push(matmul(p, v[h]));
		}

		const concat = mergeHeads(headOut, dh);
		const mask = Array.from({ length: S }, (_, i) =>
			Array.from({ length: S }, (_, j) => (causal && j > i ? 0 : 1))
		);

		return {
			out: matmul(concat, Wo),
			x,
			qFlat,
			kFlat,
			vFlat,
			q,
			k,
			v,
			scores,
			maskedScores,
			probs,
			headOut,
			concat,
			mask,
			causal: !!causal
		};
	},

	steps(trace, ctx) {
		const { num_heads: H, head_dim: dh, seq_len: S, d_model: D } = ctx.cfg;
		const { Wq, Wk, Wv, Wo } = ctx.w as MhaWeights;
		const heads = Array.from({ length: H }, (_, h) => `head ${h} [${S} × ${S}]`);
		const focusRow = Math.floor(S / 2);

		// 真实尺寸标注：序列长度是我们选的示例长度（真实模型可到 16 万），
		// 因此只有"列"（特征维）会显示缩小倍数。
		const realSeq = S;
		const R = REAL_R1;

		/** 输入 X 在三个投影步骤里复用，用同一个 handoffId 表示"同一个 X" */
		const X_REF = {
			name: 'X',
			shape: [S, D],
			data: trace.x,
			realShape: [realSeq, R.d_model],
			handoffId: 'x'
		};

		/** 投影权重在"产出"和"使用"两处都会出现，形状说明复用 */
		const projWeight = (name: string, data: Mat) => ({
			name,
			shape: [D, D],
			data,
			realShape: [R.d_model, R.d_model]
		});

		return [
			// ── 1. Q 投影 ─────────────────────────────────
			{
				id: 'attn-q',
				kind: 'PROJECT',
				label: `把输入 X 投影成 Q —— 就是一次矩阵乘`,
				formula: 'Q=XW_Q',
				tensors: [
					{
						name: 'q-proj',
						kind: 'matmul',
						shape: [S, D],
						a: X_REF,
						b: projWeight('W_Q', Wq),
						out: {
							name: 'Q',
							shape: [S, D],
							data: trace.qFlat,
							realShape: [realSeq, R.d_model],
							handoffId: 'q',
							label: '↓ 第 {{step:attn-score}} 步会用到它'
						}
					},
					{
						name: 'Q / K / V',
						label: '算完后按头切开',
						kind: 'shape',
						shape: [S, H, dh],
						note: `Q/K/V 各切成 ${H} 个头，每头 ${dh} 维（真实 ${R.head_dim} 维）· ${H} × ${dh} = ${D}`
					}
				]
			},

			// ── 2. K 投影 ─────────────────────────────────
			{
				id: 'attn-k',
				kind: 'PROJECT',
				label: `K 的计算过程和上一步的 Q 完全相同，只是权重矩阵换成 W_K`,
				formula: 'K=XW_K',
				tensors: [
					{
						name: 'k-proj',
						kind: 'matmul',
						shape: [S, D],
						a: X_REF,
						b: projWeight('W_K', Wk),
						out: {
							name: 'K',
							shape: [S, D],
							data: trace.kFlat,
							realShape: [realSeq, R.d_model],
							handoffId: 'k',
							label: '↓ 第 {{step:attn-score}} 步会用到它'
						}
					}
				]
			},

			// ── 3. V 投影 ─────────────────────────────────
			{
				id: 'attn-v',
				kind: 'PROJECT',
				label: `V 也一样：同一个 X，换成 W_V`,
				formula: 'V=XW_V',
				tensors: [
					{
						name: 'v-proj',
						kind: 'matmul',
						shape: [S, D],
						a: X_REF,
						b: projWeight('W_V', Wv),
						out: {
							name: 'V',
							shape: [S, D],
							data: trace.vFlat,
							realShape: [realSeq, R.d_model],
							handoffId: 'v',
							label: '↓ 第 {{step:attn-mask-softmax-av}} 步会用到它'
						}
					}
				]
			},

			// ── 4. 算注意力分数 ───────────────────────────
			{
				id: 'attn-score',
				kind: 'MATMUL',
				label: `第 {{step:attn-q}} 步的 Q 和第 {{step:attn-k}} 步的 K 在这里相遇：每个头各自算 Q·Kᵀ，再除以 √${dh} 缩放。输出第 i 行第 j 列 = 第 i 个 token 对第 j 个 token 的关注分数`,
				formula: 'S_h=\\frac{Q_h K_h^{\\top}}{\\sqrt{d_h}}',
				tensors: [
					{
						name: 'score-matmul',
						kind: 'matmul',
						shape: [S, S],
						a: {
							name: 'Q_h',
							shape: [S, dh],
							data: trace.q[0],
							realShape: [realSeq, R.head_dim],
							handoffId: 'q',
							label: '← 来自第 {{step:attn-q}} 步算出的 Q'
						},
						b: {
							name: 'K_hᵀ',
							shape: [dh, S],
							data: transpose(trace.k[0]),
							realShape: [R.head_dim, realSeq],
							handoffId: 'k',
							label: '← 来自第 {{step:attn-k}} 步算出的 K'
						},
						out: {
							name: 'S_h',
							shape: [S, S],
							data: trace.scores[0],
							realShape: [realSeq, realSeq]
						},
						scaleNote: `1/√${dh} = ${(1 / Math.sqrt(dh)).toFixed(2)}`
					},
					{
						name: 'all-head-scores',
						label: `${H} 个头各算各的（上面演示 head 0 的过程）`,
						kind: 'tiles',
						shape: [H, S, S],
						data: trace.scores,
						tileLabels: heads
					}
				]
			},

			// ── 5. 掩码 → softmax → P·V（一条链，S_h / P_h 各只画一次） ──
			{
				id: 'attn-mask-softmax-av',
				kind: 'MASK',
				label: `掩码 → softmax → 用权重对 V 加权求和：先把 j > i 的分数置为 −∞，再逐行归一化成权重 P，最后 P·V 得到本头的输出。三段按先后顺序算`,
				formula:
					'S_{ij}=-\\infty\\ (j>i)\\ \\Rightarrow\\ P_{h,ij}=\\frac{e^{S_{ij}}}{\\sum_{j}e^{S_{ij}}}\\ \\Rightarrow\\ O_h=P_hV_h',
				tensors: [
					{
						name: 'mask-softmax-av',
						kind: 'transform',
						shape: [S, S],
						op: 'softmax',
						// 掩码直接作用在 S_h 这个块上（S_h 只画一次）
						preMask: trace.causal ? trace.mask : undefined,
						input: {
							name: 'S_h',
							shape: [S, S],
							data: trace.scores[0],
							realShape: [realSeq, realSeq],
							label: trace.causal
								? `← 第 {{step:attn-score}} 步算出的分数，先掩码：j > i → −∞`
								: `← 第 {{step:attn-score}} 步算出的分数`
						},
						output: {
							name: 'P_h',
							shape: [S, S],
							data: trace.probs[0],
							realShape: [realSeq, realSeq],
							label: '← 每行归一化后（和为 1）'
						},
						// 第三段：P_h 同时充当这次矩阵乘的左操作数（P_h 只画一次）
						then: {
							op: '× V_h',
							b: {
								name: 'V_h',
								shape: [S, dh],
								data: trace.v[0],
								realShape: [realSeq, R.v_head_dim],
								handoffId: 'v',
								label: `← 第 {{step:attn-v}} 步算出的 V`
							},
							result: {
								name: 'O_h',
								shape: [S, dh],
								data: trace.headOut[0],
								realShape: [realSeq, R.v_head_dim],
								label: '← 本头输出'
							}
						}
					}
				]
			},

			// ── 6. 拼接 → 输出投影（一条链，Concat 只画一次） ──
			{
				id: 'attn-out',
				kind: 'CONCAT',
				label: `${H} 个头各自算出 ${dh} 维的输出，横向拼接成 ${D} 维，再把拼接结果过一次输出投影得到本层输出。两段按先后顺序算`,
				formula:
					'O=\\underbrace{\\mathrm{Concat}(O_1,\\dots,O_H)}_{H\\times d_h\\ \\rightarrow\\ d_{model}}\\,W_O',
				tensors: [
					{
						name: 'concat-anim',
						label: `拼接：每个头占 ${dh} 列，${H} 个头依次排开就是 ${D} 列`,
						kind: 'concat',
						shape: [S, D],
						parts: trace.headOut.map((h, i) => ({
							name: `head ${i} 的输出`,
							label: `← 第 {{step:attn-mask-softmax-av}} 步 head ${i} 算出的 O_h`,
							shape: [S, dh],
							data: h
						})),
						result: {
							name: 'Concat',
							label: `← ${H} 个头横向拼起来（下面这次矩阵乘的左操作数就是它）`,
							shape: [S, D],
							data: trace.concat
						},
						// 第二段：Concat 直接当左操作数做输出投影
						then: {
							op: '× W_O',
							b: projWeight('W_O', Wo),
							result: {
								name: 'O',
								shape: [S, D],
								data: trace.out,
								realShape: [realSeq, R.d_model],
								handoffId: 'o',
								label: '↓ 这一层的输出，进入 FFN / MoE'
							}
						}
					}
				]
			}
		];
	}
};
