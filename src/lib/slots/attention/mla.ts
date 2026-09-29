/**
 * MLA（Multi-head Latent Attention）——DeepSeek-V2/V3/R1 的注意力。
 *
 * 与 MHA 的「唯一」结构性差别：K/V 不再"每个头各自投影"，而是
 * 「所有头共享一个低秩潜向量」 `c^KV`，再各自上投影还原。
 * 好处是推理时只需要缓存 `c^KV`（512 维）+ 一小段位置编码 `k^R`（64 维），
 * 每 token 576 个数，而 MHA 要缓存 2 × 128 × 128 = 32768 个——差 56.9 倍。
 *
 * 步骤划分（比 MHA 多一次压缩 / 还原）：
 *   1 Q 低秩压缩 + RMSNorm        c_Q = W_DQ·X,  c_Q~ = RMSNorm(c_Q)
 *   2 Q 升维                      q = W_UQ·c_Q~  （按头拆成 nope | rope）
 *   3 KV 低秩压缩 ★               [c_KV | k_pe] = W_a·X
 *   4 KV 升维                     [k_nope | v] = W_UKV·RMSNorm(c_KV)
 *   5 RoPE                       q_pe / k_pe 各自按位置旋转（并行）
 *   6 分数                        S_h = Q_h·K_hᵀ / √(d_h+d_h^R)
 *   7 掩码 → softmax → P·V       与 MHA 同款链式
 *   8 拼接 → 输出投影             W_O 是「降维」（真实 16384 → 7168）
 *   9 KV Cache ★                 缓存里到底躺着什么，与 MHA 比体积
 */
import {
	matmul,
	mergeHeads,
	randMat,
	rmsNorm,
	rope,
	softmaxRow,
	splitHeads,
	transpose,
	type Mat
} from '$lib/core/mat';
import type { DataflowEdge, DataflowNode, DataflowSpec, MatRef, SemanticsSpec } from '$lib/core/types';
import { REAL_R1 } from '$lib/model/config';

/**
 * 权重按「本仓库的约定」存成 `[in × out]`，直接 `matmul(x, W)`。
 * 注意这与论文 / `nn.Linear` 的 `[out × in]` 是转置关系——
 * 显示时矩阵的"真实形状"一栏写的也是转置后的（即 `[in × out]`）。
 */
export interface MlaWeights {
	/** W_DQ: [d_model × q_lora_rank] —— Q 的下投影 */
	Wdq: Mat;
	/** W_UQ: [q_lora_rank × n_h·qk_head_dim] —— Q 的上投影（含 RoPE 那一段） */
	Wuq: Mat;
	/** W_a: [d_model × (kv_lora_rank + qk_rope_head_dim)] —— KV 下投影 + 位置编码投影 */
	Wa: Mat;
	/** W_UKV: [kv_lora_rank × n_h·(qk_nope_head_dim + v_head_dim)] —— KV 上投影 */
	Wukv: Mat;
	/** W_O: [n_h·v_head_dim × d_model] —— 输出投影（真实是降维） */
	Wo: Mat;
}

export interface MlaTrace {
	out: Mat;
	x: Mat;

	// ── Q 的低秩路径 ──────────────────────────────
	cQ: Mat;
	cQNorm: Mat;
	/** [seq, n_h·qk_head_dim]，RoPE 已写回 */
	qFlat: Mat;
	/** [head][seq][qk_head_dim] */
	q: number[][][];
	qNope: number[][][];
	/** RoPE 之前 */
	qPeRaw: number[][][];
	/** RoPE 之后 */
	qPe: number[][][];

	// ── KV 的低秩路径 ─────────────────────────────
	/** 原始潜向量（未归一化） */
	cKv: Mat;
	/** RMSNorm 之后，喂给上投影 */
	cKvNorm: Mat;
	/** 位置编码段（未过 RoPE；sglang 里直接从 latent_cache 切出来） */
	kPeRaw: Mat;
	/** 已过 RoPE —— 这一份才是进缓存的 */
	kPe: Mat;
	/** [seq, n_h·(qk_nope_head_dim + v_head_dim)]，上投影的原始输出 */
	kvUpFlat: Mat;
	k: number[][][];
	v: number[][][];

	// ── 注意力 ────────────────────────────────────
	scores: number[][][];
	maskedScores: number[][][];
	probs: number[][][];
	headOut: number[][][];
	concat: Mat;
	mask: Mat;
	causal: boolean;
}

const NEG_INF = Number.NEGATIVE_INFINITY;



// ── MLA 的数据流图 ────────────────────────────────────────
/**
 * 这张图**挂在每一个 MLA 步骤的下方**，回答"Q/K/V 到底从哪来、到哪去"。
 *
 * 为什么非要单独画一张：MLA 的 Q 与 K/V 各自要走「下投影 → 归一化 → 上投影」三段，
 * 位置编码还要单独走一路（而且它进缓存、不归一化、不可压缩），
 * 只看单页的矩阵根本拼不出全貌。图上三条主线的颜色就是三种角色：
 * 靛蓝 = 低秩潜向量、琥珀 = 位置编码段、紫 = 按头升维后的 Q/K/V。
 *
 * 位置是**手写**的（`col` / `row`）——这张图我比自动布局更清楚线该怎么绕。
 */
export const DIAGRAM_NODES: DataflowNode[] = [
	{ id: 'x', label: 'X', sub: '[6 × 8] 输入', col: 0, row: 2, tone: 'input' },

	// 第一段：三路下投影（真实实现把前两路融合成一次矩阵乘，这里拆开画）
	{ id: 'cq', label: 'c_Q', sub: '[6 × 3] 低秩', col: 1, row: 0, tone: 'latent' },
	{ id: 'ckv', label: 'c_KV', sub: '[6 × 2] 所有头共享', col: 1, row: 2, tone: 'latent' },
	{ id: 'kpe', label: 'k_pe', sub: '[6 × 2] 未旋转', col: 1, row: 4, tone: 'rope' },

	// 第二段：归一化 / 旋转
	{ id: 'cqn', label: 'c_Q~', sub: 'RMSNorm', col: 2, row: 0, tone: 'latent' },
	{ id: 'ckvn', label: 'c_KV~', sub: 'RMSNorm', col: 2, row: 2, tone: 'latent' },
	{ id: 'kper', label: "k_pe′", sub: 'RoPE 后 → 进缓存', col: 2, row: 4, tone: 'rope' },

	// 第三段：上投影（一次矩阵乘算全部头，所以 Q 和 [k_nope|v] 各自是一个整块）
	{ id: 'q', label: 'q', sub: '[6 × 10] 每头 3 + 2', col: 3, row: 0, tone: 'head' },
	{ id: 'kv', label: '[k_nope | v]', sub: '[6 × 12] 每头 3 + 3', col: 3, row: 2, tone: 'head' },

	// 第四段：把上面两块**按列切开**（q 的后 2 列要过 RoPE，v 是后 3 列）
	{ id: 'qpe', label: 'q_pe', sub: '每头后 2 列', col: 4, row: 1, tone: 'rope' },
	{ id: 'knope', label: 'k_nope', sub: '每头前 3 列', col: 4, row: 2, tone: 'head' },
	{ id: 'v', label: 'v', sub: '每头后 3 列', col: 4, row: 3, tone: 'head' },

	// 第五段：拼成注意力真正吃的 Q / K / V（Q、K 是「nope 段 + rope 段」拼的条）
	{
		id: 'qh',
		label: 'Q_h',
		col: 5,
		row: 0,
		tone: 'head',
		split: [
			{ label: 'nope 3', ratio: 3, tone: 'head' },
			{ label: 'rope 2', ratio: 2, tone: 'rope' }
		]
	},
	{
		id: 'kh',
		label: 'K_h',
		col: 5,
		row: 2,
		tone: 'head',
		split: [
			{ label: 'nope 3', ratio: 3, tone: 'head' },
			{ label: 'rope 2', ratio: 2, tone: 'rope' }
		]
	},
	{ id: 'vh', label: 'V_h', sub: '只有 nope，3 维', col: 5, row: 3, tone: 'head' },

	// 缓存（MLA 的立身之本）与注意力
	{ id: 'cache', label: 'KV Cache', sub: "c_KV~ + k_pe′ = 4", col: 3, row: 5, tone: 'cache' },
	{ id: 'attn', label: '注意力', sub: 'S_h → P_h → O_h → u', col: 6, row: 1, tone: 'result' }
];

export const DIAGRAM_EDGES: DataflowEdge[] = [
	{ from: 'x', to: 'cq', label: 'W_DQ' },
	{ from: 'x', to: 'ckv', label: 'W_DKV' },
	{ from: 'x', to: 'kpe', label: 'W_KR' },

	{ from: 'cq', to: 'cqn', label: 'RMSNorm' },
	{ from: 'ckv', to: 'ckvn', label: 'RMSNorm' },
	{ from: 'kpe', to: 'kper', label: 'RoPE', tone: 'rope' },

	{ from: 'cqn', to: 'q', label: 'W_UQ' },
	{ from: 'ckvn', to: 'kv', label: 'W_UKV' },

	// 按列切开
	{ from: 'q', to: 'qh', label: '前 3 列' },
	{ from: 'q', to: 'qpe', label: '后 2 列', tone: 'rope' },
	{ from: 'kv', to: 'knope', label: '前 3 列' },
	{ from: 'kv', to: 'v', label: '后 3 列' },

	// 两段各自到位后拼起来 —— 这就是参考图里 q_t = [q^C ; q^R] / k_t = [k^C ; k^R]
	{ from: 'qpe', to: 'qh', label: 'RoPE', tone: 'rope' },
	{ from: 'knope', to: 'kh' },
	{ from: 'v', to: 'vh' },
	{ from: 'kper', to: 'kh', label: 'rope 段', tone: 'rope' },

	{ from: 'qh', to: 'attn' },
	{ from: 'kh', to: 'attn' },
	{ from: 'vh', to: 'attn' },

	// 进缓存的两块，以及 decode 时从缓存取
	{ from: 'ckvn', to: 'cache', routeOffset: 26, tone: 'latent' },
	{ from: 'kper', to: 'cache', routeOffset: 10, tone: 'rope' },
	{ from: 'cache', to: 'attn', dashed: true, enter: 'bottom', label: 'decode 时取缓存' }
];

/** 取这张图、把本步正在算的节点标出来（没说特殊说明的步骤用通用图例） */
function flow(active: string[], hint?: string): DataflowSpec {
	return { nodes: DIAGRAM_NODES, edges: DIAGRAM_EDGES, active, hint: hint ?? FLOW_HINT };
}

const FLOW_HINT =
	'实线 = 前向计算，虚线 = decode 时从缓存取。Q_h 与 K_h 都是「nope 段 + rope 段」拼起来的条：紫段来自潜向量升维、琥珀段来自位置编码（且只有它进缓存）；V_h 只有 nope 那一段。';

export const mla: SemanticsSpec<MlaTrace> = {
	id: 'mla',
	slot: 'attention',
	name: '多头潜在注意力 (MLA)',
	desc: 'K/V 不再每头各投影一次，而是共享一个低秩潜向量再上投影还原。位置编码只作用在一小段上、且必须单独缓存——换来 KV cache 缩小 56.9 倍。',
	formula:
		'c^{KV}=W^{DKV}h,\\quad k^{C}=W^{UK}c^{KV},\\quad v^{C}=W^{UV}c^{KV},\\quad k^{R}=\\mathrm{RoPE}(W^{KR}h)',
	defaultDepth: 'L2',
	maxDepth: 'L2',

	makeWeights(rnd, cfg) {
		const d = cfg.d_model;
		const H = cfg.num_heads;
		const qr = req(cfg.q_lora_rank, 'q_lora_rank');
		const kvr = req(cfg.kv_lora_rank, 'kv_lora_rank');
		const dn = req(cfg.qk_nope_head_dim, 'qk_nope_head_dim');
		const dr = req(cfg.qk_rope_head_dim, 'qk_rope_head_dim');
		const dv = req(cfg.v_head_dim, 'v_head_dim');

		return {
			Wdq: randMat(d, qr, rnd),
			Wuq: randMat(qr, H * (dn + dr), rnd),
			Wa: randMat(d, kvr + dr, rnd),
			Wukv: randMat(kvr, H * (dn + dv), rnd),
			Wo: randMat(H * dv, d, rnd)
		};
	},

	compute(x, ctx) {
		const cfg = ctx.cfg;
		const { num_heads: H, seq_len: S, causal } = cfg;
		const kvr = req(cfg.kv_lora_rank, 'kv_lora_rank');
		const dn = req(cfg.qk_nope_head_dim, 'qk_nope_head_dim');
		const dr = req(cfg.qk_rope_head_dim, 'qk_rope_head_dim');
		const dv = req(cfg.v_head_dim, 'v_head_dim');
		const { Wdq, Wuq, Wa, Wukv, Wo } = ctx.w as MlaWeights;

		// ── Q：下投影 → 归一化 → 上投影 ──────────────
		const cQ = matmul(x, Wdq);
		const cQNorm = rmsNorm(cQ);
		// 上投影一次算出 [n_h × qk_head_dim]；按头切开后，每头是 [nope | rope]
		const qRaw = matmul(cQNorm, Wuq);
		const qPre = splitHeads(qRaw, H, dn + dr);
		const qNope = qPre.map((h) => h.map((r) => r.slice(0, dn)));
		const qPeRaw = qPre.map((h) => h.map((r) => r.slice(dn)));

		// ── KV：下投影（潜向量 + 位置编码段）──────────
		const kvA = matmul(x, Wa);
		const cKv = kvA.map((r) => r.slice(0, kvr));
		const kPeRaw = kvA.map((r) => r.slice(kvr));
		// 归一化只作用在潜向量上，位置编码段不归一化（与 sglang 一致）
		const cKvNorm = rmsNorm(cKv);

		// ── RoPE：只作用在那一小段上，且「在写缓存之前」做 ──
		const kPe = rope(kPeRaw);
		const qPe = qPeRaw.map(rope);

		// ── KV 上投影：一次矩阵乘同时得到 k_nope 和 v ──
		const kvUpFlat = matmul(cKvNorm, Wukv);
		const kvHeads = splitHeads(kvUpFlat, H, dn + dv);
		const k = kvHeads.map((h) => h.map((r) => r.slice(0, dn)));
		const v = kvHeads.map((h) => h.map((r) => r.slice(dn)));

		// 把旋转后的 q_pe 写回 q（q = [q_nope | q_pe]），与 sglang 一致
		const q = qNope.map((h, i) => h.map((r, t) => [...r, ...qPe[i][t]]));
		const qFlat = Array.from({ length: S }, (_, t) => q.flatMap((h) => h[t]));

		// ── 注意力（与 MHA 完全同构，只是 K/V 的来源不同）──
		const scale = 1 / Math.sqrt(dn + dr);
		const scores: number[][][] = [];
		const maskedScores: number[][][] = [];
		const probs: number[][][] = [];
		const headOut: number[][][] = [];

		for (let h = 0; h < H; h++) {
			// K_h = [k_nope | k_pe]
			const kFull = k[h].map((r, t) => [...r, ...kPe[t]]);
			const raw = matmul(q[h], transpose(kFull)).map((row) => row.map((z) => z * scale));
			const masked = causal
				? raw.map((row, i) => row.map((val, j) => (j <= i ? val : NEG_INF)))
				: raw;
			const p = masked.map(softmaxRow);
			scores.push(raw);
			maskedScores.push(masked);
			probs.push(p);
			headOut.push(matmul(p, v[h]));
		}

		const concat = mergeHeads(headOut, dv);
		const mask = Array.from({ length: S }, (_, i) =>
			Array.from({ length: S }, (_, j) => (causal && j > i ? 0 : 1))
		);

		return {
			out: matmul(concat, Wo),
			x,
			cQ,
			cQNorm,
			qFlat,
			q,
			qNope,
			qPeRaw,
			qPe,
			cKv,
			cKvNorm,
			kPeRaw,
			kPe,
			kvUpFlat,
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
		const cfg = ctx.cfg;
		const { num_heads: H, seq_len: S, d_model: D } = cfg;
		const qr = req(cfg.q_lora_rank, 'q_lora_rank');
		const kvr = req(cfg.kv_lora_rank, 'kv_lora_rank');
		const dn = req(cfg.qk_nope_head_dim, 'qk_nope_head_dim');
		const dr = req(cfg.qk_rope_head_dim, 'qk_rope_head_dim');
		const dv = req(cfg.v_head_dim, 'v_head_dim');
		const dqk = dn + dr;
		const { Wdq, Wuq, Wa, Wukv, Wo } = ctx.w as MlaWeights;

		const R = REAL_R1;
		// 序列长度是我们选的示例长度（真实可到 16 万），所以只有特征维显示缩小倍数
		const realSeq = S;
		const realQk = R.num_heads * (R.head_dim + R.rope_head_dim);
		const realKv = R.num_heads * (R.head_dim + R.v_head_dim);

		const X_REF = {
			name: 'X',
			shape: [S, D],
			data: trace.x,
			realShape: [realSeq, R.d_model],
			handoffId: 'x'
		};
		const ref = (
			name: string,
			data: Mat,
			shape: number[],
			realShape: number[],
			tone?: MatRef['tone']
		) => ({ name, shape, data, realShape, tone });

		return [
			// ── 1. Q 低秩压缩 ──────────────────────────────
			{
				id: 'mla-q-compress',
				diagram: flow(['x', 'cq']),
				kind: 'PROJECT',
				label: `MLA 把 Q 也先压到低秩：${D} 维 → ${qr} 维（真实 ${R.d_model} → ${R.q_lora_rank}）。这是「下投影」，和后面 KV 的下投影是一回事——但 Q 不进 KV cache，所以这一步不影响缓存大小`,
				formula: 'c^Q=W^{DQ}X',
				tensors: [
					{
						name: 'mla-q-down',
						kind: 'matmul',
						shape: [S, qr],
						a: X_REF,
						b: ref('W_DQ', Wdq, [D, qr], [R.d_model, R.q_lora_rank], 'weight'),
						out: {
							name: 'c_Q',
							tone: 'latent',
							shape: [S, qr],
							data: trace.cQ,
							realShape: [realSeq, R.q_lora_rank],
							label: `↓ 第 {{step:mla-q-up}} 步先归一化、再升维`
						}
					},
					{
						name: 'mla-q-note',
						kind: 'shape',
						shape: [S, qr],
						note: `c_Q 是 ${qr} 维的低秩表示（真实 ${R.q_lora_rank} 维）：下一步先过 RMSNorm，再升维成 ${H} 头 × ${dqk} 维`
					}
				]
			},

			// ── 2. Q 升维（归一化 → 升维，一条链） ──────────
			{
				id: 'mla-q-up',
				diagram: flow(
					['cq', 'cqn', 'q', 'qpe'],
					'c_Q 在这一页只画一次：左边是它，中间是归一化后的 c_Q~，它同时充当这次矩阵乘的左操作数。'
				),
				kind: 'PROJECT',
				label: `潜向量先过 RMSNorm，再升维成 ${H} 头 × ${dqk} 维（真实每头 ${dn} 维不含位置编码 + ${dr} 维留给 RoPE）。「一次矩阵乘算出全部头」，切开后每头是 [nope | rope] 两段。归一化与升维按先后顺序算`,
				formula: '\\tilde c^Q=\\mathrm{RMSNorm}(c^Q),\\qquad q=W^{UQ}\\tilde c^Q',
				tensors: [
					{
						name: 'mla-q-up',
						kind: 'transform',
						shape: [S, qr],
						op: 'RMSNorm',
						input: {
							name: 'c_Q',
        tone: 'latent',
							shape: [S, qr],
							data: trace.cQ,
							realShape: [realSeq, R.q_lora_rank]
						},
						output: {
							name: 'c_Q~',
        tone: 'latent',
							shape: [S, qr],
							data: trace.cQNorm,
							realShape: [realSeq, R.q_lora_rank],
							label: '← 归一化后同时充当下面这次矩阵乘的左操作数'
						},
						// 第二段是矩阵乘：`c_Q~` 直接当左操作数，所以它只画一次
						then: {
							op: '× W_UQ',
							b: ref('W_UQ', Wuq, [qr, H * dqk], [R.q_lora_rank, realQk], 'weight'),
							result: {
								name: 'q',
         tone: 'head',
								shape: [S, H * dqk],
								data: trace.qFlat,
								realShape: [realSeq, realQk],
								label: `↓ 按头切开后，每头前 ${dn} 列是 nope、后 ${dr} 列是 rope`
							}
						}
					},
					{
						name: 'mla-q-split',
						kind: 'shape',
						shape: [S, H, dqk],
						note: `切开成 ${H} 头，每头 ${dqk} 维 = ${dn} 维 nope + ${dr} 维 rope（真实是 128 + 64）`
					}
				]
			},

			// ── 3. KV 低秩压缩（核心） ─────────────────────
			{
				id: 'mla-kv-compress',
				diagram: flow(
					['x', 'ckv', 'kpe'],
					'注意 c_KV 只有 2 维，却要供 2 个头共用——MLA 省缓存就省在这里。'
				),
				kind: 'PROJECT',
				label: `「MLA 的关键一步」：K 和 V 不再各自投影，而是先一起压成一个 ${kvr} 维的共享潜向量 c_KV（真实 ${R.kv_lora_rank} 维）；另外单独投影出 ${dr} 维位置编码段 k_pe。注意这里是 ${H} 个头「共用」同一份 c_KV——这正是缓存能变小的原因`,
				formula: '[\\,c^{KV}\\;|\\;k^{R}_{\\text{raw}}\\,]=W^{DKV+KR}X',
				tensors: [
					{
						name: 'mla-kv-down',
						kind: 'matmul',
						shape: [S, kvr + dr],
						a: X_REF,
						b: ref(
							'W_DKV+KR',
							Wa,
							[D, kvr + dr],
							[R.d_model, R.kv_lora_rank + R.rope_head_dim]
						),
						out: {
							name: '[c_KV | k_pe]',
							shape: [S, kvr + dr],
							data: trace.cKv.map((r, t) => [...r, ...trace.kPeRaw[t]]),
							realShape: [realSeq, R.kv_lora_rank + R.rope_head_dim],
							label: `↓ 前 ${kvr} 列是潜向量、后 ${dr} 列是位置编码（真实 ${R.kv_lora_rank} + ${R.rope_head_dim} = ${R.kv_lora_rank + R.rope_head_dim}）`
						}
					},
					{
						name: 'mla-kv-note',
						kind: 'shape',
						shape: [S, kvr],
						note: `c_KV 是 ${H} 个头「共享」的：MHA 要为每个头各存一份 K/V，MLA 只存这一份 ${kvr} 维（真实 ${R.kv_lora_rank} 维）`
					}
				]
			},

			// ── 4. KV 升维（归一化 → 升维，一条链） ──────────
			{
				id: 'mla-kv-up',
				diagram: flow(
					['ckv', 'ckvn', 'kv', 'knope', 'v'],
					'c_KV 在这一页只画一次：左边是它，中间是归一化后的 c_KV~，它同时充当这次矩阵乘的左操作数。'
				),
				kind: 'PROJECT',
				label: `潜向量先过 RMSNorm，再一次矩阵乘同时升维出「全部头的」 k_nope 和 v（真实 ${R.kv_lora_rank} → ${realKv}）。归一化只作用在潜向量上，位置编码那一段不归一化。归一化与升维按先后顺序算`,
				formula:
					'\\tilde c^{KV}=\\mathrm{RMSNorm}(c^{KV}),\\qquad [\\,k^{C}\\;|\\;v^{C}\\,]=W^{UKV}\\tilde c^{KV}',
				tensors: [
					{
						name: 'mla-kv-up',
						kind: 'transform',
						shape: [S, kvr],
						op: 'RMSNorm',
						input: {
							name: 'c_KV',
        tone: 'latent',
							shape: [S, kvr],
							data: trace.cKv,
							realShape: [realSeq, R.kv_lora_rank]
						},
						output: {
							name: 'c_KV~',
        tone: 'latent',
							shape: [S, kvr],
							data: trace.cKvNorm,
							realShape: [realSeq, R.kv_lora_rank],
							label: '← 只归一化潜向量（k_pe 那一段不归一化），同时充当下面这次矩阵乘的左操作数'
						},
						// 第二段是矩阵乘：`c_KV~` 直接当左操作数，所以它只画一次
						then: {
							op: '× W_UKV',
							b: ref('W_UKV', Wukv, [kvr, H * (dn + dv)], [R.kv_lora_rank, realKv], 'weight'),
							result: {
								name: '[k_nope | v]',
         tone: 'head',
								shape: [S, H * (dn + dv)],
								data: trace.kvUpFlat,
								realShape: [realSeq, realKv],
								label: `↓ 按头切开：每头前 ${dn} 列是 k_nope、后 ${dv} 列是 v`
							}
						}
					}
				]
			},

			// ── 5. RoPE ──────────────────────────────────
			{
				id: 'mla-rope',
				diagram: flow(
					['kpe', 'kper', 'qpe', 'qh'],
					'RoPE 只碰图上这两处：k_pe → k_pe′，以及 q 切出来的 q_pe 旋转后拼回 Q_h。'
				),
				kind: 'ROPE',
				label: `RoPE 只作用在那 ${dr} 维上（真实 ${R.rope_head_dim} 维）：按位置把每一对数字旋转一个角度。「q 和 k 各自独立旋转，可以同时算」。旋转是保范数的，只改方向不改长度`,
				formula:
					'\\mathcal{R}(x,pos)_{2i,2i+1}=\\mathrm{Rot}(\\theta_i\\,pos)\\begin{bmatrix}x_{2i}\\\\ x_{2i+1}\\end{bmatrix},\\quad \\theta_i=\\text{theta}^{-2i/d^{R}}',
				tensors: [
					{
						name: 'mla-rope-q',
						label: 'q 的 rope 段（head 0）',
						kind: 'transform',
						shape: [S, dr],
						row: 1,
						parallel: true,
						op: 'RoPE',
						input: {
							name: 'q_pe',
        tone: 'rope',
							shape: [S, dr],
							data: trace.qPeRaw[0],
							realShape: [realSeq, R.rope_head_dim]
						},
						output: {
							name: "q_pe'",
							shape: [S, dr],
							data: trace.qPe[0],
							realShape: [realSeq, R.rope_head_dim],
							label: '← 第 0 行不转，越靠后的位置转得越多'
						}
					},
					{
						name: 'mla-rope-k',
						label: 'k 的 rope 段',
						kind: 'transform',
						shape: [S, dr],
						row: 1,
						parallel: true,
						op: 'RoPE',
						input: {
							name: 'k_pe',
        tone: 'rope',
							shape: [S, dr],
							data: trace.kPeRaw,
							realShape: [realSeq, R.rope_head_dim]
						},
						output: {
							name: "k_pe'",
							shape: [S, dr],
							data: trace.kPe,
							realShape: [realSeq, R.rope_head_dim],
							label: '↓ 旋转后的这一份才会进 KV cache'
						}
					}
				]
			},

			// ── 6. 分数 ──────────────────────────────────
			{
				id: 'mla-score',
				diagram: flow(
					['qh', 'kh', 'vh', 'attn'],
					'到这里 Q/K/V 都齐了：Q_h 与 K_h 都是 [nope | rope]，V_h 只有 nope 那 3 维。'
				),
				kind: 'MATMUL',
				label: `每个头算 Q_h·K_hᵀ 再除以 √${dqk} 缩放。这里的 Q_h 和 K_h 都是「nope | rope」拼起来的：${dn} 维来自潜向量升维，${dr} 维来自旋转后的位置编码`,
				formula: 'S_h=\\frac{Q_hK_h^{\\top}}{\\sqrt{d_h+d_h^{R}}}',
				tensors: [
					{
						name: 'mla-score-matmul',
						kind: 'matmul',
						shape: [S, S],
						a: {
							name: 'Q_h',
       split: [
       	{ label: 'nope', ratio: 3, tone: 'head' },
       	{ label: 'rope', ratio: 2, tone: 'rope' }
       ],
        tone: 'head',
							shape: [S, dqk],
							data: trace.q[0],
							realShape: [realSeq, R.head_dim + R.rope_head_dim],
							label: `← ${dn} 维 nope + ${dr} 维 rope`
						},
						b: {
							name: 'K_hᵀ',
       split: [
       	{ label: 'nope', ratio: 3, tone: 'head' },
       	{ label: 'rope', ratio: 2, tone: 'rope' }
       ],
        tone: 'head',
							shape: [dqk, S],
							data: transpose(trace.k[0].map((r, t) => [...r, ...trace.kPe[t]])),
							realShape: [R.head_dim + R.rope_head_dim, realSeq],
							label: '← 同样是 nope + rope 两段'
						},
						out: {
							name: 'S_h',
							tone: 'result',
							shape: [S, S],
							data: trace.scores[0],
							realShape: [realSeq, realSeq]
						},
						scaleNote: `1/√${dqk} = ${(1 / Math.sqrt(dqk)).toFixed(2)}`
					},
					{
						name: 'mla-head-scores',
						label: `${H} 个头各算各的（上面演示 head 0）`,
						kind: 'tiles',
						shape: [H, S, S],
						data: trace.scores,
						tileLabels: Array.from({ length: H }, (_, h) => `head ${h} [${S} × ${S}]`)
					}
				]
			},

			// ── 7. 掩码 → softmax → P·V（与 MHA 同款链式） ──
			{
				id: 'mla-mask-softmax-av',
				diagram: flow(['attn'], '这一步之后就没 Q/K/V 什么事了，剩下的全是常规注意力运算。'),
				kind: 'MASK',
				label: `掩码 → softmax → 用权重对 V 加权求和。和 MHA 完全一样——MLA 变的只是 K/V 怎么来的，注意力本身没变`,
				formula:
					'S_{ij}=-\\infty\\ (j>i)\\ \\Rightarrow\\ P_{h,ij}=\\frac{e^{S_{ij}}}{\\sum_{j}e^{S_{ij}}}\\ \\Rightarrow\\ O_h=P_hV_h',
				tensors: [
					{
						name: 'mla-mask-softmax-av',
						kind: 'transform',
						shape: [S, S],
						op: 'softmax',
						preMask: trace.causal ? trace.mask : undefined,
						input: {
							name: 'S_h',
							shape: [S, S],
							data: trace.scores[0],
							realShape: [realSeq, realSeq],
							label: trace.causal
								? `← 第 {{step:mla-score}} 步算出的分数，先掩码：j > i → −∞`
								: `← 第 {{step:mla-score}} 步算出的分数`
						},
						output: {
							name: 'P_h',
        tone: 'result',
							shape: [S, S],
							data: trace.probs[0],
							realShape: [realSeq, realSeq],
							label: '← 每行归一化后（和为 1）'
						},
						then: {
							op: '× V_h',
							b: {
								name: 'V_h',
         tone: 'head',
								shape: [S, dv],
								data: trace.v[0],
								realShape: [realSeq, R.v_head_dim],
								label: `← 第 {{step:mla-kv-up}} 步从潜向量升维出来的 V`
							},
							result: {
								name: 'O_h',
         tone: 'result',
								shape: [S, dv],
								data: trace.headOut[0],
								realShape: [realSeq, R.v_head_dim],
								label: '← 本头输出'
							}
						}
					}
				]
			},

			// ── 8. 拼接 → 输出投影 ────────────────────────
			{
				id: 'mla-out',
				diagram: flow(['attn']),
				kind: 'CONCAT',
				label: `${H} 个头各 ${dv} 维拼成 ${H * dv} 维，再过输出投影回到 ${D} 维。注意真实这里是 「${realKv} → ${R.d_model} 的降维」，不是方阵`,
				formula: 'u=W^{O}\\,\\mathrm{Concat}(O_1,\\dots,O_H)',
				tensors: [
					{
						name: 'mla-concat',
						label: `拼接：每个头占 ${dv} 列`,
						kind: 'concat',
						shape: [S, H * dv],
						parts: trace.headOut.map((h, i) => ({
							name: `head ${i} 的输出`,
							label: `← 第 {{step:mla-mask-softmax-av}} 步 head ${i} 算出的 O_h`,
							shape: [S, dv],
							data: h
						})),
						result: {
							name: 'Concat',
        tone: 'result',
							shape: [S, H * dv],
							data: trace.concat,
							realShape: [realSeq, realKv],
							label: '← 下面这次矩阵乘的左操作数就是它'
						},
						then: {
							op: '× W_O',
							b: ref('W_O', Wo, [H * dv, D], [realKv, R.d_model], 'weight'),
							result: {
								name: 'u',
         tone: 'result',
								shape: [S, D],
								data: trace.out,
								realShape: [realSeq, R.d_model],
								handoffId: 'o',
								label: '↓ 这一层的输出，进入 FFN / MoE'
							}
						}
					}
				]
			},

			// ── 9. KV Cache（MLA 的立身之本） ──────────────
			{
				id: 'mla-kv-cache',
				diagram: flow(
					['ckvn', 'kper', 'cache'],
					'缓存里只有两块：归一化后的 c_KV 与旋转后的 k_pe′。K/V 都能从 c_KV 现算，所以不用存。'
				),
				kind: 'STORE',
				label: `回头看缓存：MLA 每个 token 只需要存 「c_KV（${kvr} 个数）+ 旋转后的 k_pe（${dr} 个数）」，因为 K/V 都能从 c_KV 现算出来；而 MHA 得把每头的 K、V 都存下来。位置编码那一段「不能压缩」——旋转和低秩投影不可交换，所以只能单独存`,
				formula:
					'\\text{cache}=\\underbrace{c^{KV}}_{d_c}+\\underbrace{k^{R}}_{d_h^{R}}\\quad\\text{vs}\\quad\\text{MHA}=2\\,n_h d_h',
				tensors: [
					{
						name: 'mla-kv-cache',
						kind: 'kvcache',
						shape: [S, kvr + dr],
						blocks: [
							{
								name: 'c_KV',
         tone: 'latent',
								label: `所有头共享的潜向量（真实 ${R.kv_lora_rank} 维）`,
								size: kvr,
								realSize: R.kv_lora_rank,
								data: trace.cKvNorm
							},
							{
								name: "k_pe'",
								label: `旋转后的位置编码（真实 ${R.rope_head_dim} 维，不可压缩）`,
								size: dr,
								realSize: R.rope_head_dim,
								data: trace.kPe
							}
						],
						perToken: kvr + dr,
						realPerToken: R.kv_lora_rank + R.rope_head_dim,
						compare: {
							label: 'MHA 要缓存 K 和 V',
							parts: [
								{ name: 'K', perToken: H * cfg.head_dim, sub: `每个头各存一份（${H} 头 × ${cfg.head_dim} 维）` },
								{ name: 'V', perToken: H * cfg.head_dim, sub: `每个头各存一份（${H} 头 × ${cfg.head_dim} 维）` }
							],
							realPerToken: 2 * R.num_heads * R.head_dim
						},
						note: `工程上还能更进一步（absorb）：把 W_UKV 折进 Q 的投影，注意力直接在潜空间里算——「缓存里的 c_KV 既当 K 又当 V」，连 K/V 都不用物化。这也是 decode 阶段那个 ${realKv}×${R.kv_lora_rank} 的大矩阵可以完全不参与的原因`
					}
				]
			}
		];
	}
};

/ 取一个必须有值的 MLA 维度——缺了就是配置没配全，早报错好过画出错图 */
function req(v: number | undefined, name: string): number {
	if (v === undefined) {
		throw new Error(`[mla] 配置缺少 ${name}`);
	}
	return v;
}
