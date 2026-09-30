/**
 * DeepSeek V4 的**混合注意力**——`swa` / `csa` / `hca` 三个插件共用的核心。
 *
 * 三种层型只差在两件事上：
 *   - **有没有压缩分支**（`ratio > 0`）：纯滑窗层只看最近 `swa_window` 个 token；
 *     CSA / HCA 额外维护"每组一条"的压缩 KV。
 *   - **压缩条目怎么参与**：CSA 用 indexer 挑 top-k（稀疏检索），HCA 全部参与（稠密）。
 *
 * 除了这两点，三种层**完全同构**，所以 Q/KV/注意力/输出/缓存这几段写在同一个地方。
 * 这也是 V4 设计上真正的意思：混合发生在**一次注意力**里，而不是"两种注意力各算一遍
 * 再合并"——滑窗 KV 和压缩 KV 进的是**同一个 softmax**。
 *
 * 三个 V4 独有的细节都在这里：
 *   - **K = V 共用**：只有一个 KV 头（MQA），而且键和值就是同一个向量，
 *     所以注意力输出是"权重乘 K 再求和"，没有单独的 V 投影。
 *   - **输出做逆 RoPE**：K 带着自己的绝对位置旋转 `R(j)`，输出 `Σ_j P_j·R(j)v_j`
 *     再按查询位置做一次 `R(−i)`，得到的是 `R(j−i)`——**位置信息以相对量的形式留下来**。
 *   - **分组低秩输出投影**：`W_O^A` 把每组的 `n_h/G × head_dim` 降到 `o_lora_rank`，
 *     拼起来再由 `W_O^B` 回到 `d_model`。真实是 8 组、每组 4096 → 1024。
 *
 * 还有一个容易被当成装饰、其实会改数值的偏置：**`attn_sink`**（每头一个可学习的标量）。
 * 它给每个 query 多加一个"谁都不看"的虚拟槽位，值为 0、只有分数参与 softmax——
 * 于是"这一行该不该把注意力摊平"也变成可学的。分数矩阵因此比 token 数**多一列**。
 */
import {
	matmul,
	mergeHeads,
	randMat,
	rmsNorm,
	ropeApply,
	softmaxRow,
	splitHeads,
	transpose,
	type Mat,
	type Vec
} from '$lib/core/mat';
import type { MatrixBand, ModelConfigLike, Step } from '$lib/core/types';
import { REAL_V4_FLASH } from '$lib/model/config';
import {
	compress,
	makeCompressorWeights,
	makeIndexerWeights,
	ref,
	ropeInverse,
	runIndexer,
	type CompressorPlan,
	type CompressorTrace,
	type CompressorWeights,
	type IndexerTrace,
	type IndexerWeights
} from './compressor';
import { v4Diagram, type V4Variant } from './v4-diagram';

const NEG_INF = Number.NEGATIVE_INFINITY;
const R = REAL_V4_FLASH;

function dot(a: Vec, b: Vec): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[i];
	return s;
}

/**
 * 对每行**末尾 `dr` 维**做 RoPE。
 *
 * 频率基准是 `dr`（rope 段自己的长度），不是整头维度——`ropeApply` 是按
 * "传给它的这条向量的长度"算 `theta^{-2i/d}` 的，所以必须把尾部切出来再转，
 * 不能拿整头去转。逆 RoPE 也要用同一份基准，否则正反两次转的不是同一个角度。
 */
function ropeTail(rows: Mat, dr: number, theta: number): Mat {
	if (dr <= 0) return rows;
	return rows.map((row, i) => [
		...row.slice(0, row.length - dr),
		...ropeApply(row.slice(row.length - dr), i, theta)
	]);
}

/** 这一层怎么混合 */
export interface HybridPlan {
	/** 压缩比；`0` = 纯滑窗层（没有压缩分支） */
	ratio: number;
	/** 有压缩分支时：是否用重叠窗口（CSA 用） */
	overlap: boolean;
	/** 有压缩分支时：是否带 indexer（CSA 有，HCA 没有） */
	indexer: boolean;
}

/** 这一层属于哪种层型（决定数据流图上哪条支路淡显） */
export function variantOf(plan: HybridPlan): V4Variant {
	return plan.indexer ? 'csa' : plan.ratio > 0 ? 'hca' : 'swa';
}

/** 这一层的实际维度（都从 `cfg` 取，插件不硬编码） */
export interface HybridDims {
	H: number;
	hd: number;
	dn: number;
	dr: number;
	G: number;
	oLora: number;
	W: number;
	theta: number;
	qr: number;
	perGroup: number;
}

export function hybridDims(cfg: ModelConfigLike, plan: HybridPlan): HybridDims {
	const H = cfg.num_heads;
	const hd = cfg.head_dim;
	const G = cfg.o_groups ?? 1;
	return {
		H,
		hd,
		dn: cfg.qk_nope_head_dim ?? hd,
		dr: cfg.qk_rope_head_dim ?? 0,
		G,
		oLora: cfg.o_lora_rank ?? Math.floor((H * hd) / G),
		W: cfg.swa_window ?? cfg.seq_len,
		// 纯滑窗层用主 RoPE，压缩层整层换成压缩版 YaRN RoPE（真实 base 40000）
		theta:
			plan.ratio > 0
				? (cfg.compress_rope_theta ?? cfg.rope_theta ?? 10000)
				: (cfg.rope_theta ?? 10000),
		qr: cfg.q_lora_rank ?? cfg.d_model,
		perGroup: H / G
	};
}

/** 注意力那套压缩 KV 的计划（头维 = `head_dim`） */
export function compressPlan(cfg: ModelConfigLike, plan: HybridPlan): CompressorPlan | null {
	if (plan.ratio <= 0) return null;
	return {
		ratio: plan.ratio,
		overlap: plan.overlap,
		headDim: cfg.head_dim,
		ropeTheta: cfg.compress_rope_theta ?? cfg.rope_theta ?? 10000
	};
}

/** indexer 那套压缩 KV 的计划（头维 = `index_head_dim`，更窄） */
export function indexerPlan(cfg: ModelConfigLike, plan: HybridPlan): CompressorPlan | null {
	if (!plan.indexer) return null;
	return {
		ratio: plan.ratio,
		overlap: plan.overlap,
		headDim: cfg.index_head_dim ?? cfg.head_dim,
		ropeTheta: cfg.compress_rope_theta ?? cfg.rope_theta ?? 10000
	};
}

export interface HybridWeights {
	/** `W_Q^A`: [d_model × q_lora_rank] —— Q 的下投影 */
	Wqa: Mat;
	/** `W_Q^B`: [q_lora_rank × n_h·head_dim] —— Q 的上投影 */
	Wqb: Mat;
	/** `W_KV`: [d_model × head_dim] —— 只有一个 KV 头，且 K = V 共用 */
	Wkv: Mat;
	/** `W_O^A`: 每组一个 [n_h/G·head_dim × o_lora_rank] */
	Woa: Mat[];
	/** `W_O^B`: [G·o_lora_rank × d_model] */
	Wob: Mat;
	/** 每头一个「谁都不看」的空槽偏置 */
	sink: Vec;
	comp: CompressorWeights | null;
	idx: IndexerWeights | null;
}

export function makeHybridWeights(
	rnd: () => number,
	cfg: ModelConfigLike,
	plan: HybridPlan
): HybridWeights {
	const { H, hd, G, oLora, qr, perGroup } = hybridDims(cfg, plan);
	const cp = compressPlan(cfg, plan);
	const ip = indexerPlan(cfg, plan);
	return {
		Wqa: randMat(cfg.d_model, qr, rnd),
		Wqb: randMat(qr, H * hd, rnd),
		Wkv: randMat(cfg.d_model, hd, rnd),
		Woa: Array.from({ length: G }, () => randMat(perGroup * hd, oLora, rnd)),
		Wob: randMat(G * oLora, cfg.d_model, rnd),
		sink: randMat(H, 1, rnd).map((r) => r[0]),
		comp: cp ? makeCompressorWeights(rnd, cfg, cp) : null,
		idx: ip ? makeIndexerWeights(rnd, cfg, ip) : null
	};
}

export interface HybridTrace {
	out: Mat;
	x: Mat;

	// ── Q ─────────────────────────────────────────
	cQ: Mat;
	cQNorm: Mat;
	/** [S, n_h·head_dim] 上投影的原始输出（还没按头归一化） */
	qUp: Mat;
	/** [S, n_h·head_dim] 每头过完 RMSNorm（还没 RoPE） */
	qFlat: Mat;
	/** [head][seq][head_dim] RoPE 之后 */
	q: number[][][];
	/** [head][seq][dr] 每头的位置编码段：RoPE 之前 / 之后（`q` 就是由它们拼出来的） */
	qPeRaw: number[][][];
	qPe: number[][][];

	// ── KV（单头，K = V） ─────────────────────────
	kvRaw: Mat;
	kvNorm: Mat;
	kv: Mat;

	// ── 压缩 / 检索 ───────────────────────────────
	compress: CompressorTrace | null;
	indexer: IndexerTrace | null;
	/** 压缩 KV（= 压缩分支的 `entriesReady`），K = V 共用 */
	compKv: Mat | null;

	// ── 融合注意力 ────────────────────────────────
	/** 每头一个可学习的 **sink 偏置**——只进 softmax 的分母，没有位置 */
	sink: Vec;
	/**
	 * 融合注意力的 **K 矩阵**（token 列取 `kv`、压缩条目列取压缩 KV），
	 * `[S + nE × head_dim]`——它就是分数那个矩阵乘的右操作数。
	 * sink **不在**里面：它不是点积，是直接加在分数上的一项。
	 */
	kAll: Mat;
	/** `Q·Kᵀ/√d` 的结果，`[S × (S + nE)]`（不含 sink 那一列） */
	scoreReal: Mat;
	/**
	 * `P·V` 的右操作数：`[S + nE + 1 × head_dim]`。
	 *
	 * 前 `S + nE` 行**就是 `kAll`**（K = V 共用），最后一行是 sink 的 V —— **全 0**，
	 * 所以 sink 只把 softmax 的分母撑大，一点输出都不贡献。
	 */
	vAll: Mat;
	/** 列空间宽度：token 列 + 压缩条目列 + 1 个 sink 列 */
	cols: number;
	nE: number;
	sinkCol: number;
	scores: number[][][];
	mask: Mat;
	maskedScores: number[][][];
	probs: number[][][];
	headOut: number[][][];
	/** 逆 RoPE 之后 */
	headOutInv: number[][][];
	/** 每组自己的 [S, n_h/G·head_dim] */
	groups: Mat[];
	/** 每组低秩投影后的 [S, o_lora_rank]，横向拼起来就是 `oLatent` */
	groupLatent: Mat[];
	oLatent: Mat;
}

export function hybridCompute(
	x: Mat,
	w: HybridWeights,
	plan: HybridPlan,
	cfg: ModelConfigLike
): HybridTrace {
	const { H, hd, dn, dr, G, oLora, W, theta, perGroup } = hybridDims(cfg, plan);
	const S = x.length;

	// ── Q：下投影 → 归一化 → 上投影 → 每头归一化 → RoPE ──
	const cQ = matmul(x, w.Wqa);
	const cQNorm = rmsNorm(cQ);
	const qUp = matmul(cQNorm, w.Wqb);
	const qHeads = splitHeads(qUp, H, hd).map((h) => rmsNorm(h));
	const qNope = qHeads.map((h) => h.map((r) => r.slice(0, dn)));
	const qPeRaw = qHeads.map((h) => h.map((r) => r.slice(dn)));
	const qPe = qPeRaw.map((h) => h.map((r, i) => ropeApply(r, i, theta)));
	const q = qNope.map((h, hi) => h.map((r, i) => [...r, ...qPe[hi][i]]));

	// ── KV：一次投影 + 归一化 + RoPE。K 和 V 就是这一份 ──
	const kvRaw = matmul(x, w.Wkv);
	const kvNorm = rmsNorm(kvRaw);
	const kv = ropeTail(kvNorm, dr, theta);

	// ── 压缩分支（CSA / HCA）与检索（CSA） ──
	const cp = compressPlan(cfg, plan);
	const ip = indexerPlan(cfg, plan);
	const compTrace = cp && w.comp ? compress(x, w.comp, cp) : null;
	const compKv = compTrace ? compTrace.entriesReady : null;
	const idxTrace = ip && w.idx ? runIndexer(x, cQNorm, w.idx, ip, cfg) : null;

	// ── 一次注意力：滑窗 token + 压缩条目 + sink ──
	const nE = compKv ? compKv.length : 0;
	const cols = S + nE + 1;
	const sinkCol = S + nE;

	const tokVisible = (i: number, j: number) => j <= i && i - j < W;
	const picked = (i: number, e: number) =>
		!!compTrace && compTrace.groups[e].bornAt <= i && (!idxTrace || idxTrace.topk[i].includes(e));

	/** 分数矩阵乘的两侧：K 是所有真实列拼起来的（sink 不参与点积） */
	const kAll: Mat = Array.from({ length: S + nE }, (_, j) => (j < S ? kv[j] : compKv![j - S]));

	const mask: Mat = Array.from({ length: S }, (_, i) => [
		...Array.from({ length: S }, (_, j) => (tokVisible(i, j) ? 1 : 0)),
		...Array.from({ length: nE }, (_, e) => (picked(i, e) ? 1 : 0)),
		1
	]);

	const scale = Math.pow(hd, -0.5);
	const scores: number[][][] = [];
	const maskedScores: number[][][] = [];
	const probs: number[][][] = [];
	const headOut: number[][][] = [];

	for (let h = 0; h < H; h++) {
		// 每一列都是"第 i 个 token 对第 j 个位置的关注分数"：
		// 前 S 列是窗口里的 token、接着 nE 列是压缩条目、最后一列是 sink。
		const raw = Array.from({ length: S }, (_, i) => [
			...Array.from({ length: S }, (_, j) => scale * dot(q[h][i], kv[j])),
			...Array.from({ length: nE }, (_, e) => scale * dot(q[h][i], compKv![e])),
			w.sink[h]
		]);
		const masked = raw.map((row, i) => row.map((v, j) => (mask[i][j] ? v : NEG_INF)));
		const p = masked.map(softmaxRow);
		// V：token 列取 `kv`、压缩条目列取压缩 KV、sink 列是 0（只进分母，不贡献输出）
		const out = p.map((row) =>
			Array.from({ length: hd }, (_, c) =>
				row.reduce(
					(acc, pv, j) => acc + pv * (j < S ? kv[j][c] : j === sinkCol ? 0 : compKv![j - S][c]),
					0
				)
			)
		);
		scores.push(raw);
		maskedScores.push(masked);
		probs.push(p);
		headOut.push(out);
	}

	// ── 逆 RoPE：把 K 的绝对旋转按查询位置转回去，只留相对量 ──
	const headOutInv = headOut.map((heads) =>
		heads.map((r, i) => [...r.slice(0, dn), ...ropeInverse(r.slice(dn), i, theta)])
	);

	// ── 分组低秩输出投影 ──
	const groups = Array.from({ length: G }, (_, g) =>
		Array.from({ length: S }, (_, i) =>
			Array.from(
				{ length: perGroup * hd },
				(_, c) => headOutInv[g * perGroup + Math.floor(c / hd)][i][c % hd]
			)
		)
	);
	const groupLatent = groups.map((gr, g) => matmul(gr, w.Woa[g]));
	// 注意是按**行**拼：每个 token 的 G 段低秩结果接在一起
	const oLatent: Mat = Array.from({ length: S }, (_, i) => groupLatent.flatMap((gl) => gl[i]));

	return {
		out: matmul(oLatent, w.Wob),
		x,
		cQ,
		cQNorm,
		qUp,
		qFlat: mergeHeads(qHeads, hd),
		qPeRaw,
		qPe,
		q,
		kvRaw,
		kvNorm,
		kv,
		compress: compTrace,
		indexer: idxTrace,
		compKv,
		kAll,
		scoreReal: scores[0].map((row) => row.slice(0, S + nE)),
		// V 的最后一行是 sink 的 V = 0：它只进分母
		vAll: [...kAll, new Array<number>(hd).fill(0)],
		sink: w.sink,
		cols,
		nE,
		sinkCol,
		scores,
		mask,
		maskedScores,
		probs,
		headOut,
		headOutInv,
		groups,
		groupLatent,
		oLatent
	};
}

// ══════════════════════════════════════════════════════════
// 步骤：前段（Q / KV / RoPE）与后段（注意力 / 输出 / 缓存）
// ══════════════════════════════════════════════════════════
/** 前段：Q 与 KV 两条投影 + 归一化 + RoPE */
export function hybridHeadSteps(
	trace: HybridTrace,
	w: HybridWeights,
	cfg: ModelConfigLike,
	plan: HybridPlan,
	prefix: string
): Step[] {
	const { H, hd, dn, dr, qr } = hybridDims(cfg, plan);
	const S = cfg.seq_len;
	const D = cfg.d_model;
	const variant = variantOf(plan);
	const flow = (active: string[]) => v4Diagram(variant, active);
	const X = ref('X', trace.x, [S, D], [S, R.d_model], 'input');
	const ropeNote =
		plan.ratio > 0
			? `这一层用压缩版 RoPE（真实 base ${R.compress_rope_theta}），不是纯滑窗层的 10000`
			: `纯滑窗层用主 RoPE（base 10000）`;

	return [
		{
			id: `${prefix}-q`,
			kind: 'PROJECT',
			diagram: flow(['x', 'cq']),
			label: `Q 先降到低秩（${D} → ${qr} 维，真实 ${R.d_model} → ${R.q_lora_rank}）再升维到 ${H} 头 × ${hd} 维。和 MLA 一样是"下投影 → 上投影"，但 V4 只有 ${H} 个头、且**没有独立的 V 投影**`,
			formula: 'c^Q=W^{QA}x',
			tensors: [
				{
					name: `${prefix}-q-down`,
					kind: 'matmul',
					shape: [S, qr],
					a: X,
					b: ref('W_Q^A', w.Wqa, [D, qr], [R.d_model, R.q_lora_rank], 'weight'),
					out: ref(
						'c_Q',
						trace.cQ,
						[S, qr],
						[S, R.q_lora_rank],
						'latent',
						`← 第 {{step:${prefix}-q-up}} 步先归一化、再升维`
					)
				},
				{
					name: `${prefix}-q-note`,
					kind: 'shape',
					wide: true,
					shape: [S, H, hd],
					note: `升维后按头切开：${H} 头 × ${hd} 维 = ${H * hd}（真实 ${R.num_heads} × ${R.head_dim} = ${R.num_heads * R.head_dim}，比 d_model 大 8 倍——Q 是**升维**的）`
				}
			]
		},
		{
			id: `${prefix}-kv`,
			kind: 'PROJECT',
			diagram: flow(['x', 'kvraw']),
			label: `KV 只有**一个头**（MQA）：所有 Q 头共用同一份 K，而且 K = V——键和值就是同一个 ${hd} 维向量。所以这里只投影一次，没有 V 的矩阵。真实是 ${R.d_model} → ${R.head_dim} 维`,
			formula: 'k=v=W^{KV}x',
			tensors: [
				{
					name: `${prefix}-kv-mm`,
					kind: 'matmul',
					shape: [S, hd],
					a: X,
					b: ref('W_KV', w.Wkv, [D, hd], [R.d_model, R.head_dim], 'weight'),
					out: ref(
						'k_raw',
						trace.kvRaw,
						[S, hd],
						[S, R.head_dim],
						'latent',
						`← 第 {{step:${prefix}-kv-norm-rope}} 步归一化 + RoPE`
					)
				},
				{
					name: `${prefix}-kv-note`,
					kind: 'shape',
					wide: true,
					shape: [S, hd],
					note: `K = V 是 V4 省显存的第二招：MLA 把 K/V 压成一个潜向量，V4 干脆让键和值**共用同一个向量**——缓存里每 token 只有 ${hd} 个数（真实 ${R.head_dim}）`
				}
			]
		},
		// ── Q 支路之一：潜向量归一化 → 升维到全部头 ──
		{
			id: `${prefix}-q-up`,
			kind: 'PROJECT',
			diagram: flow(['cq', 'cqn', 'q']),
			label: `Q 这一路的第一步：潜向量先过 RMSNorm，再一次矩阵乘升维到 ${H} 头 × ${hd} 维（真实 ${R.num_heads} × ${R.head_dim}）。**按头切开之后每个头还要各自归一化、各自转 RoPE**——那两步在下一页`,
			formula: 'q=W^{QB}\\mathrm{RMSNorm}(c^Q)',
			tensors: [
				{
					name: `${prefix}-q-norm-up`,
					kind: 'transform',
					shape: [S, H * hd],
					op: 'RMSNorm',
					input: ref('c_Q', trace.cQ, [S, qr], [S, R.q_lora_rank], 'latent'),
					output: ref('c_Q~', trace.cQNorm, [S, qr], [S, R.q_lora_rank]),
					then: {
						op: '× W_Q^B',
						b: ref(
							'W_Q^B',
							w.Wqb,
							[qr, H * hd],
							[R.q_lora_rank, R.num_heads * R.head_dim],
							'weight'
						),
						result: ref(
							'q',
							trace.qUp,
							[S, H * hd],
							[S, R.num_heads * R.head_dim],
							'head',
							`← 一次矩阵乘算出全部头；按头切开每头 ${hd} 维（真实 ${R.head_dim}），还没归一化、也还没转`
						)
					}
				},
				{
					name: `${prefix}-q-up-note`,
					kind: 'shape',
					wide: true,
					shape: [S, H, hd],
					note: `升维后按头切开：${H} 头 × ${hd} 维 = ${H * hd}（真实 ${R.num_heads} × ${R.head_dim} = ${R.num_heads * R.head_dim}，比 d_model 大 8 倍——Q 是**升维**的）。**每个头做的事完全一样**，下一页只演示 head 0`
				}
			]
		},

		// ── Q 支路之二：每头归一化 → RoPE（一条链，rope 那几维算到才框出来） ──
		{
			id: `${prefix}-q-norm-rope`,
			kind: 'ROPE',
			diagram: flow(['q']),
			label: `**按头**处理，一个头一个头地做：先各自过一次 RMSNorm，再对**末尾 ${dr} 维**（真实 ${R.rope_head_dim} 维）做 RoPE——前面 ${dn} 维（真实 ${R.head_dim - R.rope_head_dim} 维）是 nope，原样不动。**每个头都做同样的事**，这里只演示 head 0（上一页那 ${H * hd} 列里的前 ${hd} 列）；下面整块矩阵画的是 head 0，**算到 RoPE 那一步才把末尾 ${dr} 维框出来**。${ropeNote}`,
			formula: 'q=\\mathcal{R}\\!\\left(\\mathrm{RMSNorm}(q)\\right)\\ \\text{（逐头）}',
			tensors: [
				{
					name: `${prefix}-q-head-norm-rope`,
					kind: 'transform',
					shape: [S, hd],
					cellSize: 32,
					op: 'RMSNorm（每头）',
					input: ref(
						'q_0',
						trace.qUp.map((r) => r.slice(0, hd)),
						[S, hd],
						[S, R.head_dim],
						'head',
						'← head 0（上一页那 8 列的前 4 列）'
					),
					output: ref(
						'q_0~',
						trace.qFlat.map((r) => r.slice(0, hd)),
						[S, hd],
						[S, R.head_dim],
						'head',
						'← 整头都归一化了'
					),
					// RoPE 挂在链尾：整块矩阵只画一次，"末尾 dr 维"在这一段才被框出来
					tail: {
						op: 'RoPE',
						result: ref(
							'q_0′',
							trace.q[0],
							[S, hd],
							[S, R.head_dim],
							'head',
							`← 框出的末尾 ${dr} 维转完；前面 ${dn} 维原样`
						),
						highlightCols: Array.from({ length: dr }, (_, i) => dn + i)
					}
				},
				{
					name: `${prefix}-q-head-note`,
					kind: 'shape',
					wide: true,
					shape: [S, H, hd],
					note: `**真实每头 ${R.head_dim} 维 = ${R.head_dim - R.rope_head_dim} 维 nope + ${R.rope_head_dim} 维 rope**：RMSNorm 是整头做的（${hd} 维 = 真实 ${R.head_dim}），RoPE 只作用在**末尾**那 ${R.rope_head_dim} 维（这里缩成 ${dr} 维，列缩小 32×）——前 ${R.head_dim - R.rope_head_dim} 维是 nope，转完原样拼回去。两步都是**逐行**做的：同一个 token 的 ${H} 个头各自归一化、各自旋转，互不影响；旋转保范数，只改方向不改长度`
				}
			]
		},

		// ── KV 支路：归一化 → RoPE（只有一个头，K = V） ──
		{
			id: `${prefix}-kv-norm-rope`,
			kind: 'ROPE',
			diagram: flow(['kvraw', 'kvnorm', 'kv']),
			label: `KV 这一路简单得多：**只有一个头**（所有 Q 头共用），所以不用升维、也不用按头归一化——一次 RMSNorm 之后直接对末尾 ${dr} 维做 RoPE 就完事，和上一页 Q 那一路是**同一个格式**（整块矩阵 + 算到 RoPE 才框出末尾 ${dr} 维）。K = V 共用，所以这一份同时当键和值。${ropeNote}`,
			formula: 'k=v=\\mathcal{R}\\!\\left(\\mathrm{RMSNorm}(W^{KV}x)\\right)',
			tensors: [
				{
					name: `${prefix}-kv-norm-rope`,
					kind: 'transform',
					shape: [S, hd],
					// 同 q-norm-rope：c128 4-5-6-7 步放大（24 → 32）
					cellSize: 32,
					op: 'RMSNorm',
					input: ref('k_raw', trace.kvRaw, [S, hd], [S, R.head_dim], 'latent'),
					output: ref('k~', trace.kvNorm, [S, hd], [S, R.head_dim]),
					tail: {
						op: 'RoPE',
						result: ref(
							'k = v',
							trace.kv,
							[S, hd],
							[S, R.head_dim],
							'cache',
							`← 只有框出的末尾 ${dr} 维被旋转；这一份同时当键和值`
						),
						highlightCols: Array.from({ length: dr }, (_, i) => dn + i)
					}
				},
				{
					name: `${prefix}-kv-note`,
					kind: 'shape',
					wide: true,
					shape: [S, hd],
					note: `KV 只有一个头、宽度 ${hd}（真实 ${R.head_dim} = nope ${R.head_dim - R.rope_head_dim} + rope ${R.rope_head_dim}）——所以缓存里每 token 只有 ${hd} 个数；而 Q 是 ${H} 头 × ${hd} 维。RoPE 同样只转末尾 ${dr} 维（真实 ${R.rope_head_dim} 维，列缩小 32×）`
				}
			]
		}
	];
}

/** 后段：一次融合注意力 → 逆 RoPE + 分组输出投影 → 缓存 */
export function hybridTailSteps(
	trace: HybridTrace,
	w: HybridWeights,
	cfg: ModelConfigLike,
	plan: HybridPlan,
	prefix: string
): Step[] {
	const { H, hd, dn, dr, G, oLora, W, perGroup } = hybridDims(cfg, plan);
	const S = cfg.seq_len;
	const D = cfg.d_model;
	const nE = trace.nE;
	const cols = trace.cols;
	const variant = variantOf(plan);
	const flow = (active: string[]) => v4Diagram(variant, active);

	const colLayout = [
		`前 ${S} 列 = 每个 token 的精确 KV（**蓝框**；每行只留滑窗内那 ${W} 个）`,
		nE ? `接着 ${nE} 列 = 压缩条目（**琥珀框**）` : null,
		`最后 1 列 = sink 空槽`
	]
		.filter(Boolean)
		.join('、');

	/**
	 * K / V 上的**分组框**（`MatRef.bands`）：把"滑窗里的精确 token"和"压缩条目"分成两段框出来。
	 *
	 * 两个颜色在 K 和 V 上**一致**：蓝 = 局部精确（滑窗）、琥珀 = 远处压缩（c128）——
	 * 换一块矩阵也认得出同一类东西。纯滑窗层没有压缩段（`nE = 0`），不画框。
	 */
	const swaBand = (axis: MatrixBand['axis']): MatrixBand => ({ axis, from: 0, to: S - 1, tone: '#38bdf8' });
	const compBand = (axis: MatrixBand['axis']): MatrixBand => ({
		axis,
		from: S,
		to: S + nE - 1,
		tone: '#fbbf24'
	});
	/** K 的列（`Kᵀ` 的列 = 位置） */
	const kvColBands: MatrixBand[] | undefined = nE ? [swaBand('col'), compBand('col')] : undefined;
	/** V 的行（行 = 位置）；末行 sink 不框 */
	const kvRowBands: MatrixBand[] | undefined = nE ? [swaBand('row'), compBand('row')] : undefined;

	const cacheBlocks: {
		name: string;
		label?: string;
		size: number;
		realSize: number;
		data?: Mat;
	}[] = [
		{
			name: '滑窗 KV',
			label: `最近 ${W} 个 token 的精确 KV（有上限，不随序列变长）`,
			size: hd,
			realSize: R.head_dim,
			data: trace.kv.slice(-W)
		}
	];
	if (trace.compKv) {
		cacheBlocks.push({
			name: '压缩 KV',
			label: `每 ${plan.ratio} 个 token 压出 1 条，共 ${nE} 条`,
			size: hd,
			realSize: R.head_dim,
			data: trace.compKv
		});
	}

	// 压缩条目摊到每个 token 是多少个数（真实是每 128 个 token 1 条）
	const amortized = trace.compKv ? Math.max(1, Math.round((hd * nE) / S)) : 0;

	return [
		// ── 分数：S = Q·Kᵀ/√d（真正的矩阵乘，逐格写数字） ──
		{
			id: `${prefix}-score`,
			kind: 'MATMUL',
			diagram: flow(['q', 'kv', 'ckv', 'itop']),
			label: nE
				? `每个头算 Q_h·K_hᵀ 再除以 √${hd} 缩放，得到分数矩阵 S。K 是**所有真实列**拼起来的：前 ${S} 列是每个 token 的**精确 KV**（**蓝框**；每行只有滑窗内那 ${W} 个是活的，其余会被掩码）、后 ${nE} 列是**压缩条目**（**琥珀框**）——所以这一步就把"局部精确 + 远处压缩"放进同一个矩阵了。sink 偏置**不在这里**（它不是点积，是直接加在分数上的一项，见下）`
				: `每个头算 Q_h·K_hᵀ 再除以 √${hd} 缩放，得到分数矩阵 S。K 就是滑窗里那 ${S} 个 token 的精确 KV（单头，K = V）。sink 偏置**不在这里**（它不是点积，是直接加在分数上的一项，见下）`,
			formula: 'S_h=\\frac{Q_hK_h^{\\top}}{\\sqrt{d_h}}',
			tensors: [
				{
					name: `${prefix}-score-mm`,
					kind: 'matmul',
					shape: [S, S + nE],
					a: ref(
						'Q_h',
						trace.q[0],
						[S, hd],
						[S, R.head_dim],
						'head',
						`← 第 {{step:${prefix}-q-norm-rope}} 步算出的 q（已过 RoPE），这里演示 head 0`
					),
					b: ref(
						'K_hᵀ',
						transpose(trace.kAll),
						[hd, S + nE],
						[R.head_dim, S + nE],
						'head',
						nE
							? `← 转置之后：前 ${S} 列是每个 token 的精确 KV（**蓝框**）、后 ${nE} 列是压缩条目（**琥珀框**）`
							: '← 转置之后，每一列是一个 token 的 K',
						kvColBands
					),
					out: ref(
						'S_h',
						trace.scoreReal,
						[S, S + nE],
						[S, S + nE],
						'result',
						`← 第 i 行第 j 列 = 第 i 个 token 对第 j 个位置的关注分数（还没有 sink）`
					),
					scaleNote: `1/√${hd} = ${(1 / Math.sqrt(hd)).toFixed(2)}`
				},
				{
					name: `${prefix}-score-note`,
					kind: 'shape',
					wide: true, // 这句话比一格宽，挤在格子里会折行、还会压到隔壁
					shape: [S, S + nE],
					note: `${H} 个头各算各的（上面演示 head 0）。K = V 共用，所以这个 K 一会儿也当 V 用；sink 偏置不在这个矩阵乘里`
				},
				{
					name: `${prefix}-score-note2`,
					kind: 'shape',
					wide: true,
					shape: [H],
					note: `sink 偏置**不在**这个矩阵乘里——它不是点积，下一步会作为**右边一列**直接接在分数上（见第 {{step:${prefix}-attn}} 步）`
				}
			]
		},

		// ── 掩码 → softmax → P·V → 逆 RoPE（一条链走到底） ──
		{
			id: `${prefix}-attn`,
			kind: 'MASK',
			diagram: flow(['score', 'sink', 'attn', 'inv']),
			label:
				`**一次** softmax 吃掉两类 KV，然后**一条链走到底**：\`S ──softmax──▶ P_h ──×V──▶ O_h ──逆 RoPE──▶ O_h′\`。` +
				`掩码规则：j > i（因果）、超出滑窗、没被 indexer 选中——三种都置 −∞` +
				(nE ? `，但被拒的压缩条目**分数其实还在**（淡显的那些格子）` : '') +
				`。上一页的分数右边**再接上 sink 那一列**（${colLayout}）——滑窗和压缩历史进的是同一个 softmax，` +
				`不是"各算一遍再合并"；sink 只进分母、没有位置（见下面那块小字）。` +
				`V 的前 ${S + nE} 行**就是上一页那个 K**（K = V 共用），最后一行是 sink 的 V = 0。` +
				`链尾那段**逆 RoPE**：每头输出里只有末尾 ${dr} 维（真实 ${R.rope_head_dim} 维）带着 K 的绝对旋转 R(j)，` +
				`按查询位置再转一次 R(−i) 就只剩相对量 R(j−i)——就是框出来的那几列；其余 ${H - 1} 个头一样处理`,
			formula: nE
				? 'S_{ij}=-\\infty\\ (\\text{masked})\\ \\Rightarrow\\ P=\\mathrm{softmax}(S)\\ \\Rightarrow\\ O=P\\,V\\ \\Rightarrow\\ O\\mathcal{R}(-i)'
				: 'S_{i,\\text{sink}}=a_h,\\qquad P=\\mathrm{softmax}(S)\\ \\Rightarrow\\ O=P\\,V\\ \\Rightarrow\\ O\\mathcal{R}(-i)',
			tensors: [
				{
					name: `${prefix}-attn-softmax-av`,
					kind: 'transform',
					shape: [S, cols],
					op: 'softmax',
					preMask: trace.mask,
					input: ref(
						'S（+ sink 列）',
						trace.scores[0],
						[S, cols],
						[S, cols],
						'result',
						'← 上一页的分数 + sink 那一列'
					),
					output: ref(
						'P_h',
						trace.probs[0],
						[S, cols],
						[S, cols],
						'result',
						'← 每行归一化（和为 1）'
					),
					then: {
						op: '× V（K = V）',
						b: ref(
							'V',
							trace.vAll,
							[cols, hd],
							[cols, R.head_dim],
							'cache',
							`← 前 ${S + nE} 行 = K（**蓝框** = 滑窗的精确 token、**琥珀框** = 压缩条目）；末行 V = 0`,
							kvRowBands
						),
						result: ref(
							'O_h',
							trace.headOut[0],
							[S, hd],
							[S, R.head_dim],
							'result',
							'← 演示 head 0'
						)
					},
					// 逆 RoPE 接在**链尾**：`O_h′` 就贴在 `O_h` 右边。
					// 不这么写就得另起一行、把 `O_h` 再画一遍当输入——一页里同一个矩阵出现两次。
					tail: {
						op: '逆 RoPE',
						result: ref(
							"O_h′",
							trace.headOutInv[0],
							[S, hd],
							[S, R.head_dim],
							'result',
							'← 只剩相对量 R(j−i)'
						),
						// 框出 rope 那几维：输入输出框**同一批列**（位置不变，只有值变）
						highlightCols: Array.from({ length: dr }, (_, i) => dn + i)
					}
				},
				{
					name: `${prefix}-attn-sink`,
					kind: 'row',
					wide: true, // 下面那段小字很长，要整行
					shape: [H],
					data: trace.sink,
					labels: Array.from({ length: H }, (_, h) => `head ${h} 的 sink 偏置`),
					label: `右边多出来的那一列就是 **sink 偏置**（每头一个可学习的标量）——它**不是点积**，所以不在上一页那个矩阵乘里。kernel 里是在 online-softmax 循环**之后**一次性折进分母的（l = l·α + exp(sink − m)），分子累加的 acc 里没有它（V = 0）。所以它**没有位置**：画在最后一列纯粹是排版，挪到第一列结果一模一样；作用是让这一头能把"用不掉的概率"漏出去，而不是硬摊给可见的 token。这和 StreamingLLM 那个"拿第一个 token 当 sink"是**两回事**（那里是一个真实的 KV 位置被学得特别重要）`
				}
			]
		},
		{
			id: `${prefix}-out-a`,
			kind: 'PROJECT',
			diagram: flow(['out']),
			label: `逆 RoPE 已经在上一页的**链尾**做完了（${H} 个头各自的 ${hd} 维都成了 O_h′），这一页只剩**分组低秩投影**：${G} 组各自把 ${perGroup * hd} 维降到 ${oLora} 维（真实 ${R.o_groups} 组、每组 ${R.d_model} → ${R.o_lora_rank}）`,
			formula: '\\tilde o^{(g)}=\\tilde o^{(g)}W_O^{A(g)}',
			tensors: [
				// ${G} 组**并排一行**（`row` 相同 + `group` 不同 → `.hrow`）：
				// 它们是同一个 token 的两段，左右放着才好和"分组"对上；上下堆着会被误读成先后两步
				...Array.from({ length: G }, (_, g) => ({
					name: `${prefix}-out-a-${g}`,
					kind: 'matmul' as const,
					shape: [S, oLora],
					row: 1,
					group: `g${g}`,
					parallel: true,
					a: ref(
						`o^{(${g})}`,
						trace.groups[g],
						[S, perGroup * hd],
						[S, (R.num_heads / R.o_groups) * R.head_dim],
						'head'
					),
					b: ref(
						`W_O^A[${g}]`,
						w.Woa[g],
						[perGroup * hd, oLora],
						[(R.num_heads / R.o_groups) * R.head_dim, R.o_lora_rank],
						'weight'
					),
					out: ref(
						`õ^{(${g})}`,
						trace.groupLatent[g],
						[S, oLora],
						[S, R.o_lora_rank],
						'latent'
					)
				}))
			]
		},
		{
			id: `${prefix}-out-b`,
			kind: 'CONCAT',
			diagram: flow(['inv', 'out']),
			label: `${G} 组各自的低秩结果拼起来，再过 W_O^B 回到 ${D} 维（真实 ${R.o_groups} × ${R.o_lora_rank} = ${R.o_groups * R.o_lora_rank} → ${R.d_model}）。每组只在自己那段里做低秩，参数比一个 ${H * hd}×${D} 的大矩阵少得多`,
			formula:
				'u=\\mathrm{Concat}\\!\\left(\\tilde o^{(1)},\\dots,\\tilde o^{(G)}\\right)W_O^B',
			tensors: [
				{
					name: `${prefix}-concat`,
					kind: 'concat',
					shape: [S, G * oLora],
					parts: Array.from({ length: G }, (_, g) => ({
						name: `第 ${g} 组的低秩结果`,
						label: `← 第 {{step:${prefix}-out-a}} 步第 ${g} 组的输出`,
						shape: [S, oLora],
						realShape: [S, R.o_lora_rank],
						data: trace.groupLatent[g]
					})),
					result: ref(
						'õ',
						trace.oLatent,
						[S, G * oLora],
						[S, R.o_groups * R.o_lora_rank],
						'latent',
						'← 下面这次矩阵乘的左操作数就是它'
					),
					then: {
						op: '× W_O^B',
						b: ref(
							'W_O^B',
							w.Wob,
							[G * oLora, D],
							[R.o_groups * R.o_lora_rank, R.d_model],
							'weight'
						),
						result: ref(
							'u',
							trace.out,
							[S, D],
							[S, R.d_model],
							'result',
							'↓ 这一层的输出，进入 FFN / MoE'
						)
					}
				}
			]
		},
		{
			id: `${prefix}-cache`,
			kind: 'STORE',
			diagram: flow(['kv', 'ckv', 'cache']),
			label: nE
				? `回头看缓存：滑窗那部分只留最近 ${W} 个 token 的精确 KV（**有上限**，序列再长也不涨）；再往前的历史只留 ${nE} 条压缩 KV——每 ${plan.ratio} 个 token 才新增 1 条。真实模型里这是"每 128 个 token 加 1 条"`
				: `回头看缓存：滑窗只留最近 ${W} 个 token 的精确 KV（真实 ${R.swa_window} 个）——**有上限**，序列再长也不涨。代价是更早的历史彻底看不见了；CSA / HCA 层正是为了把那段历史也捡回来`,
			formula: '\\text{cache}=O(W)+O(S/\\text{ratio})',
			tensors: [
				{
					name: `${prefix}-cache-view`,
					kind: 'kvcache',
					shape: [S, hd],
					mineTitle: nE ? 'V4 混合注意力：滑窗 KV + 压缩 KV' : 'V4 纯滑窗层：只留最近 W 个 token',
					mineLabel: 'V4',
					blocks: cacheBlocks,
					perToken: hd + amortized,
					realPerToken: R.head_dim + (nE ? Math.round(R.head_dim / 128) : 0),
					compare: {
						label: 'MHA 要缓存 K 和 V',
						parts: [
							{ name: 'K', perToken: H * hd, sub: `每个头各存一份（${H} 头 × ${hd} 维）` },
							{ name: 'V', perToken: H * hd, sub: `每个头各存一份（${H} 头 × ${hd} 维）` }
						],
						realPerToken: 2 * R.num_heads * R.head_dim
					},
					note: nE
						? `压缩条目是"每 ${plan.ratio} 个 token 一条"，摊到每个 token 只有 ${amortized} 个数——所以缓存是 O(W) + O(S/${plan.ratio})，而不是 MHA 的 O(S)。代价是压缩条目里没有精确的位置信息，只能给出"这一段历史大概说了什么"`
						: `滑窗缓存是 O(W) 的环形缓冲，不随序列变长——这是纯滑窗层能省显存的原因，代价是更早的历史完全看不到`
				}
			]
		}
	];
}
