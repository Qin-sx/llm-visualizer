/**
 * DeepSeek V4 的 mHC（Manifold-Constrained Hyper-Connections）残差过程。
 *
 * 对照 sglang：
 *   - `srt/models/deepseek_v4.py` 的 `hc_pre` / `hc_post` / `hc_head_torch`
 *   - `kernels/ops/layernorm/mhc.py` 的 `hc_split_sinkhorn` / `hc_combine`
 *
 * 一段"读-写"（每个子层各做一次）：
 *   读（pre）：4 条流拼成一行 → `hc_fn` 投影 → RMS 缩放 → 三块门控（pre/post/comb）→
 *             `Σ_k pre_k·r_k` 混合成 1 条 → RMSNorm → 子层；
 *   写（post）：`r′_k = post_k·y + Σ_j comb_kj·r_j`（comb 经 Sinkhorn 归一化成双随机矩阵）。
 * 模型末尾 `hc_head`：同样的门控加权平均把 4 条压回 1 条（无 Sinkhorn）。
 */
import { matAdd, rmsNorm, type Mat, type Vec } from '$lib/core/mat';

/** 一个子层的混合参数（`hc_fn` 管门控 logits 的投影） */
export interface HcMixParams {
	/** `[(2+hc)·hc, hc·D]`（真实 `[24, 4×4096]`）——门控 logits 的投影 */
	fn: Mat;
	/** `[(2+hc)·hc]`——pre/post/comb 三块门控的偏置 */
	base: Vec;
	/** `[3]`——pre 用 [0]、post 用 [1]、comb 用 [2] 的标量 */
	scale: Vec;
}

/** 模型末尾压回用的参数 */
export interface HcHeadParams {
	/** `[hc, hc·D]`（真实 `[4, 4×4096]`） */
	fn: Mat;
	/** `[hc]` */
	base: Vec;
	/** `[1]`（标量） */
	scale: Vec;
}

export interface HcLayerWeights {
	attn: HcMixParams;
	ffn: HcMixParams;
}

export interface HcWeights {
	layers: HcLayerWeights[];
	head: HcHeadParams;
}

/** 一次 pre-mix（"读"）的全部中间量 */
export interface HcMixTrace {
	/** `[(2+hc)·hc, hc·D]`——门控投影的权重（gates 页的矩阵乘要画它） */
	fn: Mat;
	/** `[S, (2+hc)·hc]`——门控 logits（投影 × RMS 缩放） */
	mixes: Mat;
	/** `[S, hc]`——读门（σ + eps，不归一化） */
	pre: Mat;
	/** `[S, hc]`——写门（2σ ∈ (0,2)） */
	post: Mat;
	/** `[S, hc·hc]`——Sinkhorn 双随机矩阵（按行摊平，逐 token 一个 hc×hc） */
	comb: Mat;
	/** `[S, D]`——`Σ_k pre_k·r_k`（未归一化） */
	combined: Mat;
	/** `[S, D]`——`rmsNorm(combined)`，子层的输入 */
	normed: Mat;
}

/** 一次写回（"写"）的全部中间量 */
export interface HcWriteTrace {
	/** 每条流一份 `[S, D]`：`post[:,k]·y`（子层输出按写门分发） */
	postY: Mat[];
	/** 每条流一份 `[S, D]`：`Σ_j comb[:,k,j]·r_j`（旧流按 comb 重分配） */
	combOld: Mat[];
	/** 每条流一份 `[S, D]`：`postY[k] + combOld[k]`——写回后的新流 */
	newStreams: Mat[];
}

function dot(a: Vec, b: Vec): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[i];
	return s;
}

function sigmoid(x: number): number {
	return 1 / (1 + Math.exp(-x));
}

/**
 * Sinkhorn 归一化：先做数值稳定的行 softmax + 一次列归一，再交替行/列归一 `iters - 1` 次，
 * 收敛到双随机矩阵（行和 ≈ 列和 ≈ 1）。与 `mhc.py:_hc_split_sinkhorn_torch` 逐行一致。
 */
function sinkhorn(m: Mat, iters: number, eps: number): Mat {
	const n = m.length;
	// 初始行 softmax（数值稳定）+ eps
	let x = m.map((row) => {
		const mx = Math.max(...row);
		const e = row.map((v) => Math.exp(v - mx));
		const s = e.reduce((a, b) => a + b, 0);
		return e.map((v) => v / s + eps);
	});
	const colNorm = () => {
		const colSum: Vec = Array.from({ length: n }, (_, k) =>
			x!.reduce((a, row) => a + row[k], 0)
		);
		x = x!.map((row) => row.map((v, k) => v / (colSum[k] + eps)));
	};
	colNorm();
	for (let it = 1; it < iters; it++) {
		x = x.map((row) => {
			const s = row.reduce((a, b) => a + b, 0);
			return row.map((v) => v / (s + eps));
		});
		colNorm();
	}
	return x;
}

/**
 * 一次 pre-mix（"读"）。`streams` = hc 条 `[S, D]` 流。对照 `hc_pre`（deepseek_v4.py:2762）。
 */
export function hcMix(
	streams: Mat[],
	fn: Mat,
	base: Vec,
	scale: Vec,
	hc: number,
	iters: number,
	eps: number
): HcMixTrace {
	const S = streams[0].length;
	const D = streams[0][0].length;
	const mix = (2 + hc) * hc;
	// 4 条流**首尾拼成一行**（真实是 [S, hc·hidden] 的 2D 张量）
	const xFlat: Mat = Array.from({ length: S }, (_, i) => streams.flatMap((s) => s[i]));
	const rsqrt = xFlat.map((row) => 1 / Math.sqrt(row.reduce((a, v) => a + v * v, 0) / row.length + eps));
	const mixes: Mat = xFlat.map((row, i) =>
		Array.from({ length: mix }, (_, j) => dot(row, fn[j]) * rsqrt[i])
	);
	// 三块门控：pre（前 hc 列，σ+eps）、post（中 hc 列，2σ）、comb（后 hc² 列，Sinkhorn）
	const pre = mixes.map((row) => row.slice(0, hc).map((v, j) => sigmoid(v * scale[0] + base[j]) + eps));
	const post = mixes.map((row) => row.slice(hc, 2 * hc).map((v, j) => 2 * sigmoid(v * scale[1] + base[hc + j])));
	const comb3 = mixes.map((row) => {
		const m: Mat = Array.from({ length: hc }, (_, j) =>
			Array.from({ length: hc }, (_, k) => row[j * hc + k + 2 * hc] * scale[2] + base[j * hc + k + 2 * hc])
		);
		return sinkhorn(m, iters, eps);
	});
	const comb: Mat = comb3.map((rows) => rows.flat());
	// 混合成 1 条：h[i,d] = Σ_k pre[i,k]·streams[k][i,d]
	const combined: Mat = Array.from({ length: S }, (_, i) =>
		Array.from({ length: D }, (_, d) => streams.reduce((a, s, k) => a + pre[i][k] * s[i][d], 0))
	);
	return {
		fn,
		mixes,
		pre,
		post,
		comb,
		combined,
		normed: rmsNorm(combined)
	};
}

/**
 * 一次写回（"写"）。`y` = 子层输出 `[S, D]`，`old` = 旧 hc 条流。对照 `hc_post_torch_impl`
 * （deepseek_v4.py:2998）：`r′_k = post_k·y + Σ_j comb_kj·r_j`。
 */
export function hcWrite(
	y: Mat,
	old: Mat[],
	post: Mat,
	comb: Mat,
	hc: number
): HcWriteTrace {
	const S = y.length;
	const D = y[0].length;
	const postY: Mat[] = [];
	const combOld: Mat[] = [];
	const newStreams: Mat[] = [];
	for (let k = 0; k < hc; k++) {
		// post[:,k]·y：每行的第 k 个写门 × 子层输出
		const py = Array.from({ length: S }, (_, i) => y[i].map((v) => post[i][k] * v));
		// comb[:,k,:]·old：旧流按第 k 行 comb 加权
		const co = Array.from({ length: S }, (_, i) =>
			Array.from({ length: D }, (_, d) =>
				Array.from({ length: hc }, (_, j) => comb[i][k * hc + j]).reduce(
					(a, c, j) => a + c * old[j][i][d],
					0
				)
			)
		);
		postY.push(py);
		combOld.push(co);
		newStreams.push(matAdd(py, co));
	}
	return { postY, combOld, newStreams };
}

/**
 * 模型末尾把 hc 条流压回 1 条。对照 `hc_head_torch`（deepseek_v4.py:582）：
 * 同样"拼一行 → 投影 → RMS 缩放 → σ+eps 门控 → 加权平均"，但没有 Sinkhorn、没有写回。
 */
export function hcHead(
	streams: Mat[],
	fn: Mat,
	base: Vec,
	scale: Vec,
	hc: number,
	eps: number
): { pre: Mat; combined: Mat; out: Mat } {
	const S = streams[0].length;
	const D = streams[0][0].length;
	const xFlat: Mat = Array.from({ length: S }, (_, i) => streams.flatMap((s) => s[i]));
	const rsqrt = xFlat.map((row) => 1 / Math.sqrt(row.reduce((a, v) => a + v * v, 0) / row.length + eps));
	const mixes: Mat = xFlat.map((row, i) =>
		Array.from({ length: hc }, (_, j) => dot(row, fn[j]) * rsqrt[i])
	);
	const pre = mixes.map((row) => row.map((v, j) => sigmoid(v * scale[0] + base[j]) + eps));
	const combined: Mat = Array.from({ length: S }, (_, i) =>
		Array.from({ length: D }, (_, d) => streams.reduce((a, s, k) => a + pre[i][k] * s[i][d], 0))
	);
	return { pre, combined, out: combined };
}

/** 造一份随机混合参数（形状按 `hc_mult` 与 `d_model` 缩放，真实形状标在界面上） */
export function makeHcMixParams(
	rnd: () => number,
	hc: number,
	D: number
): HcMixParams {
	const mix = (2 + hc) * hc;
	return {
		fn: Array.from({ length: mix }, () => Array.from({ length: hc * D }, () => rnd() * 2 - 1)),
		base: Array.from({ length: mix }, () => rnd() * 2 - 1),
		scale: [1, 1, 1].map((s) => s * (rnd() + 0.5))
	};
}

/** 造一份随机 hc_head 参数 */
export function makeHcHeadParams(rnd: () => number, hc: number, D: number): HcHeadParams {
	return {
		fn: Array.from({ length: hc }, () => Array.from({ length: hc * D }, () => rnd() * 2 - 1)),
		base: Array.from({ length: hc }, () => rnd() * 2 - 1),
		scale: [rnd() + 0.5]
	};
}
