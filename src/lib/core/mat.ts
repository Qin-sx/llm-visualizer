/**
 * 极简矩阵与随机数工具。
 *
 * 模型很小（约 125 万参数），所以直接用朴素实现，不引入任何数值库。
 */
import type { Mat, Vec } from './types';

// 便于插件从一处拿到张量类型
export type { Mat, Vec };

// ── 确定性随机数 ──────────────────────────────────────────
/** mulberry32：小而稳定的 PRNG，保证同一 seed 得到同一份权重 */
export function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/**
 * 均匀分布随机矩阵，取值落在 [-scale, scale]。
 *
 * 默认 scale 用 Xavier 风格的 `√(3/rows)`——即令 var(W) = 1/rows，
 * 使 `x @ W` 前后**方差保持不变**。
 *
 * 这一点对本项目是必须的：若用固定 scale，每层线性变换都会把方差乘上一个
 * 远小于 1 的因子，几层之后信号衰减到接近 0，注意力分数全为 0、
 * softmax 退化成均匀分布，整个可视化就失去意义了。
 */
export function randMat(
	rows: number,
	cols: number,
	rnd: () => number,
	scale = Math.sqrt(3 / rows)
): Mat {
	return Array.from({ length: rows }, () =>
		Array.from({ length: cols }, () => (rnd() * 2 - 1) * scale)
	);
}

// ── 矩阵运算 ──────────────────────────────────────────────
export function zeros(rows: number, cols: number): Mat {
	return Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
}

/** a[m,k] · b[k,n] → [m,n] */
export function matmul(a: Mat, b: Mat): Mat {
	const m = a.length;
	const k = b.length;
	const n = b[0].length;
	const out = zeros(m, n);
	for (let i = 0; i < m; i++) {
		const ai = a[i];
		for (let p = 0; p < k; p++) {
			const v = ai[p];
			if (v === 0) continue;
			const bp = b[p];
			for (let j = 0; j < n; j++) out[i][j] += v * bp[j];
		}
	}
	return out;
}

export function transpose(a: Mat): Mat {
	const rows = a.length;
	const cols = a[0].length;
	const out = zeros(cols, rows);
	for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) out[j][i] = a[i][j];
	return out;
}

export function matAdd(a: Mat, b: Mat): Mat {
	return a.map((row, i) => row.map((v, j) => v + b[i][j]));
}

export function matScale(a: Mat, s: number): Mat {
	return a.map((row) => row.map((v) => v * s));
}

// ── 逐元素函数 ────────────────────────────────────────────
export function silu(x: number): number {
	return x / (1 + Math.exp(-x));
}

/**
 * 逐行 RMSNorm。
 *
 * 真实 Transformer（含 DeepSeek R1）在每个子层前都会做归一化。
 * 少了它，激活尺度不可控——注意力分数会全部趋近于 0。
 */
export function rmsNorm(x: Mat, eps = 1e-6): Mat {
	return x.map((row) => {
		let s = 0;
		for (const v of row) s += v * v;
		const rms = Math.sqrt(s / row.length) || 1;
		const d = Math.max(rms, eps);
		return row.map((v) => v / d);
	});
}

export function softmaxRow(v: Vec): Vec {
	let m = -Infinity;
	for (const x of v) if (x > m) m = x;
	const e = v.map((x) => Math.exp(x - m));
	let s = 0;
	for (const x of e) s += x;
	return e.map((x) => x / s);
}

export function softmaxRows(m: Mat): Mat {
	return m.map(softmaxRow);
}

// ── 旋转位置编码 RoPE ─────────────────────────────────────
/**
 * 把位置 `pos` 的 RoPE 作用到一段向量上：**成对旋转**。
 *
 * 第 i 对 `(v[2i], v[2i+1])` 用一个角度 $\theta_i \cdot pos$ 旋转，
 * $\theta_i = \text{theta}^{-2i/d}$（`d` = 这段向量的长度）。于是
 * "第 0 个位置不旋转、越靠后的位置转得越多"，而**两两之间的相对角度差
 * 只与位置差有关**——这正是 RoPE 表达相对位置的方式。
 *
 * 注意 RoPE 是**保范数**的（旋转矩阵正交）：旋转前后向量的模不变。
 * MLA 的解耦 RoPE 之所以不能折进低秩投影，就是因为旋转与线性投影不可交换。
 */
export function ropeApply(v: Vec, pos: number, theta = 10000): Vec {
	const d = v.length;
	const out = v.slice();
	for (let p = 0; p + 1 < d; p += 2) {
		const ang = pos * Math.pow(theta, -p / d);
		const c = Math.cos(ang);
		const s = Math.sin(ang);
		out[p] = v[p] * c - v[p + 1] * s;
		out[p + 1] = v[p] * s + v[p + 1] * c;
	}
	return out;
}

/** 逐行做 RoPE：第 i 行 = 第 i 个位置 */
export function rope(m: Mat, theta = 10000): Mat {
	return m.map((row, i) => ropeApply(row, i, theta));
}

// ── 统计辅助（供可视化用） ────────────────────────────────
export function maxAbs(m: Mat | Vec): number {
	const flat = Array.isArray(m[0]) ? (m as Mat).flat() : (m as Vec);
	let mx = 0;
	for (const x of flat) {
		const a = Math.abs(x);
		if (a > mx) mx = a;
	}
	return mx || 1;
}

/** 取 top-k 的索引（按值降序） */
export function topK(v: Vec, k: number): number[] {
	return v
		.map((value, index) => ({ value, index }))
		.sort((a, b) => b.value - a.value)
		.slice(0, k)
		.map((d) => d.index);
}

// ── 按头切分 / 合并（注意力类算子通用） ───────────────────
/** [seq, d_model] → [head][seq, head_dim] */
export function splitHeads(m: Mat, heads: number, headDim: number): number[][][] {
	return Array.from({ length: heads }, (_, h) =>
		m.map((row) => row.slice(h * headDim, (h + 1) * headDim))
	);
}

/** [head][seq, head_dim] → [seq, d_model]，与 splitHeads 互逆 */
export function mergeHeads(perHead: number[][][], headDim: number): Mat {
	const heads = perHead.length;
	const seq = perHead[0].length;
	return Array.from({ length: seq }, (_, t) =>
		Array.from({ length: heads * headDim }, (_, i) => perHead[Math.floor(i / headDim)][t][i % headDim])
	);
}

/** 随机造一对上下投影权重（Dense FFN 与 MoE 专家共用） */
export function randFfnPair(
	rnd: () => number,
	dModel: number,
	intermediate: number
): { Wup: Mat; Wdown: Mat } {
	return {
		Wup: randMat(dModel, intermediate, rnd),
		Wdown: randMat(intermediate, dModel, rnd)
	};
}

/**
 * 把每一行归一化到单位长度。
 *
 * 用在路由器权重上：否则行范数最大的专家会**系统性地赢下所有 token**，
 * 路由结果退化成"只有个别专家被激活"，看不出 MoE 真正想表达的
 * "不同 token 去不同专家"。归一化后胜负只由方向（与 token 的对齐程度）决定。
 */
export function normalizeRows(m: Mat): Mat {
	return m.map((row) => {
		let s = 0;
		for (const v of row) s += v * v;
		const n = Math.sqrt(s) || 1;
		return row.map((v) => v / n);
	});
}
