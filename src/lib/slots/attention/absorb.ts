/**
 * MLA 的**矩阵吸收合并**执行方式（absorb）。
 *
 * 这是"怎么算"而不是"算什么"——算的还是 MLA，只是把两个大矩阵**折进别的矩阵**，
 * 于是 decode 阶段完全不用物化 K/V，连 `kv_b_proj`（真实 32768 × 512）都不参与。
 * 所以它是 `ImplementationSpec`，与 `mla` 语义正交（见 core/types.ts）。
 *
 * 数学依据：
 *
 * **记法先讲清楚**：论文按 `[out × in]` 写（`W c`），本仓库按 `[in × out]` 存（`matmul(x, W)`），
 * 所以同一个矩阵在两边写出来**差一个转置**。下面两个式子都给两种写法，别混着抄——
 * 掺着抄会让展示的左操作数多（或少）一个转置，页面上"取行 × 取列"就和格子里的数对不上。
 *
 *   ① 把 W^UK 折进 Q 的投影
 *        q^C·k^C = (W^UQ c^Q)·(W^UK c^KV) = c^Q·(W^UQᵀ W^UK)·c^KV
 *      论文记 W̄^UQ = W^UQᵀ W^UK，仓库记 W̄^UQ = W^UQ·W^UKᵀ —— 同一个矩阵。
 *      于是"吸收后的查询" q̄ = c^Q · W̄^UQ **直接落在潜空间**。
 *
 *   ② 把 W^UV 折进输出投影
 *        u = W^O·Concat(v_1 … v_H)，其中 v_h = W^UV_h·o_h  ⇒  u = Σ_h W^O_h W^UV_h·o_h
 *      论文记 W̄^O = W^O W^UV，仓库记 W̄^O = W^UV·W^O
 *      （每头一块 `[d_c × d_v] · [d_v × d]` —— 收缩维是 d_v，**没有转置**）。
 *      于是 u = Concat(o)·W̄^O：输入从"每头 d_v"变成"每头 d_c"，正好吃下注意力在潜空间的输出。
 *
 *   ③ 于是注意力里的 **K 和 V 都是 c^KV 本身**（每个头一份，内容完全相同），
 *      位置编码段 k^R 照旧单独算、单独缓存（旋转与低秩投影不可交换）。
 *
 * 步骤（与朴素版页数相同，但内容完全换了一套）：
 *   1 Q 低秩压缩          c_Q = W_DQ·X
 *   2 合并权重 ★          W̄^UQ = W^UQ·W^UKᵀ、W̄^O = W^UV·W^O
 *   3 吸收后的 Q          c_Q --RMSNorm--> c_Q~ --× W̄^UQ--> q̄
 *   4 KV 低秩压缩         [c_KV | k_pe] = W_a·X
 *   5 缓存直接当 K/V ★    c_KV --RMSNorm--> c_KV~，复制 H 份既是 K 又是 V
 *   6 RoPE               q_pe / k_pe 各自旋转
 *   7 分数                S_h = [q̄_h | q^R_h]·[c_KV~ | k^R]ᵀ / √(d_h+d_h^R)
 *   8 掩码 → softmax → P·V   V 就是 c_KV~
 *   9 拼接 → W̄^O          输出用合并后的投影矩阵
 */
import {
	matmul,
	mergeHeads,
	rope,
	rmsNorm,
	softmaxRow,
	splitHeads,
	transpose,
	type Mat
} from '$lib/core/mat';
import type { DataflowEdge, DataflowNode, DataflowSpec, ImplementationSpec } from '$lib/core/types';
import { REAL_R1 } from '$lib/model/config';
import { DIAGRAM_EDGES, DIAGRAM_NODES, type MlaWeights } from './mla';

export interface MlaAbsorbTrace {
	out: Mat;
	x: Mat;

	// ── 合并出来的两个矩阵（真实推理只在加载权重时算一次）──
	/** [q_lora_rank × n_h·kv_lora_rank] —— 吸收了 W^UK 的查询投影 */
	WbarUQ: Mat;
	/** [n_h·kv_lora_rank × d_model] —— 吸收了 W^UV 的输出投影 */
	WbarO: Mat;

	// ── Q ────────────────────────────────────────
	cQ: Mat;
	cQNorm: Mat;
	/** [seq × n_h·kv_lora_rank] 吸收后的查询，已经落在潜空间 */
	qBar: Mat;
	qBarH: number[][][];
	/** RoPE 之前 */
	qPeRaw: number[][][];
	/** RoPE 之后 */
	qPe: number[][][];

	// ── KV ───────────────────────────────────────
	cKv: Mat;
	cKvNorm: Mat;
	kPeRaw: Mat;
	kPe: Mat;

	// ── 注意力 ────────────────────────────────────
	scores: number[][][];
	maskedScores: number[][][];
	probs: number[][][];
	/** [head][seq][kv_lora_rank] —— 输出在潜空间，不是 v_head_dim */
	headOut: number[][][];
	concat: Mat;
	mask: Mat;
	causal: boolean;
}

const NEG_INF = Number.NEGATIVE_INFINITY;



// ── 吸收合并版的数据流图：**在朴素版那张图上改** ─────────────
/**
 * 关键决定：**布局与朴素版完全一样**，只做差异标注。
 *
 * 直接复用 `mla.ts` 的节点表，只改三处：
 *
 *   - 不再需要的矩阵（`[k_nope | v]` / `k_nope` / `v`）→ 灰色虚线 + 淡显（`gone`）；
 *   - 被改动的环节（`q` → `q̄`、折出来的 `W̄^UQ` / `W̄^O`、换了来源的 `K_h` / `V_h`）
 *     → 青色（`absorbed`），并各自在小字里说明改了什么。
 *
 * **不画"新的连线"**：K 的 nope 段与 V 虽然直接从 `c_KV~` 来，但那条线要横穿第 2 行——
 * 而那一行正是三个灰节点所在，线会被盖住，所以把"现在从哪来"写在 `K_h` / `V_h` 的小字里。
 *
 * 于是切换实现时，图是"同一张图变了颜色"，差在哪一眼可见。
 */
const GONE = new Set(['kv', 'knope', 'v']);
const PATCH: Record<string, { label?: string; sub?: string; tone: 'gone' | 'absorbed' }> = {
	// 不再需要的三个矩阵
	kv: { tone: 'gone', sub: '上投影矩阵 · 不再需要' },
	knope: { tone: 'gone', sub: '每头前 3 列 · 不再需要' },
	v: { tone: 'gone', sub: '每头后 3 列 · 不再需要' },
	// 被改动的环节
	q: { tone: 'absorbed', label: 'q̄', sub: '[6 × 4] 吸收后直接落在潜空间' },
	kh: { tone: 'absorbed', sub: 'nope 段直接用 c_KV~，再接 rope 段' },
	vh: { tone: 'absorbed', sub: '就是 c_KV~ 本身' }
};

const ABS_NODES: DataflowNode[] = [
	...DIAGRAM_NODES.map((n) => {
		const patch = PATCH[n.id];
		if (!patch) return n;
		return { ...n, ...patch, faded: patch.tone === 'gone' };
	}),
	// 折出来的两个权重：贴在各自消费者的左下方，用青色标出"这是吸收合并新增的"
	{ id: 'wbaruq', label: 'W̄^UQ', sub: 'W^UQ·W^UKᵀ', col: 2, row: 1, tone: 'absorbed' },
	{ id: 'wbaro', label: 'W̄^O', sub: 'W^UV·W^O', col: 5, row: 1, tone: 'absorbed' }
];

const ABS_EDGES: DataflowEdge[] = [
	// 朴素版的边照抄；碰到"不再需要"的节点的那些边一并淡显
	...DIAGRAM_EDGES.map((e) =>
		GONE.has(e.from) || GONE.has(e.to) ? { ...e, tone: 'gone' as const, faded: true } : e
	),
	// 新增：折出来的权重喂给各自的消费者
	{ from: 'wbaruq', to: 'q', routeOffset: 20 },
	{ from: 'wbaro', to: 'attn', routeOffset: 34 }
];

/** 取这张图、把本步正在算的节点标出来 */
function flow(active: string[], hint?: string): DataflowSpec {
	return { nodes: ABS_NODES, edges: ABS_EDGES, active, hint: hint ?? ABS_HINT };
}

const ABS_HINT =
	'这张图和朴素版是**同一张**，只是换了颜色：青色 = 吸收合并改动的环节（点开小字看改了什么），灰色虚线 + 淡显 = 朴素版要走、现在不再需要的矩阵与路径。对照着看最直观：`[k_nope | v]` / `k_nope` / `v` 整条灰掉了。';

/** 取第 h 个头的列块：[rows × n_h·perHead] → [rows × perHead] */
function headCols(m: Mat, h: number, perHead: number): Mat {
	return m.map((r) => r.slice(h * perHead, (h + 1) * perHead));
}
/** 取第 h 个头的行块：[n_h·perHead × cols] → [perHead × cols] */
function headRows(m: Mat, h: number, perHead: number): Mat {
	return m.slice(h * perHead, (h + 1) * perHead);
}

export const mlaAbsorb: ImplementationSpec<MlaAbsorbTrace> = {
	id: 'mla-absorb',
	slot: 'attention',
	name: '矩阵吸收合并',
	desc: '把 W^UK 折进 Q 的投影、W^UV 折进输出投影，于是 K 和 V 就是缓存里的 c_KV 本身——decode 阶段连 K/V 都不用物化',
	supports: ['mla'],

	compute(x, ctx) {
		const cfg = ctx.cfg;
		const { num_heads: H, seq_len: S, causal } = cfg;
		const dn = req(cfg.qk_nope_head_dim, 'qk_nope_head_dim');
		const dr = req(cfg.qk_rope_head_dim, 'qk_rope_head_dim');
		const dv = req(cfg.v_head_dim, 'v_head_dim');
		const dc = req(cfg.kv_lora_rank, 'kv_lora_rank');
		const qr = req(cfg.q_lora_rank, 'q_lora_rank');
		// 用的是**语义插件那批权重**——吸收合并不产生新权重
		const { Wdq, Wuq, Wa, Wukv, Wo } = ctx.w as MlaWeights;

		// ── ① 合并权重（一次性） ──────────────────────
		// W̄^UQ = W^UQ · W^UKᵀ：把"每头的 k_nope 上投影"折进 Q。
		// W^UK 存在 Wukv 里（每头的前 dn 列），W^UQ 每头的前 dn 列是对应的 nope 部分。
		const WbarUQ: Mat = Array.from({ length: qr }, () => new Array<number>(H * dc).fill(0));
		const WbarO: Mat = Array.from({ length: H * dc }, () => new Array<number>(cfg.d_model).fill(0));
		for (let h = 0; h < H; h++) {
			const wuqNope = headCols(Wuq, h, dn + dr); // [qr × (dn+dr)]
			const wukNope = headCols(Wukv, h, dn + dv); // [dc × (dn+dv)]
			const wuvV = headCols(Wukv, h, dn + dv); // 同一块的 v 部分
			for (let i = 0; i < qr; i++)
				for (let c = 0; c < dc; c++) {
					let s = 0;
					for (let p = 0; p < dn; p++) s += wuqNope[i][p] * wukNope[c][p];
					WbarUQ[i][h * dc + c] = s;
				}
			// W̄^O_h = W^UV_hᵀ · W^O_h = [dc × dv] · [dv × d]
			const woHead = headRows(Wo, h, dv); // [dv × d]
			for (let c = 0; c < dc; c++)
				for (let j = 0; j < cfg.d_model; j++) {
					let s = 0;
					for (let p = 0; p < dv; p++) s += wuvV[c][dn + p] * woHead[p][j];
					WbarO[h * dc + c][j] = s;
				}
		}

		// ── ② Q：下投影 → 归一化 → **吸收后的**上投影 ──
		const cQ = matmul(x, Wdq);
		const cQNorm = rmsNorm(cQ);
		const qBar = matmul(cQNorm, WbarUQ); // [S × n_h·dc] 直接落在潜空间
		const qBarH = splitHeads(qBar, H, dc);
		// rope 段：还是从 c_Q~ 投影出来（W^QR 就是 W^UQ 每头的后 dr 列）
		const Wqr: Mat = Wuq.map((r) => {
			const out: number[] = [];
			for (let h = 0; h < H; h++) out.push(...r.slice(h * (dn + dr) + dn, h * (dn + dr) + dn + dr));
			return out;
		});
		const qPeRaw = splitHeads(matmul(cQNorm, Wqr), H, dr);
		const qPe = qPeRaw.map(rope);

		// ── ③ KV：下投影（和朴素版完全一样） ──────────
		const kvA = matmul(x, Wa);
		const cKv = kvA.map((r) => r.slice(0, dc));
		const kPeRaw = kvA.map((r) => r.slice(dc));
		const cKvNorm = rmsNorm(cKv);
		const kPe = rope(kPeRaw);

		// ── ④ 注意力：K 和 V **都是 c_KV~ 本身** ────────
		// kFull 每个头都一样（c_KV 是所有头共享的）——这就是参考图里的 "Replicate n_h"
		const kFull = cKvNorm.map((r, t) => [...r, ...kPe[t]]); // [S × (dc+dr)]
		const scale = 1 / Math.sqrt(dn + dr); // 缩放因子属于语义，不因实现而变
		const scores: number[][][] = [];
		const maskedScores: number[][][] = [];
		const probs: number[][][] = [];
		const headOut: number[][][] = [];

		for (let h = 0; h < H; h++) {
			const qFull = qBarH[h].map((r, t) => [...r, ...qPe[h][t]]);
			const raw = matmul(qFull, transpose(kFull)).map((row) => row.map((z) => z * scale));
			const masked = causal
				? raw.map((row, i) => row.map((val, j) => (j <= i ? val : NEG_INF)))
				: raw;
			const p = masked.map(softmaxRow);
			scores.push(raw);
			maskedScores.push(masked);
			probs.push(p);
			headOut.push(matmul(p, cKvNorm)); // V = 潜向量本身
		}

		const concat = mergeHeads(headOut, dc); // [S × n_h·dc]
		const mask = Array.from({ length: S }, (_, i) =>
			Array.from({ length: S }, (_, j) => (causal && j > i ? 0 : 1))
		);

		return {
			out: matmul(concat, WbarO), // 输出投影也是合并后的
			x,
			WbarUQ,
			WbarO,
			cQ,
			cQNorm,
			qBar,
			qBarH,
			qPeRaw,
			qPe,
			cKv,
			cKvNorm,
			kPeRaw,
			kPe,
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
		const dn = req(cfg.qk_nope_head_dim, 'qk_nope_head_dim');
		const dr = req(cfg.qk_rope_head_dim, 'qk_rope_head_dim');
		const dv = req(cfg.v_head_dim, 'v_head_dim');
		const dc = req(cfg.kv_lora_rank, 'kv_lora_rank');
		const qr = req(cfg.q_lora_rank, 'q_lora_rank');
		const dqk = dn + dr;
		const { Wdq, Wuq, Wa, Wukv, Wo } = ctx.w as MlaWeights;

		const R = REAL_R1;
		const realSeq = S;
		const realDqk = R.num_heads * (R.head_dim + R.rope_head_dim);
		const realKv = R.num_heads * (R.head_dim + R.v_head_dim);

		const X_REF = {
			name: 'X',
			shape: [S, D],
			data: trace.x,
			realShape: [realSeq, R.d_model],
			handoffId: 'x'
		};
		const ref = (name: string, data: Mat, shape: number[], realShape: number[]) => ({
			name,
			shape,
			data,
			realShape
		});

		// 用于展示的两个"每头第 0 头"切片（整块太大，画不下）
		// 注意 `wuqNope0` 只取**前 dn 列**（nope 部分）：这一页算的是 `W̄^UQ`，
		// 收缩维就是 dn；把后面的 RoPE 段也带上会多出几列、和 `W_UKᵀ` 的行数对不上
		const wuqNope0 = headCols(Wuq, 0, dn + dr).map((r) => r.slice(0, dn)); // [qr × dn]
		const wukvHead0 = headCols(Wukv, 0, dn + dv); // [dc × (dn+dv)]
		const woHead0 = headRows(Wo, 0, dv); // [dv × d]

		return [
			// ── 1. Q 低秩压缩（与朴素版相同） ──────────────
			{
				id: 'abs-q-compress',
				diagram: flow(['x', 'cq']),
				kind: 'PROJECT',
				label: `和朴素版一样：Q 先压到 ${qr} 维的低秩潜向量（真实 ${R.q_lora_rank}）。吸收合并改的是「后面怎么用」它，不改这一步`,
				formula: 'c^Q=W^{DQ}X',
				tensors: [
					{
						name: 'abs-q-down',
						kind: 'matmul',
						shape: [S, qr],
						a: X_REF,
						b: ref('W_DQ', Wdq, [D, qr], [R.d_model, R.q_lora_rank]),
						out: {
							name: 'c_Q',
							shape: [S, qr],
							data: trace.cQ,
							realShape: [realSeq, R.q_lora_rank],
							label: '↓ 第 {{step:abs-q-absorb}} 步先归一化、再用**合并后**的投影'
						}
					}
				]
			},

			// ── 2. 合并权重（吸收合并的核心） ──────────────
			{
				id: 'abs-merge',
				diagram: flow(['wbaruq', 'wbaro']),
				kind: 'PROJECT',
				label: `「这就是"吸收合并"」：把两个上投影矩阵折进别的矩阵里。真实推理只在加载权重时算一次，之后 decode 阶段就不用它们了。这里画的是 head 0 那两块`,
				formula:
					'\\bar W^{UQ}=W^{UQ}W^{UK\\top},\\qquad \\bar W^{O}=W^{UV}W^{O}',
				tensors: [
					{
						name: 'abs-merge-uq',
						label: `① 把 W^UK 折进 Q：以后 q 直接乘出 ${dc} 维的潜空间向量，不用再算 k_nope`,
						kind: 'matmul',
						shape: [qr, dc],
						a: ref('W_UQ(nope)', wuqNope0, [qr, dn], [R.q_lora_rank, R.head_dim]),
						b: ref('W_UKᵀ', transpose(wukvHead0.map((r) => r.slice(0, dn))), [dn, dc], [
							R.head_dim,
							R.kv_lora_rank
						]),
						out: {
							name: 'W̄^UQ',
							shape: [qr, dc],
							data: headCols(trace.WbarUQ, 0, dc),
							realShape: [R.q_lora_rank, R.kv_lora_rank],
							label: '↓ 第 {{step:abs-q-absorb}} 步用它'
						}
					},
					{
						name: 'abs-merge-o',
						label: `② 把 W^UV 折进输出投影：这一步**不用转置**——W_UV 是 [d_c × d_v]、W_O 是 [d_v × d]，收缩维 d_v 正好对上（上一个式子要转置，是因为 W_UK 的收缩维 d_h 在右边）`,
						kind: 'matmul',
						shape: [dc, D],
						a: ref('W_UV', wukvHead0.map((r) => r.slice(dn)), [dc, dv], [
							R.kv_lora_rank,
							R.v_head_dim
						]),
						b: ref('W_O', woHead0, [dv, D], [R.v_head_dim, R.d_model]),
						out: {
							name: 'W̄^O',
							shape: [dc, D],
							data: headRows(trace.WbarO, 0, dc),
							realShape: [R.kv_lora_rank, R.d_model],
							label: '↓ 第 {{step:abs-out}} 步用它'
						}
					}
				]
			},

			// ── 3. 吸收后的 Q ──────────────────────────────
			{
				id: 'abs-q-absorb',
				diagram: flow(['cq', 'cqn', 'q']),
				kind: 'PROJECT',
				label: `用合并后的 W̄^UQ 投影：算出来的 q̄ 已经「直接落在 ${dc} 维的潜空间」（真实 ${R.kv_lora_rank} 维），所以它和缓存里的 c_KV 天生同维、可以直接做内积。归一化与投影按先后顺序算`,
				formula: '\\tilde c^Q=\\mathrm{RMSNorm}(c^Q),\\qquad \\bar q=W\\!\\bar{\\,}^{UQ}\\tilde c^Q',
				tensors: [
					{
						name: 'abs-q-absorb',
						kind: 'transform',
						shape: [S, qr],
						op: 'RMSNorm',
						input: {
							name: 'c_Q',
							shape: [S, qr],
							data: trace.cQ,
							realShape: [realSeq, R.q_lora_rank]
						},
						output: {
							name: 'c_Q~',
							shape: [S, qr],
							data: trace.cQNorm,
							realShape: [realSeq, R.q_lora_rank],
							label: '← 归一化后同时充当下面这次矩阵乘的左操作数'
						},
						then: {
							op: '× W̄^UQ',
							b: ref('W̄^UQ', trace.WbarUQ, [qr, H * dc], [R.q_lora_rank, R.num_heads * R.kv_lora_rank]),
							result: {
								name: 'q̄',
								shape: [S, H * dc],
								data: trace.qBar,
								realShape: [realSeq, R.num_heads * R.kv_lora_rank],
								label: `↓ ${H} 头 × ${dc} 维，每头一个潜空间查询`
							}
						}
					}
				]
			},

			// ── 4. KV 低秩压缩（与朴素版相同） ──────────────
			{
				id: 'abs-kv-compress',
				diagram: flow(['x', 'ckv', 'kpe']),
				kind: 'PROJECT',
				label: `KV 的下投影和朴素版一模一样：一次算出共享潜向量 c_KV 和位置编码段 k_pe。「位置编码段不能吸收」——旋转与低秩投影不可交换`,
				formula: '[\\,c^{KV}\\;|\\;k^{R}_{\\text{raw}}\\,]=W^{DKV+KR}X',
				tensors: [
					{
						name: 'abs-kv-down',
						kind: 'matmul',
						shape: [S, dc + dr],
						a: X_REF,
						b: ref(
							'W_DKV+KR',
							Wa,
							[D, dc + dr],
							[R.d_model, R.kv_lora_rank + R.rope_head_dim]
						),
						out: {
							name: '[c_KV | k_pe]',
							shape: [S, dc + dr],
							data: trace.cKv.map((r, t) => [...r, ...trace.kPeRaw[t]]),
							realShape: [realSeq, R.kv_lora_rank + R.rope_head_dim],
							label: `↓ 前 ${dc} 列进缓存后「直接当 K 和 V」`
						}
					}
				]
			},

			// ── 5. 缓存直接当 K/V（★ 参考图里的 Replicate） ──
			{
				id: 'abs-replicate',
				diagram: flow(['ckvn', 'kh', 'vh', 'cache']),
				kind: 'STORE',
				label: `「吸收合并最关键的收益」：缓存里的 c_KV 归一化后「复制 ${H} 份，既是 K 又是 V」——每个头拿到的内容完全相同，所以只存一份就够了。朴素版这一步要过一个 ${realKv}×${R.kv_lora_rank} 的大矩阵，现在完全不需要`,
				formula:
					'K_h = V_h = \\mathrm{Replicate}_{n_h}\\!\\left(\\tilde c^{KV}\\right),\\qquad \\forall h',
				tensors: [
					{
						name: 'abs-kv-norm',
						kind: 'transform',
						shape: [S, dc],
						op: 'RMSNorm',
						input: {
							name: 'c_KV',
							shape: [S, dc],
							data: trace.cKv,
							realShape: [realSeq, R.kv_lora_rank]
						},
						output: {
							name: 'c_KV~',
							shape: [S, dc],
							data: trace.cKvNorm,
							realShape: [realSeq, R.kv_lora_rank],
							label: `← 复制 ${H} 份就是 K 和 V（每头内容相同）`
						}
					},
					{
						name: 'abs-replicate-tiles',
						label: `复制出来的 K / V：内容一模一样，所以缓存里只需要存一份 c_KV~`,
						kind: 'tiles',
						shape: [2 * H, S, dc],
						data: [...Array.from({ length: H }, () => trace.cKvNorm), ...Array.from({ length: H }, () => trace.cKvNorm)],
						tileLabels: [
							...Array.from({ length: H }, (_, h) => `K（head ${h}）`),
							...Array.from({ length: H }, (_, h) => `V（head ${h}）`)
						]
					},
					{
						name: 'abs-cache',
						kind: 'kvcache',
						shape: [S, dc + dr],
						blocks: [
							{
								name: 'c_KV',
								label: `所有头共享的潜向量（真实 ${R.kv_lora_rank} 维）——它同时是 K 和 V`,
								size: dc,
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
						perToken: dc + dr,
						realPerToken: R.kv_lora_rank + R.rope_head_dim,
						compare: {
							label: 'MHA 要缓存 K 和 V',
							parts: [
								{ name: 'K', perToken: H * cfg.head_dim, sub: `每个头各存一份（${H} 头 × ${cfg.head_dim} 维）` },
								{ name: 'V', perToken: H * cfg.head_dim, sub: `每个头各存一份（${H} 头 × ${cfg.head_dim} 维）` }
							],
							realPerToken: 2 * R.num_heads * R.head_dim
						},
						note: `吸收合并下，缓存里的 c_KV 直接进注意力，「既不物化 K/V，也不参与任何上投影」——decode 阶段那个 ${realKv}×${R.kv_lora_rank} 的矩阵彻底不出现`
					}
				]
			},

			// ── 6. RoPE ────────────────────────────────────
			{
				id: 'abs-rope',
				diagram: flow(['kpe', 'kper', 'qpe', 'qh']),
				kind: 'ROPE',
				label: `位置编码照旧单独走：q 的 rope 段与 k_pe 各自按位置旋转。「它不能被吸收」（旋转与线性投影不可交换），所以必须单独算、单独缓存`,
				formula:
					'\\mathcal{R}(x,pos)_{2i,2i+1}=\\mathrm{Rot}(\\theta_i\\,pos)\\begin{bmatrix}x_{2i}\\\\ x_{2i+1}\\end{bmatrix}',
				tensors: [
					{
						name: 'abs-rope-q',
						label: 'q 的 rope 段（head 0）',
						kind: 'transform',
						shape: [S, dr],
						row: 1,
						parallel: true,
						op: 'RoPE',
						input: {
							name: 'q_pe',
							shape: [S, dr],
							data: trace.qPeRaw[0],
							realShape: [realSeq, R.rope_head_dim]
						},
						output: {
							name: "q_pe'",
							shape: [S, dr],
							data: trace.qPe[0],
							realShape: [realSeq, R.rope_head_dim],
							label: '← 拼在 q̄ 后面，一起做内积'
						}
					},
					{
						name: 'abs-rope-k',
						label: 'k 的 rope 段',
						kind: 'transform',
						shape: [S, dr],
						row: 1,
						parallel: true,
						op: 'RoPE',
						input: {
							name: 'k_pe',
							shape: [S, dr],
							data: trace.kPeRaw,
							realShape: [realSeq, R.rope_head_dim]
						},
						output: {
							name: "k_pe'",
							shape: [S, dr],
							data: trace.kPe,
							realShape: [realSeq, R.rope_head_dim],
							label: '↓ 这一份进缓存'
						}
					}
				]
			},

			// ── 7. 分数 ────────────────────────────────────
			{
				id: 'abs-score',
				diagram: flow(['qh', 'kh', 'vh', 'attn']),
				kind: 'MATMUL',
				label: `分数：吸收后的 q̄ 与 c_KV 内积（两边都是 ${dc} 维潜向量），再加 rope 段的内积。注意缩放因子仍是 1/√${dqk}——它属于语义，不因实现而变`,
				formula: 'S_h=\\frac{[\\bar q_h;q^{R}_h]\\,[\\tilde c^{KV};k^{R}]^{\\top}}{\\sqrt{d_h+d_h^{R}}}',
				tensors: [
					{
						name: 'abs-score-matmul',
						kind: 'matmul',
						shape: [S, S],
						a: {
							name: 'Q̄_h',
							shape: [S, dc + dr],
							data: trace.qBarH[0].map((r, t) => [...r, ...trace.qPe[0][t]]),
							realShape: [realSeq, R.kv_lora_rank + R.rope_head_dim],
							label: `← ${dc} 维潜空间查询 + ${dr} 维 rope`
						},
						b: {
							name: 'K_hᵀ',
							shape: [dc + dr, S],
							data: transpose(trace.cKvNorm.map((r, t) => [...r, ...trace.kPe[t]])),
							realShape: [R.kv_lora_rank + R.rope_head_dim, realSeq],
							label: '← c_KV~ 拼上 rope 段（每个头都一样）'
						},
						out: {
							name: 'S_h',
							shape: [S, S],
							data: trace.scores[0],
							realShape: [realSeq, realSeq]
						},
						scaleNote: `1/√${dqk} = ${(1 / Math.sqrt(dqk)).toFixed(2)}`
					},
					{
						name: 'abs-head-scores',
						label: `${H} 个头各算各的（K 侧每个头完全相同）`,
						kind: 'tiles',
						shape: [H, S, S],
						data: trace.scores,
						tileLabels: Array.from({ length: H }, (_, h) => `head ${h} [${S} × ${S}]`)
					}
				]
			},

			// ── 8. 掩码 → softmax → P·V ─────────────────────
			{
				id: 'abs-av',
				diagram: flow(['attn']),
				kind: 'MASK',
				label: `掩码 → softmax → 加权求和。「V 就是 c_KV~」，所以输出是 ${dc} 维潜向量（真实 ${R.kv_lora_rank} 维）而不是 ${dv} 维的 v——反正最后要乘 W̄^O，维度对不对得上由它负责`,
				formula:
					'S_{ij}=-\\infty\\ (j>i)\\ \\Rightarrow\\ P_{h}=\\mathrm{softmax}(S_h)\\ \\Rightarrow\\ o_h=P_h\\,\\tilde c^{KV}',
				tensors: [
					{
						name: 'abs-av',
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
								? `← 第 {{step:abs-score}} 步算出的分数，先掩码：j > i → ∅`
								: `← 第 {{step:abs-score}} 步算出的分数`
						},
						output: {
							name: 'P_h',
							shape: [S, S],
							data: trace.probs[0],
							realShape: [realSeq, realSeq],
							label: '← 每行归一化后（和为 1）'
						},
						then: {
							op: '× c_KV~',
							b: {
								name: 'c_KV~',
								shape: [S, dc],
								data: trace.cKvNorm,
								realShape: [realSeq, R.kv_lora_rank],
								label: '← 缓存里的潜向量，直接当 V'
							},
							result: {
								name: 'o_h',
								shape: [S, dc],
								data: trace.headOut[0],
								realShape: [realSeq, R.kv_lora_rank],
								label: `← 输出是 ${dc} 维潜向量，不是 ${dv} 维 v`
							}
						}
					}
				]
			},

			// ── 9. 拼接 → W̄^O ──────────────────────────────
			{
				id: 'abs-out',
				diagram: flow(['attn', 'wbaro']),
				kind: 'CONCAT',
				label: `${H} 个头各 ${dc} 维拼成 ${H * dc} 维，再乘「合并后的」 W̄^O 回到 ${D} 维。W̄^O 里已经含了 W^UV，所以这一步同时完成了"还原 v + 输出投影"两件事`,
				formula: 'u=\\bar W^{O}\\,\\mathrm{Concat}(o_1,\\dots,o_H),\\qquad \\bar W^{O}=W^{UV\\top}W^{O}',
				tensors: [
					{
						name: 'abs-concat',
						label: `拼接：每个头占 ${dc} 列`,
						kind: 'concat',
						shape: [S, H * dc],
						parts: trace.headOut.map((h, i) => ({
							name: `head ${i} 的输出`,
							label: `← 第 {{step:abs-av}} 步 head ${i} 算出的 o_h`,
							shape: [S, dc],
							realShape: [S, R.kv_lora_rank],
							data: h
						})),
						result: {
							name: 'Concat',
							shape: [S, H * dc],
							data: trace.concat,
							realShape: [realSeq, R.num_heads * R.kv_lora_rank],
							label: '← 下面这次矩阵乘的左操作数就是它'
						},
						then: {
							op: '× W̄^O',
							b: ref('W̄^O', trace.WbarO, [H * dc, D], [R.num_heads * R.kv_lora_rank, R.d_model]),
							result: {
								name: 'u',
								shape: [S, D],
								data: trace.out,
								realShape: [realSeq, R.d_model],
								handoffId: 'o',
								label: '↓ 这一层的输出（与朴素版**数值完全相同**）'
							}
						}
					}
				]
			}
		];
	}
};

function req(v: number | undefined, name: string): number {
	if (v === undefined) {
		throw new Error(`[mla-absorb] 配置缺少 ${name}`);
	}
	return v;
}
