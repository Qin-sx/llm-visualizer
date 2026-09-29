/**
 * DeepSeek V4 的**压缩器**（Compressor）与**检索器**（Indexer）。
 *
 * ## 压缩器：把 `ratio` 个 token 压成 1 条 KV
 *
 * 滑窗只看最近 `swa_window` 个 token，再往前的历史不能整个丢掉，也不能原样存
 * （缓存会爆）。V4 的办法是**分组压缩**：每 `ratio` 个 token 归一组，组内做一次
 * **加权平均**，得到 1 条压缩 KV。序列 16 个 token、`ratio = 4` 就是 4 条。
 *
 * 关键在于"权重从哪来"：不是均匀平均，而是让模型自己算——`W_gate` 把每个 token
 * 投影出「值」和「打分」两段，打分加上一个**按槽位学习的偏置** `ape` 之后做 softmax。
 * 于是"组里哪些 token 重要"是学出来的，而不是拍脑袋定的。
 *
 * 两处容易被忽略、但代码里确实如此的细节：
 *   - **逐通道 softmax**：`s` 段和 `v` 段一样宽（每个通道一个打分），所以权重
 *     **按列**归一化——`w[j][c]` 是槽位 j 在通道 c 上的权重。kernel（`c4_forward`）
 *     就是对每个 lane 各做一次 softmax。
 *   - **重叠窗口**（只有 `ratio = 4` 的 CSA 用）：每组看的不是自己的 4 个 token，
 *     而是「上一组 4 个 + 自己 4 个」共 8 个。所以每个 token 要投影出**两份**
 *     （在上一组里当尾巴、在自己组里当主体），gate 的宽度因此是 `2·coff·head_dim`，
 *     `coff = 1 + overlap`。这样组与组的边界是渐变的，而不是硬切。
 *
 * ## 检索器：从压缩条目里挑出该看的几条
 *
 * `ratio = 4` 的 CSA 层还额外带一个 indexer。它用**自己那套**（更窄的）压缩 KV
 * 给每条压缩条目打分，每个 query 只挑 top-k 条参与注意力——历史因此是"稀疏检索"
 * 而不是全看。`ratio = 128` 的 HCA 层没有 indexer，所有压缩条目都参与（稠密）。
 *
 * 打分是 MQA 式的：每个 indexer 头算 `q·k` 再过 ReLU，然后按头加权求和。
 * 真实的 indexer 还会对 q/k 做一次 Hadamard 旋转（为了量化友好）——Hadamard 是
 * 正交且对称的，点积里两边一起做会**相互抵消**，所以这里直接省掉。
 */
import { matmul, randMat, rmsNorm, ropeApply, softmaxRow, topK, type Mat, type Vec } from '$lib/core/mat';
import type { MatRef, ModelConfigLike, Step } from '$lib/core/types';
import { REAL_V4_FLASH } from '$lib/model/config';
import { v4Diagram, type V4Variant } from './v4-diagram';

const NEG_INF = Number.NEGATIVE_INFINITY;
const R = REAL_V4_FLASH;

function dot(a: Vec, b: Vec): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[i];
	return s;
}

/** 逐行 RMSNorm（压缩条目是单个向量，不是矩阵） */
function rmsNormRow(row: Vec): Vec {
	return rmsNorm([row])[0];
}

/** 逆 RoPE：把 `+θ·pos` 的旋转转回去（等价于用 `−pos` 旋转一次） */
export function ropeInverse(row: Vec, pos: number, theta: number): Vec {
	return ropeApply(row, -pos, theta);
}

/** 一个张量引用（带真实尺寸，界面据此标注缩小倍数） */
export function ref(
	name: string,
	data: Mat | undefined,
	shape: number[],
	realShape: number[],
	tone?: MatRef['tone'],
	label?: string
): MatRef {
	return { name, shape, data, realShape, tone, label };
}

// ══════════════════════════════════════════════════════════
// 压缩器
// ══════════════════════════════════════════════════════════
export interface CompressorWeights {
	/** `W_gate`: [d_model × 2·coff·head_dim] —— 一次投影出 [v_重叠 | v | s_重叠 | s] */
	Wgate: Mat;
	/** `ape`: [ratio·coff × head_dim] —— 窗口里每个槽位的学习偏置 */
	ape: Mat;
}

export interface CompressorPlan {
	/** 压缩比：几个 token 压成 1 条 */
	ratio: number;
	/** 是否用重叠窗口（CSA 用） */
	overlap: boolean;
	/** 压缩条目的宽度（= 注意力头维，或 indexer 头维） */
	headDim: number;
	/** 压缩条目做 RoPE 的底数 */
	ropeTheta: number;
}

/** 每个 token 的投影份数：重叠窗口要两份（自己组一份、上一组的尾巴一份） */
export function copiesOf(plan: CompressorPlan): number {
	return plan.overlap ? 2 : 1;
}

/** 每条压缩条目覆盖多少个 token */
export function windowOf(plan: CompressorPlan): number {
	return plan.ratio * copiesOf(plan);
}

export function makeCompressorWeights(
	rnd: () => number,
	cfg: ModelConfigLike,
	plan: CompressorPlan
): CompressorWeights {
	return {
		Wgate: randMat(cfg.d_model, 2 * copiesOf(plan) * plan.headDim, rnd),
		ape: randMat(windowOf(plan), plan.headDim, rnd)
	};
}

export interface CompressGroup {
	/** 组序号 */
	index: number;
	/** 窗口里的槽位 → token 下标；`-1` = 空槽（序列开头，还没有上一组） */
	slots: number[];
	/** 覆盖到的 token 区间（半开，已裁到 [0, S)） */
	from: number;
	to: number;
	/** 这条条目在哪个 token 上才算得出来（组的最后一个 token） */
	bornAt: number;
	/** [window × head_dim] 打分 = s + ape；空槽是 −∞ */
	scores: Mat;
	/** [window × head_dim] 权重——**按列**归一化 */
	weights: Mat;
	/** [window × head_dim] 1 / 0 */
	mask: Mat;
	/** [window × head_dim] 窗口里各槽位实际用的值（重叠区取重叠份） */
	values: Mat;
	/** [head_dim] 加权平均的结果 */
	pooled: Vec;
}

export interface CompressorTrace {
	plan: CompressorPlan;
	/** 压缩器的输入（本层注意力的输入 X） */
	x: Mat;
	/** [S × 2·coff·head_dim]，布局 [v_重叠 | v | s_重叠 | s]（不重叠时只有 [v | s]） */
	gate: Mat;
	/** 普通份：被平均的值 */
	v: Mat;
	/** 普通份：打分 */
	s: Mat;
	/** [nEntries × head_dim] 加权平均的结果 */
	entries: Mat;
	/** [nEntries × head_dim] RMSNorm 之后 */
	entriesNorm: Mat;
	/** [nEntries × head_dim] RoPE 之后——这就是**压缩 KV**（K = V） */
	entriesReady: Mat;
	groups: CompressGroup[];
}

/** 取某个槽位对应的 token 下标；`-1` 表示空槽 */
function slotToken(group: number, slot: number, plan: CompressorPlan, seqLen: number): number {
	const start = plan.ratio * group - (plan.overlap ? plan.ratio : 0);
	const t = start + slot;
	return t >= 0 && t < seqLen ? t : -1;
}

export function compress(x: Mat, w: CompressorWeights, plan: CompressorPlan): CompressorTrace {
	const { ratio, overlap, headDim: hd, ropeTheta } = plan;
	const coff = copiesOf(plan);
	const window = windowOf(plan);
	const S = x.length;

	// ① 一次投影出 [v_重叠 | v | s_重叠 | s]（不重叠时只有 [v | s]）
	const gate = matmul(x, w.Wgate);
	const section = (i: number) => gate.map((r) => r.slice(i * hd, (i + 1) * hd));
	const vOverlap = overlap ? section(0) : null;
	const v = section(overlap ? 1 : 0);
	const sOverlap = overlap ? section(2) : null;
	const s = section(overlap ? 3 : 1);

	/** 槽位 `slot` 用的值 / 打分：重叠区（`slot < ratio`）取重叠份，其余取普通份 */
	const valueAt = (slot: number, t: number) => (vOverlap && slot < ratio ? vOverlap[t] : v[t]);
	const scoreAt = (slot: number, t: number) => (sOverlap && slot < ratio ? sOverlap[t] : s[t]);

	// ② 分组：打分 → 逐通道 softmax → 加权平均
	const nEntries = Math.floor(S / ratio);
	const groups: CompressGroup[] = [];
	const entries: Mat = [];

	for (let g = 0; g < nEntries; g++) {
		const slots = Array.from({ length: window }, (_, j) => slotToken(g, j, plan, S));
		const bornAt = ratio * (g + 1) - 1;

		// 打分（加 ape）；空槽置 −∞，softmax 后权重自然是 0
		const scores: Mat = slots.map((t, j) =>
			Array.from({ length: hd }, (_, c) => (t < 0 ? NEG_INF : scoreAt(j, t)[c] + w.ape[j][c]))
		);
		const mask: Mat = slots.map((t) => Array.from({ length: hd }, () => (t < 0 ? 0 : 1)));
		const values: Mat = slots.map((t, j) =>
			t < 0 ? new Array<number>(hd).fill(0) : valueAt(j, t)
		);

		// 逐通道 softmax：**按列**归一化（kernel 里每个 lane 各做一次）
		const weights: Mat = Array.from({ length: window }, () => new Array<number>(hd).fill(0));
		for (let c = 0; c < hd; c++) {
			const col = softmaxRow(scores.map((row) => row[c]));
			for (let j = 0; j < window; j++) weights[j][c] = col[j];
		}

		const pooled: Vec = Array.from({ length: hd }, (_, c) =>
			slots.reduce((acc, t, j) => acc + (t < 0 ? 0 : weights[j][c] * values[j][c]), 0)
		);

		const valid = slots.filter((t) => t >= 0);
		groups.push({
			index: g,
			slots,
			from: valid.length ? Math.min(...valid) : 0,
			to: valid.length ? Math.max(...valid) + 1 : 0,
			bornAt,
			scores,
			weights,
			mask,
			values,
			pooled
		});
		entries.push(pooled);
	}

	// ③ 归一化 + RoPE：位置用它所在组的最后一个 token（条目就是在那一刻写出来的）
	const entriesNorm = entries.map(rmsNormRow);
	const entriesReady = entriesNorm.map((row, g) => ropeApply(row, groups[g].bornAt, ropeTheta));

	return { plan, x, gate, v, s, entries, entriesNorm, entriesReady, groups };
}

// ══════════════════════════════════════════════════════════
// 检索器（indexer）
// ══════════════════════════════════════════════════════════
export interface IndexerWeights {
	/** `W_q^B`: [q_lora_rank × index_n_heads·index_head_dim] —— indexer 自己的 Q 上投影 */
	Wqb: Mat;
	/** `W_w`: [d_model × index_n_heads] —— 把各头的打分合成一个分数 */
	Ww: Mat;
	/** indexer 自己那套压缩器（头维是 `index_head_dim`，比注意力的窄） */
	comp: CompressorWeights;
}

export interface IndexerTrace {
	plan: CompressorPlan;
	/** indexer 自己的压缩条目（打分对象） */
	compressor: CompressorTrace;
	/** Q 的低秩潜向量（和注意力共用同一份 `c_Q~`） */
	qLatent: Mat;
	/** [S × nH·headDim] RoPE 之后 */
	qFlat: Mat;
	/** [S][nH][headDim] */
	q: number[][][];
	/** [S × nH] 头权重（已含 `1/√(headDim·nH)` 缩放） */
	headWeights: Mat;
	/** [S × nEntries] 打分 = Σ_h w_h · ReLU(q_h·k_h)；还没写出来的条目是 −∞ */
	logits: Mat;
	/** 每个 query 能看到的条目下标 */
	candidates: number[][];
	/** 每个 query 选中的条目下标（按分数降序） */
	topk: number[][];
}

export function makeIndexerWeights(
	rnd: () => number,
	cfg: ModelConfigLike,
	plan: CompressorPlan
): IndexerWeights {
	const nH = cfg.index_n_heads ?? 1;
	return {
		Wqb: randMat(cfg.q_lora_rank ?? cfg.d_model, nH * plan.headDim, rnd),
		Ww: randMat(cfg.d_model, nH, rnd),
		comp: makeCompressorWeights(rnd, cfg, plan)
	};
}

export function runIndexer(
	x: Mat,
	qLatent: Mat,
	w: IndexerWeights,
	plan: CompressorPlan,
	cfg: ModelConfigLike
): IndexerTrace {
	const nH = cfg.index_n_heads ?? 1;
	const hd = plan.headDim;
	const k = cfg.index_topk ?? 1;
	const S = x.length;

	const compressor = compress(x, w.comp, plan);

	// ① Q：用同一个 Q 潜向量升维（比注意力的 Q 窄），按头过 RoPE
	const qFlatRaw = matmul(qLatent, w.Wqb);
	const q = Array.from({ length: S }, (_, i) =>
		Array.from({ length: nH }, (_, h) =>
			ropeApply(qFlatRaw[i].slice(h * hd, (h + 1) * hd), i, plan.ropeTheta)
		)
	);
	const qFlat = q.map((heads) => heads.flat());

	// ② 头权重：`1/√(headDim·nH)` 是 indexer 的缩放（真实是 `softmax_scale · n_heads^-0.5`）
	const scale = Math.pow(hd, -0.5) * Math.pow(nH, -0.5);
	const headWeights = matmul(x, w.Ww).map((row) => row.map((v) => v * scale));

	// ③ 打分：Σ_h w_h · ReLU(q_h·k_h)
	const kEntries = compressor.entriesReady;
	const logits = q.map((heads, i) =>
		kEntries.map((kv, e) => {
			if (compressor.groups[e].bornAt > i) return NEG_INF; // 这条还没写出来
			let acc = 0;
			for (let h = 0; h < nH; h++) acc += headWeights[i][h] * Math.max(0, dot(heads[h], kv));
			return acc;
		})
	);

	// ④ top-k：只在"已经写出来"的条目里挑
	const candidates = logits.map((row) =>
		row.map((v, e) => ({ v, e })).filter((o) => Number.isFinite(o.v)).map((o) => o.e)
	);
	const topk = logits.map((row) =>
		topK(row, Math.min(k, row.length)).filter((e) => Number.isFinite(row[e]))
	);

	return { plan, compressor, qLatent, qFlat, q, headWeights, logits, candidates, topk };
}

// ══════════════════════════════════════════════════════════
// 步骤（CSA 与 HCA 共用；indexer 的压缩器复用同一套，只是 `what` 不同）
// ══════════════════════════════════════════════════════════
/** 一条压缩条目在真实模型里覆盖多少个 token（c4 是 8，c128 是 128） */
function realWindow(plan: CompressorPlan): number {
	return plan.ratio === 4 ? 8 : 128;
}

/**
 * 压缩器的三步：gate 投影 → 组内加权平均 → 归一化 + RoPE 成 KV。
 *
 * `prefix` 是步骤 id 的前缀（如 `csa`），插件自己指定，方便跨步骤互相引用；
 * `what` 是这套压缩产物的用途（"注意力的压缩 KV" / "indexer 的打分对象"）；
 * `variant` 决定数据流图上哪条支路高亮。
 */
export function compressorSteps(
	trace: CompressorTrace,
	w: CompressorWeights,
	cfg: ModelConfigLike,
	prefix: string,
	what: string,
	variant: V4Variant
): Step[] {
	const plan = trace.plan;
	const { ratio, overlap, headDim: hd } = plan;
	const coff = copiesOf(plan);
	const window = windowOf(plan);
	const S = cfg.seq_len;
	const D = cfg.d_model;
	const nE = trace.groups.length;
	const g = trace.groups[nE - 1]; // 拿最后一条当例子：它的窗口是满的
	const rw = realWindow(plan);
	const flow = (active: string[]) => v4Diagram(variant, active);

	const gateShape = [S, 2 * coff * hd];
	const layout = overlap
		? `前 ${hd} 列 = 重叠份的值、接着 ${hd} 列 = 值、再 ${hd} 列 = 重叠份的打分、最后 ${hd} 列 = 打分`
		: `前 ${hd} 列 = 值、后 ${hd} 列 = 打分`;

	return [
		// ── 1. gate 投影 ──────────────────────────────
		{
			id: `${prefix}-compress-gate`,
			kind: 'PROJECT',
			diagram: flow(['x', 'gate']),
			label: `压缩的第一步：让模型自己决定「组里每个 token 该占多大权重」。W_gate 把每个 token 投影出 ${2 * coff * hd} 个数（真实 ${2 * coff * 512} 个）：${layout}。打分用来算权重、值才是被平均的对象——所以压缩不是简单平均，而是**学出来的加权平均**`,
			formula: overlap
				? '[\\,v^{\\text{ov}}\\;|\\;v\\;|\\;s^{\\text{ov}}\\;|\\;s\\,]=W_{gate}x'
				: '[\\,v\\;|\\;s\\,]=W_{gate}x',
			tensors: [
				{
					name: `${prefix}-gate-mm`,
					kind: 'transform',
					shape: gateShape,
					op: '× W_gate',
					input: ref('X', trace.x, [S, D], [S, R.d_model], 'input'),
					output: ref(
						'[v | s]',
						trace.gate,
						gateShape,
						[S, 2 * coff * 512],
						'latent',
						`← ${layout}`
					)
				},
				{
					name: `${prefix}-gate-note`,
					kind: 'shape',
					wide: true,
					shape: gateShape,
					note: overlap
						? `重叠窗口（压缩比 ${ratio}，真实 4）：每条压缩条目看 ${window} 个 token = 上一组的 ${ratio} 个 + 自己的 ${ratio} 个，所以每个 token 要投影出两份——在上一组里当尾巴、在自己组里当主体`
						: `不重叠：每条压缩条目看 ${ratio} 个 token（真实 ${rw} 个），每个 token 只投影一份`
				}
			]
		},

		// ── 2. 组内加权平均 ───────────────────────────
		{
			id: `${prefix}-compress-pool`,
			kind: 'SOFTMAX',
			diagram: flow(['gate', 'w', 'entries']),
			label: `组内加权平均：打分先加上一个**按槽位学习的偏置** ape（"在窗口里排第几"本身也是信息），再做 softmax 得到权重，最后用权重对值加权求和——${window} 个 token 就变成 1 条。注意打分和值一样宽，所以 softmax 是**逐通道**做的：每一**列**加起来是 1。下面拿第 ${nE} 条（覆盖 token ${g.from}~${g.to - 1}）当例子`,
			formula:
				'w_{j,c}=\\mathrm{softmax}_j\\left(s_{j,c}+a_{j,c}\\right),\\qquad \\bar v_c=\\sum_j w_{j,c}\\,v_{j,c}',
			tensors: [
				{
					name: `${prefix}-pool-softmax`,
					kind: 'transform',
					shape: [window, hd],
					op: 'softmax（逐通道）',
					input: ref(
						's + ape',
						g.scores,
						[window, hd],
						[rw, 512],
						undefined,
						`← 第 {{step:${prefix}-compress-gate}} 步算出的打分加窗口偏置；空槽是 −∞`
					),
					output: ref('w', g.weights, [window, hd], [rw, 512], undefined, '← 每一列加起来是 1')
				},
				{
					name: `${prefix}-pool-sum`,
					kind: 'sum',
					shape: [1, hd],
					terms: g.values.map((row, j) =>
						ref(
							g.slots[j] < 0 ? '空槽' : `v(${g.slots[j]})`,
							[row.map((v, c) => v * g.weights[j][c])],
							[1, hd],
							[1, 512],
							undefined,
							g.slots[j] < 0 ? '← 空槽权重为 0' : '← 已经乘过权重'
						)
					),
					result: ref(
						'v̄',
						[g.pooled],
						[1, hd],
						[1, 512],
						'latent',
						`← 第 ${nE} 条压缩条目：${window} 个 token 的值按权重叠成 1 条`
					)
				},
				{
					name: `${prefix}-pool-note`,
					kind: 'shape',
					wide: true,
					shape: [nE, hd],
					note: `整段序列压出 ${nE} 条：第 ${trace.groups
						.map((gr) => gr.index + 1)
						.join('、')} 条分别在 token ${trace.groups
						.map((gr) => gr.bornAt)
						.join('、')} 上写出来——**组收尾了才算得完**`
				}
			]
		},

		// ── 3. 归一化 + RoPE ──────────────────────────
		{
			id: `${prefix}-compress-kv`,
			kind: 'ROPE',
			diagram: flow(['entries', 'ckv']),
			label: `每条压缩条目再走一次 RMSNorm + RoPE，就成为可用的${what}。RoPE 的位置用**它所在组的最后一个 token**——条目就是在那一刻被写出来的。滑窗 KV 是"每个 token 一个向量"，压缩 KV 是"每组一个向量"：前者精确、后者省地方`,
			formula: '\\bar v_k=\\mathcal{R}\\!\\left(\\mathrm{RMSNorm}(\\bar v_k),\\; pos_{k}\\right)',
			tensors: [
				{
					name: `${prefix}-kv-norm-rope`,
					kind: 'transform',
					shape: [nE, hd],
					op: 'RMSNorm',
					input: ref('v̄', trace.entries, [nE, hd], [nE, 512]),
					output: ref('v̄~', trace.entriesNorm, [nE, hd], [nE, 512]),
					then: {
						op: 'RoPE',
						result: ref(
							`压缩 KV（${what}）`,
							trace.entriesReady,
							[nE, hd],
							[nE, 512],
							'cache',
							`← ${nE} 条压缩 KV；K = V 共用，所以这一份既当键又当值`
						)
					}
				}
			]
		}
	];
}

/** 检索器的两步：给压缩条目打分 → 每个 query 挑 top-k */
export function indexerSteps(
	trace: IndexerTrace,
	w: IndexerWeights,
	cfg: ModelConfigLike,
	prefix: string,
	variant: V4Variant
): Step[] {
	const plan = trace.plan;
	const nH = cfg.index_n_heads ?? 1;
	const hd = plan.headDim;
	const S = cfg.seq_len;
	const nE = trace.compressor.groups.length;
	const k = cfg.index_topk ?? 1;
	const D = cfg.d_model;
	const qr = cfg.q_lora_rank ?? D;
	const i = S - 1; // 拿最后一个 token 当例子：它能看到的条目最多
	const picked = trace.topk[i];
	const ic = trace.compressor;
	const flow = (active: string[]) => v4Diagram(variant, active);

	return [
		{
			id: `${prefix}-indexer-score`,
			kind: 'PROJECT',
			diagram: flow(['cqn', 'itop']),
			label: `indexer 用**自己那套**压缩 KV 打分——它的头维只有 ${hd}（真实 ${R.index_head_dim}，注意力的压缩 KV 是 ${R.head_dim}），因为这里只需要"够分辨谁重要"，不需要精确。压缩方式与上面**完全一样**（同一个 W_gate + 逐通道加权平均 + RMSNorm + RoPE），只是头维换了。每个 query 对每条压缩条目算 ${nH} 个头的 q·k 再过 ReLU，最后按头加权求和成一个分数`,
			formula: 'I_{i,e}=\\sum_h w_{i,h}\\,\\mathrm{ReLU}\\!\\left(q^{I}_{i,h}\\cdot k^{I}_{e}\\right)',
			tensors: [
				{
					name: `${prefix}-indexer-q`,
					kind: 'matmul',
					shape: [S, nH * hd],
					a: ref('c_Q~', trace.qLatent, [S, qr], [S, R.q_lora_rank], 'latent'),
					b: ref(
						'W_q^I',
						w.Wqb,
						[qr, nH * hd],
						[R.q_lora_rank, R.index_n_heads * R.index_head_dim],
						'weight'
					),
					out: ref(
						'q^I',
						trace.qFlat,
						[S, nH * hd],
						[S, R.index_n_heads * R.index_head_dim],
						'head',
						'← 按头过 RoPE 之后参与打分'
					)
				},
				{
					name: `${prefix}-indexer-k`,
					kind: 'transform',
					shape: [nE, hd],
					op: 'RMSNorm → RoPE',
					input: ref(
						'v̄^I',
						ic.entries,
						[nE, hd],
						[nE, R.index_head_dim],
						undefined,
						'← indexer 自己压出来的条目（和上面同一套压缩，只是头维更窄）'
					),
					output: ref(
						'压缩 KV（indexer 用）',
						ic.entriesReady,
						[nE, hd],
						[nE, R.index_head_dim],
						'cache',
						`← ${nE} 条；它们只负责"排序"，不参与注意力`
					)
				},
				{
					name: `${prefix}-indexer-note`,
					kind: 'shape',
					wide: true,
					shape: [nE, hd],
					note: `两套压缩 KV 各有各的 W_gate：注意力那套负责"被看"（宽 ${R.head_dim}），indexer 这套只负责"排序"（宽 ${R.index_head_dim}）。打分高的条目才会进入注意力，其余连 KV 都不必读——省的就是这部分`
				}
			]
		},
		{
			id: `${prefix}-indexer-topk`,
			kind: 'SELECT',
			diagram: flow(['itop']),
			label: `打分 → 挑 top-${k}：每个 query 只让 ${k} 条压缩条目参与注意力（真实 ${R.index_topk} 条）。序列还没走到那么远时，那条条目**根本还没写出来**（−∞），自然选不上。于是"远距离历史"从"全看"变成"检索着看"——这是 CSA 比 HCA 更省的地方`,
			formula:
				'\\mathcal{S}_i=\\mathrm{top\\text{-}}k\\left(\\{I_{i,e}\\}_{e\\,:\\;born(e)\\,\\le\\,i}\\right)',
			tensors: [
				{
					name: `${prefix}-indexer-logits`,
					kind: 'matrix',
					shape: [S, nE],
					data: trace.logits,
					animated: true,
					label: `打分矩阵 [${S} × ${nE}]：行 = query、列 = 压缩条目。−∞ = 那条还没写出来。真实是「序列长度 × 序列长度/压缩比」（16 万序列、128 倍压缩时是一千多条），也是不物化的中间量，所以这里没有"真实尺寸"那一行`,
					highlight: [i, picked[0] ?? 0]
				},
				{
					name: `${prefix}-indexer-topk-row`,
					kind: 'row',
					wide: true,
					shape: [nE],
					data: trace.logits[i],
					labels: trace.logits[i].map((v, e) =>
						picked.includes(e)
							? `条目 ${e}：被选中`
							: Number.isFinite(v)
								? `条目 ${e}：分数不够，没选上`
								: `条目 ${e}：还没写出来`
					),
					label: `最后一行（token ${i}）的打分：选中的是条目 ${picked.join('、')}`
				}
			]
		}
	];
}
