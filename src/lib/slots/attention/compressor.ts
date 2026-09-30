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
import type { MatRef, MatrixBand, ModelConfigLike, Step } from '$lib/core/types';
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
	label?: string,
	bands?: MatrixBand[]
): MatRef {
	return { name, shape, data, realShape, tone, label, bands };
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
	/** gate 投影的列数（缩放后 / 真实） */
	const gateCols = 2 * coff * hd;
	const gateRealCols = 2 * coff * R.head_dim;
	/**
	 * 每一段（值 / 打分 / 重叠份）在**真实模型**里的列数。
	 *
	 * 这里必须单独标出来：`[v | s]` 画成一块，读者只能看到合计的 `[8 × 1024]`，
	 * 看不出"值占多少列、打分占多少列"——而那正是这一步的关键（打分用来算权重、
	 * 值才是被平均的对象）。
	 */
	const realPart = R.head_dim;
	/** 真实里"值"那一半的总列数（重叠时 = 重叠份 + 主体） */
	const realValueCols = coff * realPart;
	const layout = overlap
		? `前 ${hd} 列 = 重叠份的值、接着 ${hd} 列 = 值、再 ${hd} 列 = 重叠份的打分、最后 ${hd} 列 = 打分`
		: `前 ${hd} 列 = 值、后 ${hd} 列 = 打分`;
	/** 把"每一段真实多宽"写成注解：`真实里值 1024 列（每段 512）、打分 1024 列` */
	const realLayout =
		`真实里每段都是 ${realPart} 列：**值 ${realValueCols} 列**（${overlap ? `重叠份 ${realPart} + 主体 ${realPart}` : '一份'}）` +
		`、**打分 ${realValueCols} 列**，合计 ${gateRealCols}`;
	/**
	 * `[v | s]` 的"真实尺寸"那行要分段标出值 / 打分各占多少列——
	 * 这块矩阵画成一个，光看合计的 `[8 × 1024]` 看不出这一步的关键
	 * （打分算权重、值被平均）。
	 */
	const realParts = overlap
		? [
				{ label: '重叠份的值', cols: realPart },
				{ label: '值', cols: realPart },
				{ label: '重叠份的打分', cols: realPart },
				{ label: '打分', cols: realPart }
			]
		: [
				{ label: '值', cols: realPart },
				{ label: '打分', cols: realPart }
			];
	/**
	 * `[v | s]` 里**打分那几列**的下标（`pool` 那一步要框出来）。
	 *
	 * 不重叠时是最后 `hd` 列；重叠窗口下是第 3、4 段（`s_ov` 和 `s`）——
	 * 因为窗口里"上一组的那几个 token"用的是重叠份的打分。
	 */
	const scoreColList = Array.from({ length: overlap ? 2 * hd : hd }, (_, i) =>
		(overlap ? 2 * hd : hd) + i
	);

	/**
	 * 每个 token 在**自己那一组**里排第几（ape 就是按这个槽位查表的）。
	 *
	 * 重叠窗口下"在自己组里"用的是后半段槽位（前半段是当上一组的尾巴用的）。
	 */
	const ownSlot = (t: number) => (overlap ? ratio : 0) + (t % ratio);
	/**
	 * 逐 token 的三份矩阵（都是 `[S × hd]`，**8 个 token 一起画**，不切窗口那 4 行）：
	 *
	 *   - `apeFull`：每个 token 拿到的那份偏置（同一组里按槽位查表，所以表里的行会重复）
	 *   - `sApeFull`：打分 + ape —— softmax 的输入
	 *   - `wFull`   ：每个 token 在自己那组窗口里归一化出来的权重
	 *
	 * 8 个 token 一起画，读者才看得到"这个偏置/权重是给谁的、别的 token 拿到的又是什么"。
	 * 注意 softmax 的作用域仍然是**同一组那 4 个槽位**（下面文案里写明）。
	 */
	const apeFull: Mat = Array.from({ length: S }, (_, t) => w.ape[ownSlot(t)]);
	const sApeFull: Mat = trace.s.map((row, t) => row.map((v, c) => v + apeFull[t][c]));
	const wFull: Mat = Array.from(
		{ length: S },
		(_, t) => trace.groups[Math.floor(t / ratio)].weights[ownSlot(t)]
	);
	/**
	 * 逐元素的 `w ⊙ v`（`[S × hd]`，**8 个 token 一起画**）。
	 *
	 * `v` 直接取上一步 `[v | s]` 的**值**那几列（`trace.v`），`w` 取上一步 softmax 的结果
	 * （`wFull`）——两个操作数都从上一页的完整矩阵拷过来，乘完才是 `w ⊙ v`。
	 */
	const prodFull: Mat = trace.v.map((row, t) => row.map((v, c) => v * wFull[t][c]));

	return [
		// ── 1. gate 投影 → 打分加 ape → softmax（一条线走完） ──
		{
			id: `${prefix}-compress-gate`,
			kind: 'PROJECT',
			diagram: flow(['x', 'gate', 'w']),
			label: `压缩的第一步：W_gate 把每个 token 投影出 ${gateCols} 个数（真实 ${gateRealCols} 个）：${layout}——${realLayout}。**框出的打分那几列**就是"这个 token 该占多大权重"的原料：加上一个**按槽位学习的偏置** ape（"在窗口里排第几"本身也是信息），再逐通道 softmax 就得到权重 w（打分和值一样宽，所以每一**列**加起来是 1）。下面拿第 ${nE} 条（覆盖 token ${g.from}~${g.to - 1}）当例子`,
			formula: overlap
				? '[\\,v^{\\text{ov}}\\;|\\;v\\;|\\;s^{\\text{ov}}\\;|\\;s\\,]=W_{gate}x,\\qquad w_{j,c}=\\mathrm{softmax}_j\\left(s_{j,c}+a_{j,c}\\right)'
				: '[\\,v\\;|\\;s\\,]=W_{gate}x,\\qquad w_{j,c}=\\mathrm{softmax}_j\\left(s_{j,c}+a_{j,c}\\right)',
			tensors: [
				// 这里是**真正的矩阵乘**（A/B/C 都画出来，能逐格看 `[v|s][i][j] = Σ_p X[i][p]·W_gate[p][j]`）。
				{
					name: `${prefix}-gate-mm`,
					kind: 'matmul',
					shape: gateShape,
					cellSize: 24,
					// 和下面那条打分链**并排一行**：`[v | s]` 就在链的左边，
					// 框出的打分列 → ape → s + ape → w 一眼连起来。
					// 保留矩阵乘的 2×2：`W_gate` 是这次乘法的右操作数，画在 `[v | s]` **上方**才读得对。
					// 链那一列怎么贴到矩阵乘的第二行上，见 `StepPanel` 里的 `rowOffsets`（量出来的偏移）。
					//
					// `fillWidth`：这一页只有两个块，横向空得很，于是**放大到占满可用宽度**
					// （`StepPanel` 量出来反推格子边长），注解也跟着改成占满各自矩阵的宽度。
					// 只给 HCA（不重叠窗口）开：重叠窗口下 gate 输出有 16 列，这一行本来就挤满了。
					row: 1,
					group: 'gate',
					fillWidth: !overlap,
					// 算完 `[v | s]` 之后**框出打分那几列**：下一步就是拿它加 ape
					highlightCols: scoreColList,
					a: ref('X', trace.x, [S, D], [S, R.d_model], 'input'),
					b: ref(
						'W_gate',
						w.Wgate,
						[D, gateCols],
						[R.d_model, gateRealCols],
						'weight',
						overlap
							? `← 真实 ${R.d_model} × ${gateRealCols}：重叠窗口下每个 token 要出 ${coff} 份`
							: undefined
					),
					out: {
						...ref(
							'[v | s]',
							trace.gate,
							gateShape,
							[S, gateRealCols],
							'latent',
							// 注解只留最短的一行——真实的分段尺寸由 `realParts` 走"真实尺寸"那一行，
							// 详细的分段说明在步骤文案里。胶囊是 nowrap + ellipsis 的，
							// 块宽被限到 9rem 之后长句子会被省略号截掉。
							overlap ? '← 值 | 打分（各两段）' : '← 值 | 打分'
						),
						realParts
					}
				},
				// 打分 → 加 ape → softmax：**同一行**三块（ape / s + ape / w），每块只有 ${hd} 列。
				// 放在这一步（而不是下一页）是因为它就是"权重从哪来"的答案。
				{
					name: `${prefix}-gate-softmax`,
					kind: 'transform',
					row: 1,
					group: 'chain',
					// 和左边的矩阵乘同属"放大占满宽度"的那一行（见上），格子要一致
					cellSize: 24,
					fillWidth: !overlap,
					shape: [S, hd],
					op: '+ s',
					input: ref(
						'ape',
						apeFull,
						[S, hd],
						[S, 512],
						'weight',
						`← 每个 token 拿到的那份按槽位偏置（表是真实 [${rw} × 512]，同一组里按槽位查表）；8 个 token 一起画`
					),
					output: ref(
						// 名字不带空格：窄块里 `s + ape [8 × 4]` 会折行，表头一高一低矩阵就错开
						's+ape',
						sApeFull,
						[S, hd],
						[S, 512],
						undefined,
						`← 上面框出的打分那几列 + 每个 token 自己那份 ape（空槽是 −∞）`
					),
					then: {
						// 算子名保持最短（"逐通道"写在输出块的注解里）：mid 那一格的宽度
						// 也是这一行的宽度来源之一
						op: 'softmax',
						result: ref(
							'w',
							wFull,
							[S, hd],
							[S, 512],
							undefined,
							'← 在自己那一组那 4 个槽位里逐列归一化，所以每列加起来是 1'
						)
					}
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

		// ── 2. 组内加权平均（权重已经在上一页算好了）→ 归一化 + RoPE（同一行、先后算） ──────
		{
			id: `${prefix}-compress-pool`,
			kind: 'SOFTMAX',
			diagram: flow(['gate', 'w', 'entries', 'ckv']),
			label: overlap
				? `组内加权平均：**权重 w 上一页已经算好了**（打分 + ape → softmax），这一步只用它：把这一条压缩条目看的 ${window} 个 token 的**值**逐元素乘上各自那一列的权重，再**按列**加起来——${window} 个 token 就变成 1 条。下面拿第 ${nE} 条（覆盖 token ${g.from}~${g.to - 1}）当例子`
				: `组内加权平均：**权重 w 上一页已经算好了**（打分 + ape → softmax）。这一步把上一页那两块**整块拷过来**——${S} 个 token 的**值 v** 和**权重 w** 各一块，逐元素相乘得到 **w ⊙ v**（每个 token 的值乘上它那一列的权重），再**按列**加起来：一条压缩条目只看其中 ${ratio} 个 token。**${S} 个 token 分成 ${nE} 组、一组一组算**——先算第 1 组（前 ${ratio} 个 token）得到第 1 条，再算第 2 组（后 ${ratio} 个）得到第 2 条；框出的就是**正在算的那一组**。**同一行右边紧接着做归一化 + RoPE**：v̄ 先过 RMSNorm 再按它所在组的最后一个 token 做 RoPE，就成了可用的${what}（先后计算，不并发）`,
			formula:
				'\\bar v_c=\\sum_j w_{j,c}\\,v_{j,c},\\qquad \\mathrm{kv}=\\mathcal{R}\\!\\left(\\mathrm{RMSNorm}(\\bar v),\\; pos_{group}\\right)',
			tensors: [
				// "乘过权重"要画成**矩阵**，不能只写一行小字说"已经乘过"。
				//
				// 不重叠时用 `ewise`（逐元素二元运算）视图：`w` 右上、`v` 左下、`w ⊙ v` 右下
				// （和矩阵乘同一个 2×2）；`v` 取上一步 `[v | s]` 的值那几列、`w` 取上一步 softmax 的结果。
				// 框出的是**这条条目看的窗口那几行**，按列加起来就是 `v̄`。
				//
				// 重叠窗口（CSA）下**不能用这个写法**：一个 token 会同时出现在两条窗口里
				// （自己组的主体、上一组的尾巴），两处权重不同，所以"逐 token 的 8 行"加起来
				// 不等于 `v̄`——那边保持"窗口那 `window` 行"的写法。
				//
				// 这一页**不再标 `fillWidth`**：两个块并排（`compact`），
				// 格子就按并排的统一值 22 走（`matmulCellSize` / `matrix` 视图都是它）——
				// 一行里两块边长一致，行高才一致、矩阵才对齐。
				...(overlap
					? [
							{
								name: `${prefix}-pool-prod`,
								kind: 'matrix' as const,
								row: 1,
								sync: true,
								shape: [window, hd],
								animated: true,
								revealOrder: 'col' as const,
								// 没有具名操作数可画，总览靠这两个 `overview` 把它画成 `w ⊙ v → v̄`
								overview: 'w ⊙ v',
								data: g.values.map((row, j) => row.map((v, c) => v * g.weights[j][c])),
								label: `← 权重 × 值（**逐元素**）：每个 token 的值乘上它那一列的权重${g.slots.some((t) => t < 0) ? '；空槽权重为 0' : ''}（真实 [${rw} × 512]）`
							},
							{
								name: `${prefix}-pool-result`,
								kind: 'matrix' as const,
								row: 1,
								sync: true,
								cellSize: 26, // 与同一行的 w⊙v / kv 链一致（c128 4-5-6-7 放大）
								shape: [nE, hd],
								animated: true,
								revealOrder: 'col' as const,
								// 整列一起出现：上游那一列加完，这一列才亮（换算式见 MatrixGrid.revealGroup）
								revealGroup: window,
								highlightRows: [nE - 1, nE - 1] as [number, number],
								overview: 'v̄',
								title: 'v̄',
								data: trace.entries,
								label: `← 整段序列压出的 ${nE} 条压缩条目（就是后面 ${nE} 条压缩 KV；上面算的是第 ${nE} 条）`
							}
						]
					: [
							{
								name: `${prefix}-pool-ewise`,
								kind: 'ewise' as const,
								row: 1,
								sync: true,
								cellSize: 26, // 与同一行的 v̄ / kv 链一致（c128 4-5-6-7 放大）
								shape: [S, hd],
								// 总览节点：`w ⊙ v → v̄`（`ewise` 没有具名操作数，靠 `overview` 声明）
								overview: 'w ⊙ v',
								op: '⊙',
								a: ref(
									'v',
									trace.v,
									[S, hd],
									[S, R.head_dim],
									'latent',
									'← 上一页 `[v | s]` 的值那几列'
								),
								b: ref(
									'w',
									wFull,
									[S, hd],
									[S, R.head_dim],
									'weight',
									'← 上一页 softmax 出来的权重'
								),
								out: ref(
									'w ⊙ v',
									prodFull,
									[S, hd],
									[S, R.head_dim],
									undefined,
									`← 逐元素相乘（**框出的那一组**就是正在算的：${ratio} 个 token → 1 条）`
								),
								// **逐组**揭示：8 个 token 分成 2 组，先算前 4 个（第 1 条）、再算后 4 个（第 2 条）。
								// 框不再写死（原来固定框最后那一组当例子），而是跟着动画走到"正在算的那一组"。
								groupRows: ratio,
								// 组内仍按列：一整列算完，右边的结果才出对应的那一格
								revealOrder: 'col' as const
							},
							{
								name: `${prefix}-pool-result`,
								kind: 'matrix' as const,
								row: 1,
								sync: true,
								cellSize: 26, // 与同一行的 w⊙v / kv 链一致（c128 4-5-6-7 放大）
								shape: [nE, hd],
								animated: true,
								// 上游每算完"一组的一列"就出一格（`groupRows` 与上面那个视图同一个数字），
								// 逐格按**行序**出现：先出完第 1 条，再出第 2 条
								groupRows: ratio,
								overview: 'v̄',
								title: 'v̄',
								data: trace.entries,
								label: `每 ${ratio} 个 token 压成 1 条（按列 Σ）`
							}
						]),
				// 归一化 + RoPE：与 w⊙v / v̄ **同一行**，但**先后计算**
				// ——先算完 pool 那一组，这一链才开始。`hideInput`：输入 v̄ 由左边 pool 的
				// 结果矩阵画（同一个矩阵只画一次），行首的 `RMSNorm ──▶` 引头指向 v̄。
				// 注意必须放在 pool-note **之前**：行分组按"连续同 row 的视图"切，
				// note 没有 row、夹在中间会把这一行打断。
				{
					name: `${prefix}-kv-norm-rope`,
					kind: 'transform',
					shape: [nE, hd],
					row: 1,
					cellSize: 26, // 与同一行的 w⊙v / v̄ 一致
					hideInput: true,
					op: 'RMSNorm',
					input: ref(
						'v̄',
						trace.entries,
						[nE, hd],
						[nE, 512],
						undefined,
						`← 左边 pool 按列加出来的 ${nE} 条压缩条目（每条 = 组内 ${window} 个 token 的值的加权平均）`
					),
					output: ref(
						'v̄~',
						trace.entriesNorm,
						[nE, hd],
						[nE, 512],
						undefined,
						'← 整条 RMSNorm（长度归一，方向不变）'
					),
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
				},
				{
					name: `${prefix}-pool-note`,
					kind: 'shape',
					wide: true,
					shape: [nE, hd],
					note: `每条压缩条目都**只在自己那组收尾时**才算得完：第 ${trace.groups
						.map((gr) => gr.index + 1)
						.join('、')} 条分别在 token ${trace.groups
						.map((gr) => gr.bornAt)
						.join('、')} 上写出来——所以整段序列压出 ${nE} 条，一条不多一条不少`
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
					// 总览节点：`I → top-k`（这两个视图都没有具名操作数，靠 `overview` 声明）
					overview: 'I',
					label: `打分矩阵 [${S} × ${nE}]：行 = query、列 = 压缩条目。−∞ = 那条还没写出来。真实是「序列长度 × 序列长度/压缩比」（16 万序列、128 倍压缩时是一千多条），也是不物化的中间量，所以这里没有"真实尺寸"那一行`,
					highlight: [i, picked[0] ?? 0]
				},
				{
					name: `${prefix}-indexer-topk-row`,
					kind: 'row',
					wide: true,
					shape: [nE],
					overview: 'top-k',
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
