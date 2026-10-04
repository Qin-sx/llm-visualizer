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
import { matmul, randMat, rmsNorm, ropeApply, softmaxRow, topK, transpose, type Mat, type Vec } from '$lib/core/mat';
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
	/** [window × head_dim] 打分 = s + ape；空槽是 ∅（≈ −∞） */
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
	/** 重叠份：上一组的尾巴份打分（重叠窗口才有；否则 null） */
	sOverlap: Mat | null;
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

		// 打分（加 ape）；空槽置 ∅（≈ −∞），softmax 后权重自然是 0
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

	return { plan, x, gate, v, s, sOverlap, entries, entriesNorm, entriesReady, groups };
}

// ══════════════════════════════════════════════════════════
// 检索器（indexer）
// ══════════════════════════════════════════════════════════
export interface IndexerWeights {
	/** `W_q^B`: [q_lora_rank × index_n_heads·index_head_dim] —— indexer 自己的 Q 上投影 */
	Wqb: Mat;
	/** `W_w`: [d_model × index_n_heads] —— 头权重投影（每头一个标量；与真实模型一致） */
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
	/** [S × nH] 头权重的**原始投影**（`x·W^w`，还没乘 `1/√(headDim·nH)` 缩放）——第 14 页矩阵乘的 C */
	wRaw: Mat;
	/** [S × nH] 头权重（已含 `1/√(headDim·nH)` 缩放）：每个 query 一行 w_0、w_1 */
	headWeights: Mat;
	/** [nH][S × nEntries] 每头的原始 q·k 分数（还没 ReLU、还没按头加权）——打分动画用 */
	headRaw: number[][][];
	/** [S × nEntries] 打分 = Σ_h w_h · ReLU(q_h·k_h)；还没写出来的条目是 ∅（≈ −∞） */
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

	// ② 头权重：先算原始投影 `w_raw = x·W^w`（第 14 页矩阵乘的 C），
	//     再乘 `1/√(headDim·nH)` 缩放得到 `headWeights`（真实是 `softmax_scale · n_heads^-0.5`）
	const scale = Math.pow(hd, -0.5) * Math.pow(nH, -0.5);
	const wRaw = matmul(x, w.Ww);
	const headWeights = wRaw.map((row) => row.map((v) => v * scale));

	// ③ 打分：每头先算原始 q·k（`headRaw`），再 Σ_h w_h · ReLU(·) 合成一个分数
	const kEntries = compressor.entriesReady;
	const headRaw: number[][][] = Array.from({ length: nH }, (_, h) =>
		q.map((heads, i) =>
			kEntries.map((kv, e) =>
				compressor.groups[e].bornAt > i ? NEG_INF : dot(heads[h], kv)
			)
		)
	);
	const logits = q.map((heads, i) =>
		kEntries.map((kv, e) => {
			if (compressor.groups[e].bornAt > i) return NEG_INF;
			let acc = 0;
			for (let h = 0; h < nH; h++) acc += headWeights[i][h] * Math.max(0, headRaw[h][i][e]);
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

	return {
		plan,
		compressor,
		qLatent,
		qFlat,
		q,
		wRaw,
		headWeights,
		headRaw,
		logits,
		candidates,
		topk
	};
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
	variant: V4Variant,
	/** 矩阵显示名的后缀：indexer 的压缩传 `^I`（和第 9 页的 `q^I` / `k^I` / `v̄^I` 命名一致），
	 *  注意力压缩不传（保持 `[v|s]`、`窗口槽位打分`、`v̄` 原名）。 */
	suffix = ''
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
		? `前 ${hd} 列 = 重叠份的值 v_overlap、接着 ${hd} 列 = 主体份的值 v、再 ${hd} 列 = 重叠份的打分 s_overlap、最后 ${hd} 列 = 主体份的打分 s（dsv4 记作 [\\,v^{\\text{ov}}\\;|\\;v\\;|\\;s^{\\text{ov}}\\;|\\;s\\,]）`
		: `前 ${hd} 列 = 值、后 ${hd} 列 = 打分`;
	/** 把"每一段真实多宽"写成注解：`真实里值 1024 列（每段 512）、打分 1024 列` */
	const realLayout =
		`真实里每段都是 ${realPart} 列：**值 ${realValueCols} 列**（${overlap ? `v_overlap ${realPart} + v ${realPart}` : '一份'}）` +
		`、**打分 ${realValueCols} 列**（${overlap ? `s_overlap ${realPart} + s ${realPart}` : '一份'}），合计 ${gateRealCols}`;
	/**
	 * `[v | s]` 的"真实尺寸"那行要分段标出值 / 打分各占多少列——
	 * 这块矩阵画成一个，光看合计的 `[8 × 1024]` 看不出这一步的关键
	 * （打分算权重、值被平均）。
	 */
	const realParts = overlap
		? [
				{ label: 'v_overlap', cols: realPart },
				{ label: 'v', cols: realPart },
				{ label: 's_overlap', cols: realPart },
				{ label: 's', cols: realPart }
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
	// 尾巴份：token t 被下一组当"上一组尾巴"时，在窗口里占槽位 t%ratio（前 ratio 个槽位）。
	// 所以它那份 ape 查的是表的前 ratio 行。只有 CSA（重叠）有。
	const apeTailFull: Mat | null = trace.sOverlap
		? Array.from({ length: S }, (_, t) => w.ape[t % ratio])
		: null;
	const sApeTailFull: Mat | null = trace.sOverlap && apeTailFull
		? trace.sOverlap.map((row, t) => row.map((v, c) => v + apeTailFull[t][c]))
		: null;
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

	// ── 重叠窗口（CSA）专用：把 4 条窗口各自的值 / 权重 / 乘积按窗口上下堆成一块 ──
	// 一个窗口 `window` 行，`nE` 条窗口堆起来 = [nE·window × hd]。逐组（window 行一组）揭示。
	// 不能用 HCA 那种"逐 token 的 8 行"：重叠下同一个 token 会同时出现在两条窗口里、
	// 权重不同，逐 token 行直接求和算不出条目——所以必须按"窗口"切，每条窗口对应该组自己那 1 条。
	const winStacked = (pick: (grp: (typeof trace.groups)[number]) => Mat): Mat =>
		trace.groups.flatMap((grp) => pick(grp));
	const vWinStacked: Mat = winStacked((grp) => grp.values);
	const wWinStacked: Mat = winStacked((grp) => grp.weights);
	/** 每条窗口的 `w ⊙ v` 乘积；每条窗口按列求和 = 那组自己的 1 条压缩条目 */
	const wvWinStacked: Mat = winStacked((grp) =>
		grp.values.map((row, j) => row.map((v, c) => v * grp.weights[j][c]))
	);
	/** 每条窗口的"槽位打分 + 槽位 ape"（softmax 的输入），堆叠成 [nE·window × hd] */
	const winScores: Mat = winStacked((grp) => grp.scores);
	/**
	 * 窗口槽位打分**每一行**所属的来源**行块**（扫描拼接动画用）：按 `ratio` 行一组拷贝——
	 * 每条窗口前 `ratio` 行（上一组 2 槽）来自 s_overlap+ape 的 2 行、后 `ratio` 行（自己组 2 槽）
	 * 来自 s+ape 的 2 行。`seq[r]` 是窗口槽位打分第 r 行所属块的来源
	 * （`target` = 源视图名、`from`/`to` = 源矩阵那一段行；空槽块为 `null`）。
	 *
	 * 所以扫描是**交替**的：先框 s_overlap+ape 的 2 行（上一组）、再框 s+ape 的 2 行（自己组），
	 * 三个矩阵（源两处 + 窗口槽位打分）的框都保持 2 行高。
	 */
	const winScanSeq: ({ target: string; from: number; to: number } | null)[] = trace.groups.flatMap(
		(grp) => {
			const tailT = grp.slots.slice(0, ratio).filter((t) => t >= 0);
			const ownT = grp.slots.slice(ratio).filter((t) => t >= 0);
			const tailBlock = tailT.length
				? { target: `${prefix}-sape-tail`, from: Math.min(...tailT), to: Math.max(...tailT) }
				: null;
			const ownBlock = ownT.length
				? { target: `${prefix}-sape-own`, from: Math.min(...ownT), to: Math.max(...ownT) }
				: null;
			// 每条窗口 `window` 行：前 `ratio` 行 = tailBlock、后 `ratio` 行 = ownBlock
			return Array.from({ length: window }, (_, j) => (j < ratio ? tailBlock : ownBlock));
		}
	);

	return [
		// ── 1. gate 投影 → 打分加 ape → softmax（一条线走完） ──
		{
			id: `${prefix}-compress-gate`,
			kind: 'PROJECT',
			diagram: flow(['x', 'gate', 'w']),
			label: `压缩的第一步：W_gate 把每个 token 投影出 ${gateCols} 个数（真实 ${gateRealCols} 个）：${layout}——${realLayout}。**框出的打分那几列**（${overlap ? 's_overlap 和 s 两段' : '打分'}）就是"这个 token 该占多大权重"的原料——权重从打分来，这一步先把它算出来；**打分加 ape、再 softmax 成 w** 在下一页全量做（重叠窗口下每个 token 有主体 / 尾巴两份打分）`,
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
						`W_gate${suffix}`,
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
							`[v | s]${suffix}`,
							trace.gate,
							gateShape,
							[S, gateRealCols],
							'latent',
							// 注解只留最短的一行——真实的分段尺寸由 `realParts` 走"真实尺寸"那一行，
							// 详细的分段说明在步骤文案里。胶囊是 nowrap + ellipsis 的，
							// 块宽被限到 9rem 之后长句子会被省略号截掉。
							overlap ? '← v_overlap | v | s_overlap | s' : '← 值 | 打分'
						),
						realParts
					}
				},
				// HCA（不重叠）的"打分 → 加 ape → softmax → w"仍在这一页（没有两份打分）：
				// 每个 token 只投影一份打分，加 ape 再 softmax 就得到 w。
				// CSA（重叠）把它挪到下一页（-sape，两份打分全量），所以这里不画。
				...(overlap
					? []
					: [
							{
								name: `${prefix}-gate-softmax`,
								kind: 'transform' as const,
								row: 1,
								group: 'chain',
								// 和左边的矩阵乘同属"放大占满宽度"的那一行（见上），格子要一致
								cellSize: 24,
								fillWidth: !overlap,
								shape: [S, hd],
								op: '+ s',
								input: ref(
									`ape${suffix}`,
									apeFull,
									[S, hd],
									[S, 512],
									'weight',
									`← 每个 token 拿到的那份按槽位偏置（表是真实 [${rw} × 512]，同一组里按槽位查表）；8 个 token 一起画`
								),
								output: ref(
									// 名字不带空格：窄块里 `s + ape [8 × 4]` 会折行，表头一高一低矩阵就错开
									`s+ape${suffix}`,
									sApeFull,
									[S, hd],
									[S, 512],
									undefined,
									`← 上面框出的打分那几列 + 每个 token 自己那份 ape（空槽是 ∅，数学上 ≈ −∞）`
								),
								then: {
									// 算子名保持最短（"逐通道"写在输出块的注解里）：mid 那一格的宽度
									// 也是这一行的宽度来源之一
									op: 'softmax',
									result: ref(
										`w${suffix}`,
										wFull,
										[S, hd],
										[S, 512],
										undefined,
										'← 在自己那一组那 4 个槽位里逐列归一化，所以每列加起来是 1'
									)
								}
							}
						]),
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

		// ── 1.5 打分 + ape → softmax → w（重叠窗口 CSA 专用）：逐 token 双份全量 ──
		...(overlap
			? [
					{
						id: `${prefix}-sape`,
						kind: 'SOFTMAX' as const,
						diagram: flow(['gate', 'w']),
						label: `打分加 ape → 拼接成窗口槽位打分 → softmax → w：重叠窗口下每个 token 投影出**两份打分**——主体份 s（在自己那组里用）和尾巴份 s_overlap（被下一组当"上一组尾巴"用），各加**各自槽位**的 ape。左边上、下两条链：s_overlap+ape 在上、s+ape 在下。**右边中间**的「窗口槽位打分」**从空开始**，两个框分别扫过 s_overlap+ape（每条窗口"上一组 2 槽"）和 s+ape（"自己 2 槽"）逐渐拼接填满（窗口重叠 → token 会重复出现）；填满后才在每条窗口那 4 个槽位上逐通道 softmax → w，供下一步加权平均用`,
						formula:
							's^{\\text{ov}}+a^{\\text{ov}},\\ s+a\\ \\Rightarrow\\ w_{j,c}=\\mathrm{softmax}_j(s_{j,c}+a_{j,c})',
						tensors: [
							// Row 0, 左列 group 'scores'：s_overlap + ape（上）、s + ape（下）——**并行**同时算
							// （相关计算：都汇向同一个窗口槽位打分，所以 `noHsep` 列间画箭头而不是虚线）
							{
								name: `${prefix}-sape-tail`,
								kind: 'transform' as const,
								row: 0,
								group: 'scores',
								parallel: true,
								noHsep: true,
								shape: [S, hd],
								op: '+ape',
								input: ref(
									`s_overlap${suffix}`,
									trace.sOverlap!,
									[S, hd],
									[S, R.head_dim],
									'latent',
									'← 尾巴份打分（被下一组当尾巴时用）'
								),
								output: ref(
									`s_overlap+ape${suffix}`,
									sApeTailFull!,
									[S, hd],
									[S, R.head_dim],
									undefined,
									`← 加尾巴槽位的 ape（表的前 ${ratio} 行）`
								)
							},
							{
								name: `${prefix}-sape-own`,
								kind: 'transform' as const,
								row: 0,
								group: 'scores',
								parallel: true,
								shape: [S, hd],
								op: '+ape',
								input: ref(
									`s${suffix}`,
									trace.s,
									[S, hd],
									[S, R.head_dim],
									'latent',
									'← 主体份打分（在自己组里用）'
								),
								output: ref(
									`s+ape${suffix}`,
									sApeFull,
									[S, hd],
									[S, R.head_dim],
									undefined,
									`← 加主体槽位的 ape（表的后 ${ratio} 行）`
								)
							},
							// Row 0, 右列：窗口槽位打分（matrix，**从空**逐列拼接填）→ softmax → w
							{
								name: `${prefix}-sape-w-in`,
								kind: 'matrix' as const,
								row: 0,
								group: 'win-in',
								vcenter: true,
								shape: [nE * window, hd],
								cellSize: 22,
								animated: true,
								revealOrder: 'row' as const,
								overview: `窗口槽位打分${suffix}`,
								title: `窗口槽位打分${suffix}`,
								scan: { seq: winScanSeq },
								data: winScores,
								label: `← **初始为空**：框交替扫过 s_overlap+ape 与 s+ape 的行，**逐行**拼接填出来（每条窗口 4 槽 = 上一组 2 槽用 s_overlap、自己 2 槽用 s）`
							},
							{
								name: `${prefix}-sape-w`,
								kind: 'transform' as const,
								row: 0,
								group: 'win-out',
								shape: [nE * window, hd],
								cellSize: 22,
								op: 'softmax',
								hideInput: true,
								input: ref(
									`窗口槽位打分${suffix}`,
									winScores,
									[nE * window, hd],
									[nE * rw, R.head_dim],
									'latent',
									'← 左边拼接填出的窗口槽位打分（初始为空）'
								),
								output: ref(
									`w${suffix}`,
									wWinStacked,
									[nE * window, hd],
									[nE * rw, R.head_dim],
									'weight',
									'← 每条窗口那 4 个槽位逐通道 softmax（每列和为 1）；供下一步 w ⊙ v 用'
								)
							},
							{
								name: `${prefix}-sape-splice-note`,
								kind: 'shape' as const,
								wide: true,
								shape: [S, hd],
								note: `**拼接动画**：窗口槽位打分**从空开始**，两个框分别扫过左边上下两条链的输出（s_overlap+ape 和 s+ape）的各槽位填进去——每条窗口 4 槽 = 上一组 2 槽（s_overlap+ape）+ 自己 2 槽（s+ape）；因为窗口重叠，同一个 token 会同时出现在两条窗口里、两处的槽位打分不同`
							}
						]
					}
				]
			: []),

		// ── 2. 组内加权平均 → 归一化 + RoPE（同一行、先后算） ──
		//    · HCA（不重叠）：权重 w 上一页算好，这一步把 v 与 w 整块拷过来 ewise；
		//    · CSA（重叠）：权重 w 在上一页（-sape）算好，这一步把 4 条窗口各 4 行堆成
		//      一块逐组揭示（只用 w，不再算它）。
		{
			id: `${prefix}-compress-pool`,
			kind: 'SOFTMAX' as const,
			diagram: flow(['gate', 'w', 'entries', 'ckv']),
			label: overlap
				? `组内加权平均（**权重 w 上一页已经算好**）：把 ${nE} 条窗口**一起画**（一条 4 行，窗口互相重叠：token 会同时出现在两条窗口里、权重不同）。**分两步**：先算 **w ⊙ v**——每条窗口的值逐元素乘上各自那一列的权重（框出的 4 行 = 正在算的那条窗口）；再**按列求和**得 v̄——每条窗口的 ${window} 个槽位加起来变成那 1 条压缩条目（先出完第 1 条、再第 2 条…）`
				: `组内加权平均：**权重 w 上一页已经算好了**（打分 + ape → softmax）。这一步把上一页那两块**整块拷过来**——${S} 个 token 的**值 v** 和**权重 w** 各一块。**分两步**：先算 **w ⊙ v**——每个 token 的值乘上它那一列的权重（${S} 个 token 分成 ${nE} 组、一组一组乘，框出的就是**正在算的那一组**）；再**按列求和**得 v̄——每组那 ${ratio} 个 token 加起来变成 1 条（先出第 1 条、再第 2 条）。**同一行右边紧接着做归一化 + RoPE**：v̄ 先过 RMSNorm 再按它所在组的最后一个 token 做 RoPE，就成了可用的${what}（先后计算，不并发）`,
			formula:
				'\\bar v_c=\\sum_j w_{j,c}\\,v_{j,c},\\qquad \\mathrm{kv}=\\mathcal{R}\\!\\left(\\mathrm{RMSNorm}(\\bar v),\\; pos_{group}\\right)',
			tensors: [
				// "乘过权重"要画成**矩阵**，不能只写一行小字说"已经乘过"。
				//
				// 不重叠（HCA）用 `ewise`：`w` 右上、`v` 左下、`w ⊙ v` 右下（和矩阵乘同一个 2×2）；
				// 两个操作数都从上一页的完整矩阵拷过来，**8 个 token 全画**、逐组（ratio 行一组）揭示。
				//
				// 重叠窗口（CSA）下不能用"逐 token 的 8 行"：一个 token 会同时出现在两条窗口里、
				// 权重不同，逐 token 行直接求和算不出条目。所以把 **4 条窗口**各 `window` 行上下
				// 堆成一块 `[nE·window × hd]`，同样用 `ewise` + `groupRows = window` 逐组揭示，
				// 每条窗口的 window 行按列加 = 那组自己的 1 条。权重 w 在上一页（-sape）算好，
				// 这里只用它（`w`），不再画 softmax。
				//
				// 这一页**不再标 `fillWidth`**：两个块并排（`compact`），
				// 格子就按并排的统一值 22 走（`matmulCellSize` / `matrix` 视图都是它）——
				// 一行里两块边长一致，行高才一致、矩阵才对齐。
				...(overlap
					? [
							// ── 4 条窗口的 w ⊙ v → 各按列求和 → v̄ → kv ──
							{
								name: `${prefix}-pool-ewise`,
								kind: 'ewise' as const,
								row: 1,
								cellSize: 22,
								shape: [nE * window, hd],
								// 总览节点：`w ⊙ v → v̄`（`ewise` 没有具名操作数，靠 `overview` 声明）
								overview: `w ⊙ v${suffix}`,
								op: '⊙',
								a: ref(
									`v${suffix}`,
									vWinStacked,
									[nE * window, hd],
									[nE * rw, R.head_dim],
									'latent',
									`← ${nE} 条窗口各自的值（**一条 4 行**，重叠 token 会在两条窗口里重复出现，权重不同）`
								),
								b: ref(
									`w${suffix}`,
									wWinStacked,
									[nE * window, hd],
									[nE * rw, R.head_dim],
									'weight',
									'← 每条窗口自己归一化出来的逐槽位权重（4 条窗口堆成一块）'
								),
								out: ref(
									`w ⊙ v${suffix}`,
									wvWinStacked,
									[nE * window, hd],
									[nE * rw, R.head_dim],
									undefined,
									`← 逐元素相乘；**框出的那 4 行 = 正在算的那条窗口**；下一步再把每列加起来 = 那 1 条 v̄`
								),
								// 逐组揭示：一条窗口一组（window 行），组内按列走
								groupRows: window,
								revealOrder: 'col' as const
							},
							{
								name: `${prefix}-pool-result`,
								kind: 'matrix' as const,
								row: 1,
								cellSize: 26,
								shape: [nE, hd],
								animated: true,
								// 分开算：w ⊙ v 整块算完，v̄ 才一行一组地出（`groupRows = window`：一条窗口 = 1 行）
								groupRows: window,
								overview: `v̄${suffix}`,
								title: `v̄${suffix}`,
								data: trace.entries,
								label: `← ${nE} 条压缩条目：w ⊙ v 算完后，每条 = 它那 ${window} 个槽位按列求和（先出完第 1 条、再第 2 条…）`
							}
						]
					: [
							{
								name: `${prefix}-pool-ewise`,
								kind: 'ewise' as const,
								row: 1,
								cellSize: 26, // 与同一行的 v̄ / kv 链一致（c128 4-5-6-7 放大）
								shape: [S, hd],
								// 总览节点：`w ⊙ v → v̄`（`ewise` 没有具名操作数，靠 `overview` 声明）
								overview: `w ⊙ v${suffix}`,
								op: '⊙',
								a: ref(
									`v${suffix}`,
									trace.v,
									[S, hd],
									[S, R.head_dim],
									'latent',
									'← 上一页 `[v | s]` 的值那几列'
								),
								b: ref(
									`w${suffix}`,
									wFull,
									[S, hd],
									[S, R.head_dim],
									'weight',
									'← 上一页 softmax 出来的权重'
								),
								out: ref(
									`w ⊙ v${suffix}`,
									prodFull,
									[S, hd],
									[S, R.head_dim],
									undefined,
									`← 逐元素相乘（**框出的那一组**就是正在算的：${ratio} 个 token）；下一步按列加起来 = 那 1 条 v̄`
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
								cellSize: 26, // 与同一行的 w⊙v / kv 链一致（c128 4-5-6-7 放大）
								shape: [nE, hd],
								animated: true,
								// 分开算：w ⊙ v 整块算完，v̄ 才一行一组地出（`groupRows = ratio`），
								// 逐行按行序出现：先出完第 1 条，再出第 2 条
								groupRows: ratio,
								overview: `v̄${suffix}`,
								title: `v̄${suffix}`,
								data: trace.entries,
								label: `w ⊙ v 算完后，每 ${ratio} 个 token 压成 1 条（按列 Σ）`
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
						`v̄${suffix}`,
						trace.entries,
						[nE, hd],
						[nE, 512],
						undefined,
						`← 左边 pool 按列加出来的 ${nE} 条压缩条目（每条 = 组内 ${window} 个 token 的值的加权平均）`
					),
					output: ref(
						`v̄~${suffix}`,
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

/** 检索器的三步：准备 q^I → 打分（q·k → ReLU → 按头加权 → I）→ 挑 top-k */
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
	/** 每行挑出的 top-k 摊成 0/1（[S × nE]）——"谁进注意力"的选择矩阵；
	 *  下一页（注意力）掩码里的**压缩条目列**就是它，逐行揭示选中 / 落选 / 未出生 */
	const topkMask: Mat = Array.from({ length: S }, (_, i) =>
		Array.from({ length: nE }, (_, e) => (trace.topk[i].includes(e) ? 1 : 0))
	);
	/** 每头的 ReLU 分数（[nH][S × nE] = max(0, 原始 q·k)）——打分流程的第二步 */
	const headRelu = trace.headRaw.map((h) => h.map((row) => row.map((v) => Math.max(0, v))));
	/** 每头 ReLU 分数 × 该头权重（逐行乘标量）——乘法动画输出与 Σ 项都用它 */
	const hwProd = (h: number) =>
		headRelu[h].map((row, i) => row.map((v) => v * (trace.headWeights[i][h] ?? 0)));
	/** w_0 / w_1 画成 [S × nE]：每头一个标量按行广播到 4 个条目列（权重按头、条目共用） */
	const wBroadcast = (h: number) =>
		trace.headWeights.map((r) => Array.from({ length: nE }, () => r[h]));

	return [
		{
			id: `${prefix}-indexer-q`,
			kind: 'PROJECT',
			diagram: flow(['cqn', 'itop']),
			label: `这一页先把 indexer 的**查询 q^I** 准备好——从共用的 c_Q~ 升维（头更窄，每头 ${hd} 维，真实 ${R.index_head_dim} vs 注意力的 ${R.head_dim}，够分辨谁重要就行），按头过 RoPE。**k^I（压缩 KV）上一页已经压好**，下一页用它把打分流程完整走一遍：q^I·k^I → ReLU → 按头加权求和 → 打分 I → top-k`,
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
						`← ${nH} 头 × 每头 ${hd} 维（真实 ${R.index_n_heads} × ${R.index_head_dim}；注意力的 Q 是 ${cfg.num_heads} 头 × ${cfg.head_dim} 维——indexer 的头更窄），按头过 RoPE 后参与打分`
					)
				},
				{
					name: `${prefix}-indexer-note`,
					kind: 'shape',
					wide: true,
					shape: [nE, hd],
					note: `两套压缩 KV 各有各的 W_gate：注意力那套负责"被看"（宽 ${R.head_dim}），indexer 这套只负责"排序"（宽 ${R.index_head_dim}）。打分高的条目才会进入注意力，其余连 KV 都不必读——省的就是这部分。**下一步**用这份 q^I 对 k^I 完整打分`
				}
			]
		},
		{
			id: `${prefix}-indexer-score`,
			kind: 'MATMUL',
			diagram: flow(['cqn', 'itop']),
			label: `打分第一步，先看**单个头**怎么算分：head 0 的 q^I_0 对每条压缩条目算点积 q^I_0 · k^Iᵀ（左边这块矩阵乘，和注意力 S = Q·Kᵀ 同套路）→ **头0分数**；紧挨着过 **ReLU**（右边：负数截成 0，负的相似度不算数）→ **头0 ReLU 分数**。head 1 同样算一份；**两头的 ReLU 分数怎么按 w_h 加权合成打分矩阵 I、再挑 top-k，在下面两页**`,
			formula: 'r_{i,e}=\\mathrm{ReLU}\\!\\left(q^{I}_{0,i}\\cdot k^{I}_{e}\\right)\\ \\text{（单头演示）}',
			tensors: [
				// ① 单头打分：q^I_0 · k^Iᵀ → 头 0 原始分数（和注意力 S = Q·Kᵀ 同套路），
				// 右边紧跟 ReLU 链（`hideInput`：头0分数 由矩阵乘的 C 画，不重画）
				{
					name: `${prefix}-indexer-score-mm`,
					kind: 'matmul',
					row: 1,
					group: 'mm',
					cellSize: 40,
					shape: [S, nE],
					a: ref(
						'q^I_0',
						trace.qFlat.map((r) => r.slice(0, hd)),
						[S, hd],
						[S, R.index_head_dim],
						'head',
						`← head 0 的 q（每头 ${hd} 维，真实 ${R.index_head_dim}）；这里演示 head 0，head 1 同样算一份`
					),
					b: ref(
						'k^Iᵀ',
						transpose(ic.entriesReady),
						[hd, nE],
						[R.index_head_dim, nE],
						'head',
						`← indexer 那套压缩 KV 转置（**第 {{step:${prefix}-idx-compress-pool}} 步已经压好**，每列一条压缩条目）`
					),
					out: ref(
						'头0分数',
						trace.headRaw[0],
						[S, nE],
						[S, nE],
						'result',
						`← head 0 的 q·k 原始分数。**∅ = 那条条目还没写出来**，token 0 时一条都还没写完、第一行整行是 ∅；之后每过一个块尾，右侧多亮一格。`
					),
					scaleNote: '每头 q·k（无缩放）'
				},
				// ② 单头 ReLU：头0分数 → ReLU → 头0 ReLU 分数（负数截成 0；输入由左边矩阵乘画）
				{
					name: `${prefix}-indexer-relu`,
					kind: 'transform',
					row: 1,
					group: 'relu',
					cellSize: 40,
					shape: [S, nE],
					op: 'ReLU',
					hideInput: true,
					input: ref(
						'头0分数',
						trace.headRaw[0],
						[S, nE],
						[S, nE],
						'result',
						'← 左边矩阵乘算出的 头0分数（负数会被截成 0）'
					),
					output: ref(
						'头0 ReLU 分数',
						headRelu[0],
						[S, nE],
						[S, nE],
						'result',
						'← 负数截成 0（和左边一对比就看得出）；head 1 也这样 ReLU，在下一页 Σ 里一起加权'
					)
				}
			]
		},
		{
			id: `${prefix}-indexer-w`,
			kind: 'MATMUL',
			diagram: flow(['itop']),
			label: `打分第二步：**头权重 w_h 从哪来**——每个 query 的隐藏状态 x 过一个小的投影 W^w（d_model → ${nH} 头，每头一个标量），得到**原始投影 w_raw**（左边矩阵乘）；紧挨着**乘 1/√(hd·nH) 缩放**（右边，和注意力 softmax_scale 同一个道理）→ **w_h**。然后**每头的 ReLU 分数乘上它自己的权重**：head 0 乘 w_0、head 1 乘 w_1（下面两块**同时算**——一个 token 的两头互不相干），得到 头0·w0 和 头1·w1`,
			formula:
				'w_{i,h}=\\mathrm{scale}\\,\\left(x_i W^{w}\\right)_h,\\qquad \\hat r^{(h)}_{i,e}=w_{i,h}\\,\\mathrm{ReLU}\\!(q^{I}_{i,h}\\cdot k^{I}_{e})',
			tensors: [
				// ① 原始投影：x · W^w → w_raw [S × nH]（还没缩放）
				{
					name: `${prefix}-indexer-w-mm`,
					kind: 'matmul',
					row: 1,
					group: 'wmm',
					cellSize: 34,
					shape: [S, nH],
					a: ref('x', ic.x, [S, D], [S, R.d_model], 'input', '← 每个 query 的隐藏状态（和 Q/KV 同一份 x）'),
					b: ref(
						'W^w',
						w.Ww,
						[D, nH],
						[R.d_model, R.index_n_heads],
						'weight',
						`← 头权重的投影（真实 ${R.d_model} → ${R.index_n_heads} 头，每头一个标量）`
					),
					out: ref(
						'w_raw',
						trace.wRaw,
						[S, nH],
						[S, R.index_n_heads],
						'weight',
						`← 原始投影（还没缩放）；右边乘 1/√(hd·nH) 才是 w_h`
					)
				},
				// ② 缩放：× 1/√(hd·nH) → w_h（输入由左边矩阵乘的 C 画，不重画）
				{
					name: `${prefix}-indexer-w-scale`,
					kind: 'transform',
					row: 1,
					group: 'wscale',
					cellSize: 34,
					shape: [S, nH],
					op: '× 1/√(hd·nH)',
					hideInput: true,
					input: ref(
						'w_raw',
						trace.wRaw,
						[S, nH],
						[S, R.index_n_heads],
						'weight',
						'← 左边矩阵乘算出的 原始投影'
					),
					output: ref(
						'w_h',
						trace.headWeights,
						[S, nH],
						[S, R.index_n_heads],
						'weight',
						`← 缩放后：每头一个权重、每个 query 一行（w_0、w_1）；供下面两个乘法用`
					)
				},
				// ③ 每头 ReLU 分数 × 自己的权重（**并列**：parallel + 同一行 → 同一个动画单元同时算，
				//     列间画浅色虚线而不是箭头——两个乘法互不相干，只是各自独立地算）。
				//     ewise：w_0 / w_1 按行广播成 [S × nE] 的可见权重矩阵
				{
					name: `${prefix}-indexer-mul-0`,
					kind: 'ewise',
					row: 2,
					group: 'mul0',
					parallel: true,
					cellSize: 34,
					shape: [S, nE],
					op: '⊙',
					a: ref(
						'头0 ReLU 分数',
						headRelu[0],
						[S, nE],
						[S, nE],
						'result',
						`← 第 {{step:${prefix}-indexer-score}} 步的 头0 ReLU 分数`
					),
					b: ref(
						'w_0',
						wBroadcast(0),
						[S, nE],
						[S, nE],
						'weight',
						`← w_h 的第 0 列（head 0 的权重）按行广播成 ${nE} 格——权重按头，${nE} 个条目共用同一个 w_0`
					),
					out: ref(
						'头0·w0',
						hwProd(0),
						[S, nE],
						[S, nE],
						'result',
						'← 逐格相乘：每个条目的 ReLU 分数 × 这一行的 w_0'
					)
				},
				{
					name: `${prefix}-indexer-mul-1`,
					kind: 'ewise',
					row: 2,
					group: 'mul1',
					parallel: true,
					cellSize: 34,
					shape: [S, nE],
					op: '⊙',
					a: ref(
						'头1 ReLU 分数',
						headRelu[1],
						[S, nE],
						[S, nE],
						'result',
						`← 第 {{step:${prefix}-indexer-score}} 步的 头1 ReLU 分数`
					),
					b: ref(
						'w_1',
						wBroadcast(1),
						[S, nE],
						[S, nE],
						'weight',
						`← w_h 的第 1 列（head 1 的权重）按行广播成 ${nE} 格`
					),
					out: ref(
						'头1·w1',
						hwProd(1),
						[S, nE],
						[S, nE],
						'result',
						'← 逐格相乘：每个条目的 ReLU 分数 × 这一行的 w_1'
					)
				}
			]
		},
		{
			id: `${prefix}-indexer-logits`,
			kind: 'ADD',
			diagram: flow(['itop']),
			label: `打分第三步：把两头的乘积**加起来**得到打分矩阵 I——Σ 那块逐行把两项相加：头0·w0 + 头1·w1 = 这一格的打分（负的原始分数早在 ReLU 里变 0，不会拖累总分）；I 算完**紧接着挑 top-k**（旁边那一列，逐行揭示）：每个 query 从自己那一行挑分数最高的 top-${k} 条（真实 ${R.index_topk} 条；还没写出来的那条是 ∅，自然选不上）——选中的才进入下一步的注意力，其余连 KV 都不必读。演示最后一行（token ${i}）：选中的是条目 ${picked.join('、')}；"远距离历史"从"全看"变成"检索着看"，这是 CSA 比 HCA 更省的地方`,
			formula:
				'I_{i,e}=\\hat r^{(0)}_{i,e}+\\hat r^{(1)}_{i,e},\\qquad \\mathcal{S}_i=\\mathrm{top\\text{-}}k\\left(\\{I_{i,e}\\}_{e\\,:\\;born(e)\\,\\le\\,i}\\right)',
			tensors: [
				// ① 求和：头0·w0 + 头1·w1 → 打分矩阵 I（逐行动画）
				// 与下面的 top-k 选择**同一行并排**（`row: 1`）：算完 I 紧接着挑 top-k，
				// 中间用 `→` 连起来——"从打分到选择"是一条连续的链
				{
					name: `${prefix}-indexer-logits-sum`,
					kind: 'sum',
					row: 1,
					shape: [S, nE],
					terms: [
						{
							name: '头0·w0',
							shape: [S, nE],
							realShape: [S, nE],
							data: hwProd(0),
							label: `← 第 {{step:${prefix}-indexer-w}} 步的 头0·w0`
						},
						{
							name: '头1·w1',
							shape: [S, nE],
							realShape: [S, nE],
							data: hwProd(1),
							label: `← 第 {{step:${prefix}-indexer-w}} 步的 头1·w1`
						}
					],
					result: {
						name: 'I',
						shape: [S, nE],
						realShape: [S, nE],
						data: trace.logits,
						label: `打分矩阵 I：逐行相加（头0·w0 + 头1·w1 = 这一格的打分；∅ = 那条条目还没写出来）`
					}
				},
				// ② 挑 top-k：把**每行**的选择摊成 0/1（选中 = 保留、其余 ∅），逐行揭示——
				//    这就是下一页注意力掩码里"压缩条目列"的直接来源（每行从 I 里挑的 top-k）
				{
					name: `${prefix}-indexer-topk-mask`,
					kind: 'mask',
					row: 1,
					shape: [S, nE],
					overview: 'top-k',
					scores: trace.logits,
					mask: topkMask,
					matrixName: 'top-k 选择',
					label: `每个 query 从 I 自己的那一行挑分数最高的 top-${k} 条（真实 ${R.index_topk} 条）：**选中才保留（显示分数）、其余置 ∅**（还没写出来的那条是 ∅，自然选不上）。逐行揭示；最后一行（token ${i}）选中的是条目 ${picked.join('、')}。**这张 0/1 矩阵就是下一步注意力掩码里"压缩条目列"的直接来源**`
				}
			]
		}
	];
}
