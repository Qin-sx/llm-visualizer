/**
 * DeepSeek V4.1 的 **CSA2（Compressed Sparse Attention 2）**——`csa2-full` / `csa2-reindex` /
 * `csa2-reuse` 三个插件的共享核心。
 *
 * 与 V4 的 hybrid 同构的部分（Q/KV 投影、融合注意力、逆 RoPE、分组低秩输出）直接复用
 * `hybrid.ts`（`hybridHeadSteps` / `fusedTail` / `hybridTailSteps`），本文件只实现 V4.1 的三处增量：
 *
 *   ① **层间共享**：只有 kv_source 层压缩，消费者（Reuse / Reindex）通过 `ctx.pool(src)`
 *      读共享池——不重算、不重选（Reindex 只换自己的 q^I 重打分）。
 *   ② **压缩器换血**：ratio 只有 1 / 2；**无重叠窗口、无 ape**；ratio-1 干脆只有 `wkv`（无 gate）。
 *   ③ **indexer K 从 main KV 投影**：`k^I = rope(norm(wk(压缩条目)))`，取代 V4 里
 *      "indexer 自己的压缩路径"；候选源层还额外建**候选池**（块级 max 选块），
 *      Reindex 层只在池内打分。
 *
 * 另外 `q_head_norm=false` 与 Single-Pass mHC 都在 `hybrid.ts` / forward 骨架里处理，
 * 本文件不重复。
 */
import { matmul, randMat, rmsNorm, ropeApply, softmaxRow, topK, transpose, type Mat, type Vec } from '$lib/core/mat';
import type { KvPool, LayerCtx, MatRef, MatrixBand, ModelConfigLike, Step } from '$lib/core/types';
import { REAL_V4_1 } from '$lib/model/config';
import {
	fusedTail,
	hybridDims,
	hybridHeadSteps,
	hybridTailSteps,
	type HybridPlan,
	type HybridWeights
} from './hybrid';
import type { CompressorTrace } from './compressor';
import { v4Diagram, type V4Variant } from './v4-diagram';

const NEG_INF = Number.NEGATIVE_INFINITY;
const R = REAL_V4_1;

function dot(a: Vec, b: Vec): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[i];
	return s;
}

/** 逐行 RMSNorm（压缩条目是单个向量，不是矩阵） */
function rmsNormRow(row: Vec): Vec {
	return rmsNorm([row])[0];
}

/** 一个张量引用（带真实尺寸，界面据此标注缩小倍数） */
function ref(
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
// 角色（Full / Reindex / Reuse）与计划
// ══════════════════════════════════════════════════════════
export type Csa2Role = 'full' | 'reindex' | 'reuse';

export interface Csa2Plan {
	role: Csa2Role;
	/** 压缩比（UI 第 3-5 层是 2、第 6-8 层是 1） */
	ratio: number;
	/** 本层是 kv_source：自己压缩、写池 */
	isKvSource: boolean;
	/** 本层是 index_source：自己跑 indexer（Full 扫全量 / Reindex 池内重打分） */
	isIndexSource: boolean;
	/** 本层是 candidate_source：扫全量 + 按块选块建候选池 */
	isCandidateSource: boolean;
	/** 压缩 KV 从哪个 kv_source 的池读（自己写时 = 自己） */
	kvSrc: number;
	/** top-k 复用哪个 index_source 的（Reuse 用；自己跑时 = 自己） */
	idxSrc: number;
	/** 演示 top-k 数（真实 512） */
	indexTopk: number;
}

/** 同 ratio 组内"最近的 index_source"（真 CSA2 的 Reuse 复用最近一次 Full/Reindex 的 top-k） */
function nearestSource(sources: number[], layer: number): number {
	const rev = [...sources].reverse();
	return rev.find((s) => s <= layer) ?? -1;
}

export function csa2PlanOf(cfg: ModelConfigLike, layer: number): Csa2Plan {
	const ratio = cfg.compress_ratios?.[layer] ?? 0;
	const kvSources = cfg.kv_source_layer_ids ?? [];
	const indexSources = cfg.index_source_layer_ids ?? [];
	const isKvSource = kvSources.includes(layer);
	const isIndexSource = indexSources.includes(layer);
	const isCandidateSource = layer === cfg.candidate_source_layer_id;
	const role: Csa2Role = isKvSource ? 'full' : isIndexSource ? 'reindex' : 'reuse';
	return {
		role,
		ratio,
		isKvSource,
		isIndexSource,
		isCandidateSource,
		kvSrc: isKvSource ? layer : nearestSource(kvSources, layer),
		idxSrc: isIndexSource ? layer : nearestSource(indexSources, layer),
		indexTopk: cfg.index_topk ?? 1
	};
}

/** 条目 e 在哪个位置才算得出来（ratio-1：每个位置一条；ratio-2：一组 2 个 token 一条） */
export function bornAtOf(ratio: number, e: number): number {
	return ratio <= 1 ? e : ratio * (e + 1) - 1;
}

/** 步骤 id 前缀（每个检视层一条流程，前缀只在本层内保证唯一） */
export function csa2Prefix(role: Csa2Role): string {
	return role === 'full' ? 'c2f' : role === 'reindex' ? 'c2i' : 'c2u';
}

// ══════════════════════════════════════════════════════════
// 权重
// ══════════════════════════════════════════════════════════
export interface Csa2Weights {
	/** 与 V4 相同的注意力/输出权重（Q 上/下投影、SWA KV、sink、分组输出） */
	base: HybridWeights;
	/** ratio-1：`wkv` 直投（每个 token 的 latent 就是一条全局 KV）；ratio-2 时也可不用（见 wkvGate） */
	wkv: Mat;
	/** ratio-2：融合的 `wkv_gate` [D × 2·hd]，一次投影出 [v | s]（前 hd 列值、后 hd 列打分，无 ape） */
	wkvGate: Mat | null;
	/** indexer：自己的 Q 上投影 + 头权重投影 + 从 main KV 投影 k^I */
	WqbI: Mat;
	Ww: Mat;
	wk: Mat;
}

export function makeCsa2Weights(
	rnd: () => number,
	cfg: ModelConfigLike,
	plan: Csa2Plan
): Csa2Weights {
	const D = cfg.d_model;
	const hd = cfg.head_dim;
	const nH = cfg.index_n_heads ?? 1;
	const idxHd = cfg.index_head_dim ?? hd;
	const G = cfg.o_groups ?? 1;
	const oLora = cfg.o_lora_rank ?? hd;
	// 复用 hybrid 的 Q/KV/输出权重（同形状、同随机流——V4.1 与 V4 的这些部件完全一致）
	const base: HybridWeights = {
		Wqa: randMat(D, cfg.q_lora_rank ?? D, rnd),
		Wqb: randMat(cfg.q_lora_rank ?? D, cfg.num_heads * hd, rnd),
		Wkv: randMat(D, hd, rnd),
		Woa: Array.from({ length: G }, () => randMat((cfg.num_heads / G) * hd, oLora, rnd)),
		Wob: randMat(G * oLora, D, rnd),
		sink: randMat(cfg.num_heads, 1, rnd).map((r) => r[0]),
		comp: null,
		idx: null
	};
	return {
		base,
		// ratio-1 直投的 wkv；ratio-2 用融合的 wkv_gate（下面），wkv 也备一份（compute 里二选一）
		wkv: randMat(D, hd, rnd),
		// ratio-2 的 gate 一次投影出 [v | s] 两段（各 hd 列）——宽 2·hd，与真实 `wkv_gate` 同构
		wkvGate: plan.ratio > 1 ? randMat(D, 2 * hd, rnd) : null,
		WqbI: randMat(cfg.q_lora_rank ?? D, nH * idxHd, rnd),
		Ww: randMat(D, nH, rnd),
		wk: randMat(hd, idxHd, rnd)
	};
}

// ══════════════════════════════════════════════════════════
// 压缩器（ratio 1 / 2，无重叠、无 ape）
// ══════════════════════════════════════════════════════════
/** 与 V4 CompressorTrace 同形的压缩 trace（pool 页与掩码共用） */
export interface Csa2CompressTrace extends CompressorTrace {
	/** `wkv(x)` 投影出的值（ratio-2 的被平均对象；ratio-1 直接当条目） */
	kv: Mat;
	/** `wgate(x)` 投影出的打分（softmax 输入，**无 ape**）；ratio-1 为 null */
	score: Mat | null;
}

/**
 * V4.1 压缩：ratio-2 两两一组（`pool = Σ softmax(score)·kv`，无重叠窗口、无 ape）；
 * ratio-1 不压（每个 token 的 latent 就是一条，wkv 直投）。
 * 返回的 `groups` 与 V4 同构（slots / bornAt / weights / pooled），pool 页与掩码共用。
 */
export function csa2Compress(
	x: Mat,
	wkv: Mat,
	wkvGate: Mat | null,
	ratio: number,
	ropeTheta: number,
	/** 压缩条目的宽度 = 注意力头维（不是 d_model） */
	headDim: number
): Csa2CompressTrace {
	const hd = headDim;
	const S = x.length;
	// ratio-2：融合的 wkv_gate 一次投影出 [v | s]（值前 hd 列、打分后 hd 列）；
	// ratio-1：wkv 直投，没有打分段
	const gateRaw = wkvGate ? matmul(x, wkvGate) : null;
	const kv = gateRaw ? gateRaw.map((r) => r.slice(0, hd)) : matmul(x, wkv);
	const score = gateRaw ? gateRaw.map((r) => r.slice(hd)) : null;
	const gate: Mat = gateRaw ?? kv;

	const nEntries = Math.max(1, Math.floor(S / ratio));
	const groups: Csa2CompressTrace['groups'] = [];
	const entries: Mat = [];
	for (let g = 0; g < nEntries; g++) {
		const from = ratio * g;
		const to = Math.min(S, from + ratio);
		const slots = Array.from({ length: to - from }, (_, j) => from + j);
		const bornAt = Math.min(S - 1, ratio * (g + 1) - 1);
		// 逐通道 softmax（无 ape）：w[j][c] = softmax_j(score[j][c])
		const weights: Mat = Array.from({ length: slots.length }, () => Array(hd).fill(0));
		if (score) {
			const sc = slots.map((t) => score[t]);
			for (let c = 0; c < hd; c++) {
				const col = softmaxRow(sc.map((row) => row[c]));
				for (let j = 0; j < slots.length; j++) weights[j][c] = col[j];
			}
		} else {
			// ratio-1：单槽，权重恒 1（不平均，直通）
			weights[0] = Array(hd).fill(1);
		}
		const values = slots.map((t) => kv[t]);
		const pooled: Vec = Array.from({ length: hd }, (_, c) =>
			slots.reduce((acc, t, j) => acc + weights[j][c] * kv[t][c], 0)
		);
		groups.push({
			index: g,
			slots,
			from,
			to,
			bornAt,
			scores: slots.map((t) => (score ? score[t] : Array(hd).fill(0))),
			weights,
			mask: slots.map(() => Array(hd).fill(1)),
			values,
			pooled
		});
		entries.push(pooled);
	}

	const entriesNorm = entries.map(rmsNormRow);
	const entriesReady = entriesNorm.map((row, g) => ropeApply(row, groups[g].bornAt, ropeTheta));

	return {
		plan: { ratio, overlap: false, headDim: hd, ropeTheta },
		x,
		gate,
		v: kv,
		s: score ?? [],
		sOverlap: null,
		entries,
		entriesNorm,
		entriesReady,
		groups,
		kv,
		score
	} as Csa2CompressTrace;
}

// ══════════════════════════════════════════════════════════
// indexer（k^I 从 main KV 投影；可限定候选池）
// ══════════════════════════════════════════════════════════
export interface Csa2IndexerTrace {
	/** 打分对象 k^I [nE × idx_head_dim]（Full 投影 / Reindex 复用池里的） */
	kEntries: Mat;
	/** Q 的低秩潜向量（和注意力共用同一份 c_Q~） */
	qLatent: Mat;
	/** [S × nH·idxHeadDim] RoPE 之后 */
	qFlat: Mat;
	/** [S][nH][idxHeadDim] */
	q: number[][][];
	/** 头权重原始投影（还没缩放） */
	wRaw: Mat;
	/** 头权重（已含 1/√(idxHd·nH) 缩放） */
	headWeights: Mat;
	/** [nH][S × nE] 每头原始 q·k（打分动画用） */
	headRaw: number[][][];
	/** [S × nE] 打分 = Σ_h w_h · ReLU(·)；未出生条目 ∅ */
	logits: Mat;
	/** 每个 query 可见的条目下标 */
	candidates: number[][];
	/** 每个 query 选中的条目下标 */
	topk: number[][];
	/** 条目总数（打分矩阵宽度） */
	nE: number;
	/** 打分只看这些条目（Reindex 的候选池；Full 扫全量为 null） */
	scope?: number[];
}

/** 从 main KV 投影 k^I：wk → RMSNorm → RoPE（V4.1 的新来源） */
export function csa2ProjectIndexerK(entriesNorm: Mat, wk: Mat, ropeTheta: number, ratio: number): Mat {
	const kRaw = matmul(entriesNorm, wk);
	const kNorm = kRaw.map(rmsNormRow);
	return kNorm.map((row, e) => ropeApply(row, bornAtOf(ratio, e), ropeTheta));
}

export function csa2RunIndexer(
	x: Mat,
	qLatent: Mat,
	kEntries: Mat,
	w: Pick<Csa2Weights, 'WqbI' | 'Ww'>,
	ropeTheta: number,
	cfg: ModelConfigLike,
	/** 打分只在这些条目里做（Reindex 的候选池；Full 层为 null = 扫全量） */
	scope: number[] | null,
	/** 压缩比（条目 e 的出生位置 = bornAtOf(ratio, e)） */
	ratio: number
): Csa2IndexerTrace {
	const nH = cfg.index_n_heads ?? 1;
	const idxHd = cfg.index_head_dim ?? cfg.head_dim;
	const k = cfg.index_topk ?? 1;
	const S = x.length;
	const nE = kEntries.length;

	// q^I：用同一个 Q 潜向量升维（比注意力的 Q 窄），按头过 RoPE
	const qFlatRaw = matmul(qLatent, w.WqbI);
	const q = Array.from({ length: S }, (_, i) =>
		Array.from({ length: nH }, (_, h) => ropeApply(qFlatRaw[i].slice(h * idxHd, (h + 1) * idxHd), i, ropeTheta))
	);
	const qFlat = q.map((heads) => heads.flat());

	// 头权重：w_raw = x·W^w → × 1/√(idxHd·nH)（与 V4 相同）
	const scale = Math.pow(idxHd, -0.5) * Math.pow(nH, -0.5);
	const wRaw = matmul(x, w.Ww);
	const headWeights = wRaw.map((row) => row.map((v) => v * scale));

	// 打分：每头原始 q·k → Σ_h w_h · ReLU(·)；scope 之外的条目与未出生条目 ∅
	const headRaw: number[][][] = Array.from({ length: nH }, (_, h) =>
		q.map((heads, i) =>
			kEntries.map((kv, e) => (bornAtOf(ratio, e) > i ? NEG_INF : dot(heads[h], kv)))
		)
	);
	const logits = q.map((heads, i) =>
		kEntries.map((kv, e) => {
			if (scope && !scope.includes(e)) return NEG_INF;
			if (bornAtOf(ratio, e) > i) return NEG_INF;
			let acc = 0;
			for (let h = 0; h < nH; h++) acc += headWeights[i][h] * Math.max(0, headRaw[h][i][e]);
			return acc;
		})
	);

	// top-k：只在可见条目里挑
	const candidates = logits.map((row) =>
		row.map((v, e) => ({ v, e })).filter((o) => Number.isFinite(o.v)).map((o) => o.e)
	);
	const topk = logits.map((row) =>
		topK(row, Math.min(k, row.length)).filter((e) => Number.isFinite(row[e]))
	);

	return { kEntries, qLatent, qFlat, q, wRaw, headWeights, headRaw, logits, candidates, topk, nE, scope: scope ?? undefined };
}

/** 候选池：块级 max 选块（真实 2048 块 × 8 条 = 16,384 候选；演示 2 块 × 2 条 = 4） */
export function csa2BuildCandidates(
	logits: Mat,
	topkBlocks: number,
	blockSize: number
): number[] {
	const nE = logits[0].length;
	const nBlocks = Math.ceil(nE / blockSize);
	const blockMax: { b: number; v: number }[] = Array.from({ length: nBlocks }, (_, b) => {
		let mx = NEG_INF;
		for (let e = b * blockSize; e < Math.min(nE, (b + 1) * blockSize); e++) {
			for (let i = 0; i < logits.length; i++) {
				if (logits[i][e] > mx) mx = logits[i][e];
			}
		}
		return { b, v: mx };
	});
	const picked = [...blockMax].sort((a, b) => b.v - a.v).slice(0, topkBlocks);
	const cand: number[] = [];
	for (const { b } of picked) {
		for (let e = b * blockSize; e < Math.min(nE, (b + 1) * blockSize); e++) cand.push(e);
	}
	return cand;
}

// ══════════════════════════════════════════════════════════
// compute（按角色）
// ══════════════════════════════════════════════════════════
/** csa2 的 trace：HybridTrace 的超集（hybridHeadSteps / hybridTailSteps 共用） */
export interface Csa2Trace {
	out: Mat;
	x: Mat;
	cQ: Mat;
	cQNorm: Mat;
	qUp: Mat;
	qFlat: Mat;
	qPeRaw: number[][][];
	qPe: number[][][];
	q: number[][][];
	kvRaw: Mat;
	kvNorm: Mat;
	kv: Mat;
	compress: Csa2CompressTrace | null;
	indexer: Csa2IndexerTrace | null;
	compKv: Mat | null;
	kAll: Mat;
	scoreReal: Mat;
	vAll: Mat;
	sink: Vec;
	cols: number;
	nE: number;
	sinkCol: number;
	scores: number[][][];
	mask: Mat;
	maskedScores: number[][][];
	probs: number[][][];
	headOut: number[][][];
	headOutInv: number[][][];
	groups: Mat[];
	groupLatent: Mat[];
	oLatent: Mat;
	/** 消费者读的池（页面标注"输入来自第 N 层"） */
	pool?: KvPool;
}

/** 按头切开（q_head_norm=true 时每头先 RMSNorm；false 时直通） */
function splitHeadsNorm(m: Mat, H: number, hd: number, norm: boolean): Mat[] {
	const out: Mat[] = [];
	for (let h = 0; h < H; h++) {
		const head = m.map((r) => r.slice(h * hd, (h + 1) * hd));
		out.push(norm ? rmsNorm(head) : head);
	}
	return out;
}

/** 对每行末尾 dr 维做 RoPE（与 hybrid 的 ropeTail 相同口径） */
function ropeTailLocal(rows: Mat, dr: number, theta: number): Mat {
	if (dr <= 0) return rows;
	return rows.map((row, i) => [
		...row.slice(0, row.length - dr),
		...ropeApply(row.slice(row.length - dr), i, theta)
	]);
}

export function csa2Compute(x: Mat, ctx: LayerCtx): Csa2Trace {
	const cfg = ctx.cfg;
	const plan = csa2PlanOf(cfg, ctx.layer);
	const w = ctx.w as Csa2Weights;
	const { H, hd, dn, dr, G, oLora, theta, perGroup } = hybridDims(cfg, { ratio: plan.ratio, overlap: false, indexer: plan.isIndexSource });
	const S = x.length;

	// ── Q / SWA KV（与 V4 同路径；q_head_norm 关掉时只做 RoPE）──
	const cQ = matmul(x, w.base.Wqa);
	const cQNorm = rmsNorm(cQ);
	const qUp = matmul(cQNorm, w.base.Wqb);
	const qHeads = splitHeadsNorm(qUp, H, hd, cfg.q_head_norm !== false);
	const qNope = qHeads.map((h) => h.map((r) => r.slice(0, dn)));
	const qPeRaw = qHeads.map((h) => h.map((r) => r.slice(dn)));
	const qPe = qPeRaw.map((h) => h.map((r, i) => ropeApply(r, i, theta)));
	const q = qNope.map((h, hi) => h.map((r, i) => [...r, ...qPe[hi][i]]));
	// qFlat：[S × H·hd] 按行交错（mergeHeads 语义）
	const qFlat: Mat = Array.from({ length: S }, (_, i) => qHeads.flatMap((h) => h[i]));
	const kvRaw = matmul(x, w.base.Wkv);
	const kvNorm = rmsNorm(kvRaw);
	const kv = ropeTailLocal(kvNorm, dr, theta);

	// ── 压缩 / 共享池 ──
	const poolSide = ctx.pool ?? (() => undefined);
	const writePool = ctx.writePool ?? (() => undefined);
	let compress: Csa2CompressTrace | null = null;
	let indexer: Csa2IndexerTrace | null = null;
	let compKv: Mat | null = null;
	let topk: number[][] | null = null;

	if (plan.isKvSource) {
		// Full 层自己压缩（ratio-2：gate+pool；ratio-1：wkv 直投）——写进共享池
		compress = csa2Compress(x, w.wkv, w.wkvGate, plan.ratio, cfg.compress_rope_theta ?? theta, hd);
		compKv = compress.entriesReady;
		const pool: KvPool = { src: ctx.layer, ratio: plan.ratio, mainKv: compress.entriesReady };
		if (plan.isIndexSource) {
			const kEntries = csa2ProjectIndexerK(compress.entriesNorm, w.wk, cfg.compress_rope_theta ?? theta, plan.ratio);
			indexer = csa2RunIndexer(x, cQNorm, kEntries, w, cfg.compress_rope_theta ?? theta, cfg, null, plan.ratio);
			topk = indexer.topk;
			pool.indexerK = kEntries;
			pool.topK = topk;
		}
		if (plan.isCandidateSource) {
			pool.candidates = csa2BuildCandidates(indexer?.logits ?? [], cfg.candidate_topk_blocks ?? 1, cfg.candidate_block_size ?? 1);
		}
		writePool(pool);
	} else if (plan.role === 'reuse') {
		// Reuse：不压缩、不跑 indexer——直接读 kv_source 的池与最近的 top-k
		const p = poolSide(plan.kvSrc);
		compKv = p?.mainKv ?? null;
		topk = poolSide(plan.idxSrc)?.topK ?? null;
	} else {
		// Reindex：压缩 KV 与 indexer K 都读池，但用自己的 q^I 在候选池里重打分、写自己的 top-k
		const p = poolSide(plan.kvSrc);
		compKv = p?.mainKv ?? null;
		const scope = p?.candidates ?? null;
		indexer = csa2RunIndexer(x, cQNorm, p?.indexerK ?? [], w, cfg.compress_rope_theta ?? theta, cfg, scope, plan.ratio);
		topk = indexer.topk;
		if (compKv) writePool({ src: ctx.layer, ratio: plan.ratio, mainKv: compKv, topK: topk });
	}

	// ── 一次注意力：滑窗 token + 压缩条目 + sink（掩码的压缩列 = topk）──
	const nE = compKv ? compKv.length : 0;
	const tokVisible = (i: number, j: number) => j <= i && i - j < (cfg.swa_window ?? S);
	const bornAt = (e: number) => bornAtOf(plan.ratio, e);
	const picked = (i: number, e: number) => bornAt(e) <= i && (topk ? topk[i].includes(e) : true);
	const kAll: Mat = Array.from({ length: S + nE }, (_, j) => (j < S ? kv[j] : compKv![j - S]));
	const mask: Mat = Array.from({ length: S }, (_, i) => [
		...Array.from({ length: S }, (_, j) => (tokVisible(i, j) ? 1 : 0)),
		...Array.from({ length: nE }, (_, e) => (picked(i, e) ? 1 : 0)),
		1
	]);

	const tail = fusedTail(q, kv, compKv, w.base.sink, mask, w.base, hd, dn, theta, G, perGroup, oLora, H);

	return {
		out: tail.out,
		x,
		cQ,
		cQNorm,
		qUp,
		qFlat,
		qPeRaw,
		qPe,
		q,
		kvRaw,
		kvNorm,
		kv,
		compress,
		indexer,
		compKv,
		kAll,
		scoreReal: tail.scoreReal,
		vAll: tail.vAll,
		sink: w.base.sink,
		cols: tail.cols,
		nE,
		sinkCol: tail.sinkCol,
		scores: tail.scores,
		mask,
		maskedScores: tail.maskedScores,
		probs: tail.probs,
		headOut: tail.headOut,
		headOutInv: tail.headOutInv,
		groups: tail.groups,
		groupLatent: tail.groupLatent,
		oLatent: tail.oLatent,
		pool: plan.role === 'full' ? undefined : poolSide(plan.kvSrc)
	};
}

// ══════════════════════════════════════════════════════════
// 步骤（按角色）
// ══════════════════════════════════════════════════════════
/** 把 csa2 计划翻译成 hybrid 视图需要的计划（overlap=false、indexer 按角色） */
function toHybridPlan(plan: Csa2Plan): HybridPlan {
	return { ratio: plan.ratio, overlap: false, indexer: plan.isIndexSource };
}

/** 压缩器的步骤（Full 层）：ratio-2 gate+pool；ratio-1 只有 wkv 直投（CED 全局 KV） */
function csa2CompressSteps(
	trace: Csa2Trace,
	w: Csa2Weights,
	cfg: ModelConfigLike,
	prefix: string,
	plan: Csa2Plan,
	/** 本层的 UI 层号（1 开头），label 里的层号都按它写——和界面"检视层"一致 */
	layer: number
): Step[] {
	const S = cfg.seq_len;
	const D = cfg.d_model;
	const hd = cfg.head_dim;
	const dn = cfg.qk_nope_head_dim ?? hd;
	const dr = cfg.qk_rope_head_dim ?? 0;
	const nE = trace.nE;
	const compress = trace.compress!;
	// Full 层是 V4.1 的源层：压缩支路 + indexer 实线（提示换 CSA2 口径）
	const variant: V4Variant = 'v41-full';
	const flow = (active: string[]) => v4Diagram(variant, active);

	if (plan.ratio === 1) {
		// ratio-1：wkv 直投（无 gate）→ RMSNorm → RoPE（CED：从编码器末态投影全局 KV）
		// 编码器末态 = 上一层（UI 层号 − 1）的输出；decoder 段的 Reuse / Reindex 读本层池（UI 层号 + 1 / + 2）
		const prev = layer - 1;
		const readers = `${layer + 1}-${layer + 2}`;
		return [
			{
				id: `${prefix}-global-kv`,
				kind: 'PROJECT',
				diagram: flow(['kvraw']),
				label: `**CED：全局 KV 从这里来**——第 ${layer} 层从"编码器末态"（= 第 ${prev} 层的输出，即本层输入 X）用 W_KV 投影一份潜向量，再过 RMSNorm + RoPE 就是**全局 KV**（ratio-1：每个位置一条，K = V）。这是解码器段唯一一份 main KV——第 ${readers} 层直接读它，不再各自投影`,
				formula: 'k^{\\mathrm{glob}}_t=\\mathcal{R}\\!\\left(\\mathrm{RMSNorm}(W^{KV}x_t)\\right)',
				tensors: [
					{
						name: `${prefix}-global-kv-mm`,
						kind: 'matmul',
						shape: [S, hd],
						a: ref('X', trace.x, [S, D], [S, R.d_model], 'input', `← 第 ${prev} 层的输出（编码器末态）`),
						b: ref('W_KV', w.wkv, [D, hd], [R.d_model, R.head_dim], 'weight'),
						out: ref('k_raw', compress.kv, [S, hd], [S, R.head_dim], 'latent', '← 全局 KV（CED）的原始投影，从编码器末态来（还没归一化/RoPE）')
					},
					{
						name: `${prefix}-global-kv-note`,
						kind: 'shape',
						wide: true,
						shape: [S, hd],
						note: `没有 gate、没有压缩——ratio-1 就是"不压"：每个 token 的潜向量直接当一条 KV。**真实里这是解码器唯一的 main KV**（20 层共享一份），SWA 那部分仍是各层自己算`
					}
				]
			},
			{
				id: `${prefix}-global-kv-norm-rope`,
				kind: 'ROPE',
				diagram: flow(['kvraw', 'kvnorm', 'kv']),
				label: `归一化 + RoPE：一次 RMSNorm 之后对末尾 rope 维做 RoPE，得到全局 KV。K = V 共用——这一份同时当键和值`,
				formula: 'k=v=\\mathcal{R}\\!\\left(\\mathrm{RMSNorm}(W^{KV}x)\\right)',
				tensors: [
					{
						name: `${prefix}-global-kv-norm-rope`,
						kind: 'transform',
						shape: [S, hd],
						cellSize: 32,
						op: 'RMSNorm',
						input: ref('k_raw', compress.kv, [S, hd], [S, R.head_dim], 'latent', '← 上一页的全局 k_raw（CED）'),
						output: ref('k~', compress.entriesNorm, [S, hd], [S, R.head_dim]),
						tail: {
							op: 'RoPE',
							result: ref('全局 KV', compress.entriesReady, [S, hd], [S, R.head_dim], 'cache', '← 框出的末尾 rope 维被旋转；这一份同时当键和值'),
							highlightCols: Array.from({ length: dr }, (_, i) => dn + i)
						}
					},
					{
						name: `${prefix}-global-kv-note`,
						kind: 'shape',
						wide: true,
						shape: [S, hd],
						note: `归一化 + RoPE 与 V4 的 SWA 支路同格式；这里的产物写进共享池（第 ${plan.kvSrc + 1} 层），后续 Reuse / Reindex 层直接读`
					}
				]
			}
		];
	}

	// ratio-2：gate 投影（无重叠 → [v|s] 只有两段）→ softmax（无 ape）→ 组内加权平均
	const gateCols = 2 * hd;
	const scoreCols = Array.from({ length: hd }, (_, i) => hd + i);
	const perGroup = plan.ratio;
	/** 每个 token 在它自己那组里归一化出来的权重（[S × hd]，逐组 groupRows 揭示） */
	const wFull: Mat = Array.from({ length: S }, (_, t) => {
		const g = compress.groups[Math.floor(t / perGroup)];
		return g ? (g.weights[t - g.from] ?? Array(hd).fill(0)) : Array(hd).fill(0);
	});
	const wvFull: Mat = compress.v.map((row, t) => row.map((v, c) => v * wFull[t][c]));
	return [
		{
			id: `${prefix}-compress-gate`,
			kind: 'PROJECT',
			diagram: flow(['x', 'gate', 'w']),
			label: `压缩的第一步：W_gate 把每个 token 投影出**值**（前 ${hd} 列）与**打分**（后 ${hd} 列）两段——**没有重叠窗口**（V4.1 去掉了 CSA 的重叠：每个 token 只投影一份），也**没有 ape**（V4.1 删掉了压缩条目的绝对位置编码）。框出的打分列就是"这个 token 该占多大权重"的原料`,
			formula: '[\\,v\\;|\\;s\\,]=W_{gate}x,\\qquad w_{j,c}=\\mathrm{softmax}_j\\left(s_{j,c}\\right)',
			tensors: [
				{
					name: `${prefix}-gate-mm`,
					kind: 'matmul',
					shape: [S, gateCols],
					cellSize: 24,
					row: 1,
					group: 'gate',
					highlightCols: scoreCols,
					a: ref('X', trace.x, [S, D], [S, R.d_model], 'input'),
					b: ref('W_gate', w.wkvGate!, [D, gateCols], [R.d_model, 2 * R.head_dim], 'weight'),
					out: ref(
						'[v | s]',
						compress.gate,
						[S, gateCols],
						[S, 2 * R.head_dim],
						'latent',
						'← 值 | 打分'
					)
				},
				{
					name: `${prefix}-gate-softmax`,
					kind: 'transform' as const,
					row: 1,
					group: 'chain',
					cellSize: 24,
					shape: [S, hd],
					op: 'softmax（打分）',
					input: ref('s', compress.s, [S, hd], [S, R.head_dim], 'latent', '← [v|s] 的后半段（打分），无 ape 直接 softmax'),
					output: ref('w', wFull, [S, hd], [S, R.head_dim], 'latent', '← 逐通道 softmax 的权重（组内归一化）')
				}
			]
		},
		{
			id: `${prefix}-compress-pool`,
			kind: 'SOFTMAX',
			diagram: flow(['gate', 'w', 'entries', 'ckv']),
			// 与 V4 c4 的 compress-pool 同构：w⊙v / v̄ / 归一化+RoPE **同一行**、先后计算
			label: `组内加权平均：w ⊙ v（打分算出来的权重 × 值）逐组按列求和 = 那条压缩条目 v̄。每组 = 连续的 ${perGroup} 个 token（**不相交**——没有重叠窗口，所以也不用像 V4 的 CSA 那样按窗口堆叠）。**同一行右边紧接着做归一化 + RoPE**：v̄ 先过 RMSNorm 再按它所在组的最后一个 token 做 RoPE，就成了可用的压缩 KV（先后计算，不并发）`,
			formula:
				'\\bar v_g=\\sum_{j\\in g} w_{j}\\,v_{j},\\qquad \\mathrm{kv}=\\mathcal{R}\\!\\left(\\mathrm{RMSNorm}(\\bar v),\\; pos_{group}\\right)',
			tensors: [
				{
					name: `${prefix}-pool-ewise`,
					kind: 'ewise' as const,
					row: 1,
					cellSize: 26,
					shape: [S, hd],
					overview: 'w ⊙ v',
					op: '⊙',
					groupRows: perGroup,
					revealOrder: 'col' as const,
					a: ref('v', compress.v, [S, hd], [S, R.head_dim], 'latent', '← 上一页 [v|s] 的值那几列'),
					b: ref('w', wFull, [S, hd], [S, R.head_dim], 'weight', '← 上一页 softmax 出来的权重（逐 token，组内归一化）'),
					out: ref(
						'w ⊙ v',
						wvFull,
						[S, hd],
						[S, R.head_dim],
						undefined,
						`← 逐元素相乘；框出的那 ${perGroup} 行 = 正在算的那一组；下一步再把每列加起来 = 那 1 条 v̄`
					)
				},
				{
					name: `${prefix}-pool-result`,
					kind: 'matrix' as const,
					row: 1,
					cellSize: 26,
					shape: [nE, hd],
					animated: true,
					// 分开算：w ⊙ v 整块算完，v̄ 才一行一组地出（groupRows = 组宽：一组 2 个 token → 1 条）
					groupRows: perGroup,
					overview: 'v̄',
					title: 'v̄',
					data: compress.entries,
					label: `← ${nE} 条压缩条目：w ⊙ v 算完后，每条 = 它那 ${perGroup} 个 token 按列求和（先出完第 1 条、再第 2 条…）`
				},
				// 归一化 + RoPE：与 w⊙v / v̄ **同一行**，但**先后计算**——先算完 pool 那一组，
				// 这一链才开始。`hideInput`：输入 v̄ 由左边 pool 的结果矩阵画（同一个矩阵只画一次）。
				// 必须放在 note 之前：行分组按"连续同 row 的视图"切，note 没有 row、夹在中间会打断这一行。
				{
					name: `${prefix}-kv-norm-rope`,
					kind: 'transform' as const,
					shape: [nE, hd],
					row: 1,
					cellSize: 26, // 与同一行的 w⊙v / v̄ 一致
					hideInput: true,
					op: 'RMSNorm',
					input: ref(
						'v̄',
						compress.entries,
						[nE, hd],
						[nE, R.head_dim],
						undefined,
						`← 左边 pool 按列加出来的 ${nE} 条压缩条目（每条 = 组内 ${perGroup} 个 token 的值的加权平均）`
					),
					output: ref('v̄~', compress.entriesNorm, [nE, hd], [nE, R.head_dim], undefined, '← 整条 RMSNorm（长度归一，方向不变）'),
					tail: {
						op: 'RoPE',
						result: ref(
							'压缩 KV',
							compress.entriesReady,
							[nE, hd],
							[nE, R.head_dim],
							'cache',
							`← 按组尾位置旋转的 ${nE} 条压缩 KV；K = V 共用，所以这一份既当键又当值`
						),
						highlightCols: Array.from({ length: dr }, (_, i) => dn + i)
					}
				},
				{
					name: `${prefix}-compress-kv-note`,
					kind: 'shape' as const,
					wide: true,
					shape: [nE, hd],
					note: `压缩 KV 只有 ${nE} 条（真实 ratio-2：每 2 个 token 一条）。**无 ape**：V4.1 的压缩条目不带绝对位置编码（对比 V4 CSA 的 ape 页）。产物写进共享池（第 ${plan.kvSrc + 1} 层），后续 Reuse / Reindex 层直接读`
				}
			]
		}
	];
}

/** indexer 步骤（Full 扫全量 / Reindex 池内）：k^I → q^I → 打分 → 加权求和 → top-k（候选源另加建池页） */
function csa2IndexerSteps(trace: Csa2Trace, w: Csa2Weights, cfg: ModelConfigLike, prefix: string, plan: Csa2Plan): Step[] {
	const S = cfg.seq_len;
	const D = cfg.d_model;
	const qr = cfg.q_lora_rank ?? D;
	const nH = cfg.index_n_heads ?? 1;
	const idxHd = cfg.index_head_dim ?? cfg.head_dim;
	const nE = trace.nE;
	const k = cfg.index_topk ?? 1;
	const idx = trace.indexer!;
	// 显式画"W^k 投影 → RMSNorm → RoPE"三步链：k_raw / k_norm 是中间量，trace 里没存，
	// 这里现算一份（和压缩页的 wFull / wvFull 一样，步骤函数从 trace 数据派生展示矩阵）
	const kRaw: Mat = trace.compress?.entriesNorm ? matmul(trace.compress.entriesNorm, w.wk) : [];
	const kNorm: Mat = kRaw.map(rmsNormRow);
	// Full 层是 V4.1 的源层：压缩支路 + indexer 实线（与 V4 的 csa 同构，提示换 CSA2 口径）
	// Full 层：压缩支路实线 + indexer 实线（源层）；Reindex 层：压缩支路淡显（复用）、indexer 实线（池内重打分）
	const variant: V4Variant = plan.role === 'reindex' ? 'v41-reindex' : 'v41-full';
	const srcAction = plan.kvSrc >= 0 ? (cfg.compress_ratios?.[plan.kvSrc] ?? 0) > 1 ? '压缩' : '投影' : undefined;
	const flow = (active: string[]) =>
		v4Diagram(variant, active, plan.role === 'reindex' ? plan.kvSrc : undefined, plan.role === 'reindex' ? srcAction : undefined);
	const scopeLabel = plan.role === 'reindex' ? '（只打候选池）' : '（扫全量）';
	const topkMask: Mat = Array.from({ length: S }, (_, i) =>
		Array.from({ length: nE }, (_, e) => (idx.topk[i].includes(e) ? 1 : 0))
	);
	const headRelu = idx.headRaw.map((h) => h.map((row) => row.map((v) => Math.max(0, v))));
	const hwProd = (h: number) =>
		headRelu[h].map((row, i) => row.map((v) => v * (idx.headWeights[i][h] ?? 0)));
	/** w_0 / w_1 画成 [S × nE]：每头一个标量按行广播到全部条目列（权重按头、条目共用） */
	const wBroadcast = (h: number) =>
		idx.headWeights.map((r) => Array.from({ length: nE }, () => r[h]));

	const steps: Step[] = [
		// ── k^I：从 main KV 投影（Full 层；Reindex 复用池里的，不重投）──
		...(plan.role === 'full'
			? ([
					{
						id: `${prefix}-indexer-k`,
						kind: 'ROPE',
						diagram: flow(['ckv', 'itop']),
						label: `indexer 的打分对象 k^I 从 **main KV 直接投影**（V4.1 的新来源：wk → RMSNorm → RoPE；取代 V4 里"indexer 自己的压缩路径"）——先**投影降维**（每头 ${idxHd} 维，真实 ${R.index_head_dim}；只够分辨谁重要，不承载内容），再归一化、再转 RoPE。本层投影一次，写进共享池`,
						formula: 'k^{I}_e=\\mathcal{R}\\!\\left(\\mathrm{RMSNorm}(W^{k}\\bar v_e)\\right)',
						tensors: [
							{
								name: `${prefix}-indexer-k-mm`,
								kind: 'matmul' as const,
								row: 1,
								group: 'kproj',
								shape: [nE, idxHd],
								cellSize: 32,
								a: ref(
									'v̄~',
									trace.compress?.entriesNorm ?? [],
									[nE, cfg.head_dim],
									[nE, R.head_dim],
									'latent',
									`← 压缩器压好的条目（RMSNorm 后）——每头 ${cfg.head_dim} 维，还没投影`
								),
								b: ref(
									'W^k',
									w.wk,
									[cfg.head_dim, idxHd],
									[R.head_dim, R.index_head_dim],
									'weight',
									`← 投影矩阵：宽从 ${cfg.head_dim} 压到 ${idxHd}（真实 ${R.head_dim} → ${R.index_head_dim}）——**降维就发生在这**`
								),
								out: ref(
									'k_raw',
									kRaw,
									[nE, idxHd],
									[nE, R.index_head_dim],
									'latent',
									'← 投影出的窄键（还没归一化、也没转 RoPE）'
								)
							},
							{
								name: `${prefix}-indexer-k-norm-rope`,
								kind: 'transform' as const,
								row: 1,
								group: 'kchain',
								shape: [nE, idxHd],
								cellSize: 32,
								hideInput: true,
								op: 'RMSNorm',
								input: ref(
									'k_raw',
									kRaw,
									[nE, idxHd],
									[nE, R.index_head_dim],
									'latent',
									'← 左边 W^k 投影出的窄键'
								),
								output: ref(
									'k_raw~',
									kNorm,
									[nE, idxHd],
									[nE, R.index_head_dim],
									'head',
									'← 整行归一化（长度归一，方向不变——RMSNorm 不改形状）'
								),
								tail: {
									op: 'RoPE',
									result: ref(
										'k^I',
										idx.kEntries,
										[nE, idxHd],
										[nE, R.index_head_dim],
										'head',
										'← 按条目出生位置旋转；写进共享池，Reuse / Reindex 层直接读'
									)
								}
							}
						]
					}
				] as Step[])
			: []),
		// ── q^I ──
		{
			id: `${prefix}-indexer-q`,
			kind: 'PROJECT',
			diagram: flow(['cqn', 'itop']),
			label: `indexer 的**查询 q^I**：从共用的 c_Q~ 升维（头更窄，每头 ${idxHd} 维，真实 ${R.index_head_dim} vs 注意力的 ${R.head_dim}——够分辨谁重要就行），按头过 RoPE。${scopeLabel}`,
			formula: 'I_{i,e}=\\sum_h w_{i,h}\\,\\mathrm{ReLU}\\!\\left(q^{I}_{i,h}\\cdot k^{I}_{e}\\right)',
			tensors: [
				{
					name: `${prefix}-indexer-q`,
					kind: 'matmul',
					shape: [S, nH * idxHd],
					a: ref('c_Q~', idx.qLatent, [S, qr], [S, R.q_lora_rank], 'latent'),
					b: ref('W_q^I', w.WqbI, [qr, nH * idxHd], [R.q_lora_rank, R.index_n_heads * R.index_head_dim], 'weight'),
					out: ref('q^I', idx.qFlat, [S, nH * idxHd], [S, R.index_n_heads * R.index_head_dim], 'head', `← ${nH} 头 × 每头 ${idxHd} 维（真实 ${R.index_n_heads} × ${R.index_head_dim}），按头过 RoPE 后参与打分`)
				}
			]
		},
		// ── 打分矩阵乘：q^I_0 · k^Iᵀ → ReLU（单头演示；与 V4 的 indexer 页序对齐：先打分、再算权重并加权）──
		{
			id: `${prefix}-indexer-score`,
			kind: 'MATMUL',
			diagram: flow(['cqn', 'itop']),
			label: `打分矩阵乘，先看**单个头**：head 0 的 q^I_0 对每条压缩条目算点积 q^I_0 · k^Iᵀ → **头0分数**，过 **ReLU**（负数截成 0）→ **头0 ReLU 分数**。head 1 同样算一份；**两头的 ReLU 分数怎么按 w_h 加权成打分矩阵 I、再挑 top-k，在下面两页**。${scopeLabel}`,
			formula: 'r_{i,e}=\\mathrm{ReLU}\\!\\left(q^{I}_{0,i}\\cdot k^{I}_{e}\\right)\\ \\text{（单头演示）}',
			tensors: [
				{
					name: `${prefix}-indexer-score-mm`,
					kind: 'matmul',
					row: 1,
					group: 'mm',
					cellSize: 40,
					shape: [S, nE],
					a: ref('q^I_0', idx.qFlat.map((r) => r.slice(0, idxHd)), [S, idxHd], [S, R.index_head_dim], 'head', '← head 0 的 q'),
					b: ref('k^Iᵀ', transpose(idx.kEntries), [idxHd, nE], [R.index_head_dim, nE], 'head',
						plan.role === 'reindex'
							? `← indexer K 转置（**Reindex 复用第 ${plan.kvSrc + 1} 层的 k^I，不重投**）`
							: `← indexer K 转置（第 {{step:${prefix}-indexer-k}} 步投好）`),
					out: ref('头0分数', idx.headRaw[0], [S, nE], [S, nE], 'result', '← head 0 的 q·k 原始分数；∅ = 条目还没写出来')
				},
				{
					name: `${prefix}-indexer-relu`,
					kind: 'transform',
					row: 1,
					group: 'relu',
					cellSize: 40,
					shape: [S, nE],
					op: 'ReLU',
					hideInput: true,
					input: ref('头0分数', idx.headRaw[0], [S, nE], [S, nE], 'result', '← 左边算出的 头0分数（负数会被截成 0）'),
					output: ref('头0 ReLU 分数', headRelu[0], [S, nE], [S, nE], 'result', '← 负数截成 0；下一页 ⊙ w_0 时作为左操作数再画一遍')
				}
			]
		},
		// ── 头权重 + 按头加权：w_raw → ×scale → w_h，下面 ⊙ w_0 / ⊙ w_1（与 V4 第 17 页对齐：
		//    头0·w0 / 头1·w1 放在 w_h 计算的下面，两个头并列同时算）──
		{
			id: `${prefix}-indexer-w`,
			kind: 'MATMUL',
			diagram: flow(['itop']),
			label: `打分第二步：**头权重 w_h 从哪来**——x 过一个小投影 W^w → **w_raw**（左边矩阵乘），乘 1/√(${idxHd}·${nH}) → **w_h**。然后**每头的 ReLU 分数（上一页）乘上它自己的权重**：head 0 ⊙ w_0、head 1 ⊙ w_1（下面两块**同时算**——一个 token 的两头互不相干），得到 **头0·w0** 和 **头1·w1**`,
			formula: 'w_{i,h}=\\mathrm{scale}\\,(x_i W^{w})_h,\\qquad \\tilde r^{(h)}_{i,e}=w_{i,h}\\,\\mathrm{ReLU}\\!\\left(q^{I}_{i,h}\\cdot k^{I}_{e}\\right)',
			tensors: [
				{
					name: `${prefix}-indexer-w-mm`,
					kind: 'matmul',
					row: 1,
					group: 'wmm',
					cellSize: 34,
					shape: [S, nH],
					a: ref('x', trace.x, [S, D], [S, R.d_model], 'input'),
					b: ref('W^w', w.Ww, [D, nH], [R.d_model, R.index_n_heads], 'weight'),
					out: ref('w_raw', idx.wRaw, [S, nH], [S, R.index_n_heads], 'weight', '← 原始投影（还没缩放）')
				},
				{
					name: `${prefix}-indexer-w-scale`,
					kind: 'transform',
					row: 1,
					group: 'wscale',
					cellSize: 34,
					shape: [S, nH],
					op: `×1/√(${idxHd}·${nH})`,
					hideInput: true,
					input: ref('w_raw', idx.wRaw, [S, nH], [S, R.index_n_heads], 'weight'),
					output: ref('w_h', idx.headWeights, [S, nH], [S, R.index_n_heads], 'weight', '← 缩放后的头权重；下面按列（每头）取用')
				},
				// 每头的 ReLU 分数 × 自己的权重（**并列**：parallel + 同一行 → 同一个动画单元同时算，
				// 列间画浅色虚线——两个乘法互不相干，只是各自独立地算）。与 V4 第 17 页同一套 ewise。
				{
					name: `${prefix}-indexer-mul-0`,
					kind: 'ewise' as const,
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
						`← w_h 的第 0 列（head 0 的权重）按行广播成 ${nE} 格`
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
					kind: 'ewise' as const,
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
		// ── 打分矩阵 I：Σ_h w_h · ReLU(·) + top-k（mask 视图）──
		{
			id: `${prefix}-indexer-logits`,
			kind: 'SELECT',
			diagram: flow(['itop']),
			label: `打分收尾：**Σ 把两头的加权分数相加** → 打分矩阵 I（**头0·w0 / 头1·w1 已在上一页乘好**），算完**紧接着挑 top-${k}**（旁边那块 0/1 选择矩阵，逐行揭示）——被选中的条目才会进注意力。${scopeLabel}`,
			formula: 'I_{i,e}=\\sum_h w_{i,h}\\,\\mathrm{ReLU}\\!\\left(q^{I}_{i,h}\\cdot k^{I}_{e}\\right),\\qquad \\mathrm{top}_{k}(I_{i,\\cdot})',
			tensors: [
				{
					name: `${prefix}-indexer-logits-sum`,
					kind: 'sum',
					row: 1,
					shape: [S, nE],
					terms: Array.from({ length: nH }, (_, h) => ({
						name: `头${h}·w${h}`,
						shape: [S, nE],
						realShape: [S, nE],
						data: hwProd(h),
						label:
							h === 0
								? `← 第 {{step:${prefix}-indexer-w}} 步的 头0·w0`
								: `← 第 {{step:${prefix}-indexer-w}} 步的 头1·w1`
					})),
					result: {
						name: 'I',
						shape: [S, nE],
						realShape: [S, nE],
						data: idx.logits,
						label: `打分矩阵 I：逐行相加；∅ = 条目还没写出来${plan.role === 'reindex' ? '（候选池之外的条目也是 ∅）' : ''}`
					}
				},
				{
					name: `${prefix}-indexer-topk-mask`,
					kind: 'mask',
					row: 1,
					shape: [S, nE],
					overview: 'top-k',
					scores: idx.logits,
					mask: topkMask,
					matrixName: 'top-k 选择',
					label: `每个 query 从 I 自己的那一行挑分数最高的 top-${k} 条（真实 ${R.index_topk} 条）：选中才保留、其余置 ∅。**这张 0/1 矩阵就是下一步注意力掩码里"压缩条目列"的直接来源**`
				},
				// Reindex 页：候选池（打分区）——第 kvSrc+1 层建的池传过来，本层只在池内打分。
				// 跨层引用：数据在全局 kvPools（trace 的 scope）、标签用层号，不走步骤号占位符。
				// 放 row 2：打分页 row 1（sum + top-k 两个 [8×8]）加第三个 [8×8] 会横向溢出，候选池单独一行不打挤
				...(plan.role === 'reindex' && idx.scope && idx.scope.length
					? ([
							{
								name: `${prefix}-indexer-scope-pool`,
								kind: 'matrix' as const,
								row: 2,
								shape: [S, nE],
								cellSize: 22,
								title: '候选池（打分区）',
								// 琥珀框出池内列（按选中块的列分组，与第 6 层建池页的框一致）
								highlightColGroups: Array.from(
									{ length: Math.ceil(idx.scope.length / (cfg.candidate_block_size ?? 1)) },
									(_, g) => idx.scope!.slice(g * (cfg.candidate_block_size ?? 1), Math.min(idx.scope!.length, (g + 1) * (cfg.candidate_block_size ?? 1)))
								),
								data: idx.logits,
								label: `← 第 ${plan.kvSrc + 1} 层建的候选池（打分区，全局一份、每行相同）：琥珀列 = 池内。本层只在池内打分——池外条目不参与（∅），top-k 也只从池里挑`
							}
						] as Step['tensors'])
					: [])
			]
		}
	];

	// 候选源层（UI 第 6 层）追加"候选池"页
	if (plan.isCandidateSource) {
		// full 层的 trace.pool 是 undefined（csa2Compute 里 full 自己读局部 pool，不写 trace.pool），
		// 所以候选池必须从这一层的 indexer 打分**重新算**（与 compute 写进 kvPools 的是同一份推导）。
		/** 块级取 max：按块摊成 [nBlocks × nE]（每行 = 一块，块内格填该块最大分） */
		const bs = cfg.candidate_block_size ?? 1;
		const nBlocks = Math.ceil(nE / bs);
		const blockMaxRows: Mat = Array.from({ length: nBlocks }, (_, b) =>
			Array.from({ length: nE }, (_, e) => {
				if (e < b * bs || e >= Math.min(nE, (b + 1) * bs)) return NEG_INF;
				let mx = NEG_INF;
				for (let i = 0; i < S; i++) if (idx.logits[i][e] > mx) mx = idx.logits[i][e];
				return mx;
			})
		);
		const REAL_BLOCKS = R.candidate_topk_blocks ?? nBlocks;
		const REAL_POOL = (cfg.candidate_topk_blocks ?? 1) * (cfg.candidate_block_size ?? 1);
		const REAL_POOL_TOTAL = (R.candidate_topk_blocks ?? 1) * (R.candidate_block_size ?? 1);
		// 选块：每块的块级最大分（块内所有格子再取 max，与 csa2BuildCandidates 同一推导）降序排序
		const pickK = cfg.candidate_topk_blocks ?? 1;
		const blockScores = blockMaxRows
			.map((row, b) => ({ b, v: Math.max(...row.filter((v) => v !== NEG_INF && Number.isFinite(v))) }))
			.sort((a, b2) => b2.v - a.v);
		// blockmax 矩阵的行标记顺序（分数降序的行下标）与选中块的列组（按选中顺序，给 pool 逐组框）
		const rowOrder = blockScores.map((s) => s.b);
		const colGroups = blockScores.slice(0, pickK).map((s) =>
			Array.from({ length: bs }, (_, k) => s.b * bs + k).filter((e) => e < nE)
		);
		// blockmax 的动画两段：揭示段占比（供 src 的琥珀框压缩到同一窗口同步）——
		// 与 MatrixGrid 的 `pickRatio`（段2 每行按整行权重）同一公式
		const pickRatio = (nBlocks * nE) / (nBlocks * nE + nBlocks * nE);
		steps.push({
			id: `${prefix}-candidate`,
			kind: 'SELECT',
			diagram: flow(['itop']),
			label: `**Hierarchical Sparse Indexer：建候选池**——第 ${plan.kvSrc + 1} 层扫完全量后，按块取**块内最大分数**选 top-${cfg.candidate_topk_blocks} 块（每块 ${bs} 条 = 共 ${(cfg.candidate_topk_blocks ?? 1) * bs} 个候选位置）。后续 Reindex 层**只在池里打分**——深层 indexer 的成本从此与上下文长度无关`,
			formula: '\\text{pool}=\\mathrm{top\\text{-}blocks}\\!\\left(\\max_{e\\in\\text{block}} I_{i,e}\\right)',
			tensors: [
				// 来源：打分矩阵 I（上一步 indexer-logits 的产物）——让块级最大分"从哪来"一眼可见
				{
					name: `${prefix}-candidate-src`,
					kind: 'matrix' as const,
					row: 1,
					shape: [S, nE],
					cellSize: 22,
					title: '打分矩阵 I',
					// 与右边"块级最大分"同一动画单元（同 row + sync）同步推进：`noReveal`
					// 让所有分数一开始就全量显示，`highlightBlockSize` 让琥珀框随块级最大分
					// 逐块移动、框出当前块分数最高的那几格（块级 max 的来源）；
					// `highlightBlockRatio` 把框压缩到 blockmax 的揭示段（选块标记段里 src 静止）。
					animated: true,
					sync: true,
					noReveal: true,
					highlightBlockSize: bs,
					highlightBlockRatio: pickRatio,
					data: idx.logits,
					label: `← 上一步 indexer-logits 的产物（每 query 一行、每条目一列；∅ = 条目还没写出来）。按块取列内最大分 → 右边`
				},
				{
					name: `${prefix}-candidate-blockmax`,
					kind: 'matrix' as const,
					row: 1,
					shape: [nBlocks, nE],
					realShape: [REAL_BLOCKS, R.candidate_block_size ?? nE],
					cellSize: 22,
					animated: true,
					// 与左边"打分矩阵 I"同一动画单元（同 row + sync）：两块同步推进，
					// 块级最大分逐块揭示时，来源矩阵同帧框出当前块的那几格；
					// `pickRows` 让揭示完后再演"选块"：按块分数降序逐行标记（前 top-k 琥珀选中、其余灰落选）
					sync: true,
					overview: '块级最大分',
					title: '块级最大分',
					groupRows: 1,
					pickRows: { rowOrder, selected: pickK },
					data: blockMaxRows,
					label: `← 每行一块（${bs} 条连续条目），格 = 该块内所有 token 里最大的那个分数`
				},
				{
					name: `${prefix}-candidate-pool`,
					kind: 'matrix' as const,
					row: 1,
					shape: [S, nE],
					realShape: [S, REAL_POOL_TOTAL],
					cellSize: 22,
					// 与 src/blockmax 合并进同一动画单元（同 row + sync）：`noReveal` 全量显示分数，
					// `highlightColGroups` 让琥珀列**逐组出现**；`highlightColRatio` + `highlightColSteps`
					// 让列组出现与 blockmax 的行标记严格同步——块 1 标记选中的同时块 1 的列进池、
					// 块 0 标记选中的同时块 0 的列进池（不再等独立单元播，第二框也就不会慢一拍）
					animated: true,
					sync: true,
					noReveal: true,
					overview: '候选池',
					title: '候选池（打分区）',
					highlightColGroups: colGroups,
					highlightColRatio: pickRatio,
					highlightColSteps: nBlocks,
					data: idx.logits,
					label: `← 琥珀框出的列 = 池内候选位置（全局一份、每行相同）；池外只是**不在打分区**，KV 仍保留`
				}
			]
		});
	}
	return steps;
}

/** 按角色组装整条步骤 */
export function csa2Steps(trace: Csa2Trace, ctx: LayerCtx): Step[] {
	const cfg = ctx.cfg;
	const w = ctx.w as Csa2Weights;
	const plan = csa2PlanOf(cfg, ctx.layer);
	const prefix = csa2Prefix(plan.role);
	const hplan = toHybridPlan(plan);
	// 流程图角色：Full = 源层（压缩 + indexer 实线）、Reuse = 都读共享、Reindex = 复用 KV + 池内重打分。
	// 来源层号（Reuse / Reindex 读哪个 kv_source）传给 ckv 节点的淡显说明。
	// 源层动作（cache 节点标"来自哪"）：UI 第 3 层是真压缩（ratio-2 gate+pool）、第 6 层是 CED 全局 KV 投影（ratio-1 wkv 直投）。
	const srcRatio = plan.kvSrc >= 0 ? cfg.compress_ratios?.[plan.kvSrc] ?? 0 : 0;
	const srcAction = srcRatio > 1 ? '压缩' : '投影';
	const diagramRole: { variant: V4Variant; src?: number; srcAction?: string } = (
		plan.role === 'reuse'
			? { variant: 'v41-reuse', src: plan.kvSrc, srcAction }
			: plan.role === 'reindex'
				? { variant: 'v41-reindex', src: plan.kvSrc, srcAction }
				: { variant: 'v41-full' }
	) as { variant: V4Variant; src?: number; srcAction?: string };
	const steps: Step[] = [
		...hybridHeadSteps(trace, w.base, cfg, hplan, prefix, diagramRole),
		...(plan.role === 'full' ? csa2CompressSteps(trace, w, cfg, prefix, plan, ctx.layer + 1) : []),
		...(plan.isIndexSource ? csa2IndexerSteps(trace, w, cfg, prefix, plan) : [])
	];
	// 消费者层（Reuse / Reindex）把池来源传给 hybridTailSteps：琥珀段文案/掩码来源跟着变
	const poolSrc = plan.role === 'full' ? undefined : plan.kvSrc >= 0 ? plan.kvSrc : undefined;
	steps.push(...hybridTailSteps(trace, w.base, cfg, hplan, prefix, poolSrc, diagramRole));
	return steps;
}
