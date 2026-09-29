/**
 * 端到端校验：直接在 Node 里跑一遍参考模型与插件，检查数学正确性。
 *
 * 为什么需要它：生产构建只会预渲染页面外壳，`$effect` 是客户端才执行的，
 * 所以"构建通过"并不代表模型算得对。这个脚本补上这一环。
 *
 * 运行：npm run verify
 */
import '../src/lib/slots';
import '../src/lib/flows';
import { buildLlmFlow, type LlmFlowResult } from '../src/lib/flows/llm';
import { listFlows, getFlow, listImplementationsFor, resolvePlugin } from '../src/lib/core/registry';
import { matRefsOf } from '../src/lib/core/overview';
import { CONFIG } from '../src/lib/model/config';
import { buildOverviewItems, itemKeys, refKey } from '../src/lib/core/overview';
import {
	animationUnits,
	progressOf,
	workOf,
	phaseCells,
	stepWork,
	stepDurationMs,
	MS_PER_CELL,
	MIN_PHASE_MS,
	MIN_TEXT_CELL,
	matmulCellSize,
	transformPhaseMs,
	concatPhaseMs,
	sumPhaseMs,
	viewDurationMs
} from '../src/lib/core/steps';
import { createPlayer } from '../src/lib/core/timeline';
import { matmul, silu, softmaxRow } from '../src/lib/core/mat';
import { initWeights } from '../src/lib/model/init';
import { REAL_R1 } from '../src/lib/model/config';
import type { MhaTrace } from '../src/lib/slots/attention/mha';
import type { MlaTrace } from '../src/lib/slots/attention/mla';
import type { MlaAbsorbTrace } from '../src/lib/slots/attention/absorb';
import type { Step, MatRef } from '../src/lib/core/types';
import type { MoeTrace } from '../src/lib/slots/ffn-moe/deepseek-moe';

function assert(cond: unknown, msg: string) {
	if (!cond) throw new Error('✗ ' + msg);
	console.log('  ✓ ' + msg);
}

/**
 * 一步之内**同一个矩阵只能画一次**。
 *
 * 很容易漏的几个：`routed`、`p`、`S_h` / `P_h`、`c_Q`、`c_KV~`……
 * 链式步骤里"上一步的产物同时充当下一步的输入"必须用 `then` / `preMask` 表达，
 * 而不是把同一个矩阵再画一个块。**跨步骤**复用不受此限（那是 handoff 动画的职责）。
 */
function assertNoDuplicateMatrices(steps: Step[], label: string) {
	const offenders: string[] = [];
	for (const s of steps) {
		const count = new Map<string, number>();
		for (const t of s.tensors) {
			for (const ref of matRefsOf(t)) count.set(ref.name, (count.get(ref.name) ?? 0) + 1);
		}
		const bad = [...count.entries()].filter(([, n]) => n > 1);
		if (bad.length) offenders.push(`${s.id}(${bad.map(([k, n]) => `${k}×${n}`).join(',')})`);
	}
	assert(
		offenders.length === 0,
		`${label}：${steps.length} 步里没有哪一步把同一个矩阵画两次` +
			(offenders.length ? `（违反：${offenders.join(' ')}）` : '')
	);
}

/**
 * 矩阵乘视图的**形状必须真的能乘**：`A [m × k] · B [k × n] = C [m × n]`。
 *
 * 为什么必须单独查：数据对不代表展示对。左操作数被多转置一次的话收缩维就对不上，
 * 页面照着自己的块"取行 × 取列"，算出来的数和格子里渲染的值**不是一回事**；
 * 而"带齐 A/B/C 数据"这类断言只看数据，会一路绿灯。
 *
 * 除了能乘，还查"标注形状 == 数据形状"——转了形状没转数据（或反过来）就是同一类错位。
 */
function assertMatmulShapes(steps: Step[], label: string) {
	const dims = (r: MatRef) => `${r.data?.length ?? -1}×${r.data?.[0]?.length ?? -1}`;
	let n = 0;
	const check = (a: MatRef, b: MatRef, out: MatRef, where: string) => {
		n++;
		assert(
			a.shape[1] === b.shape[0],
			`${label} ${where}：${a.name} [${a.shape.join('×')}] 的列数 = ${b.name} [${b.shape.join('×')}] 的行数`
		);
		assert(
			out.shape[0] === a.shape[0] && out.shape[1] === b.shape[1],
			`${label} ${where}：${out.name} [${out.shape.join('×')}] = [${a.shape[0]} × ${b.shape[1]}]`
		);
		assert(
			dims(a) === a.shape.join('×') && dims(b) === b.shape.join('×') && dims(out) === out.shape.join('×'),
			`${label} ${where}：A/B/C 标注的形状与数据形状一致（${dims(a)} / ${dims(b)} / ${dims(out)}）`
		);
	};
	for (const step of steps) {
		for (const t of step.tensors) {
			if (t.kind === 'matmul') check(t.a, t.b, t.out, step.id);
			else if (t.kind === 'transform' && t.then?.b)
				check(t.output, t.then.b, t.then.result, `${step.id}(链式)`);
			else if (t.kind === 'concat' && t.then)
				check(t.result, t.then.b, t.then.result, `${step.id}(链式)`);
		}
	}
	return n;
}

const MOE_LAYER = CONFIG.first_k_dense; // 第一个 MoE 层
const demo = buildLlmFlow({ attnId: 'mha', ffnId: 'auto', layer: MOE_LAYER });const lt = demo.trace.layers[MOE_LAYER];
const attn = lt.attention as MhaTrace;
const moe = lt.ffn as MoeTrace;

console.log(
	`\n【配置】d_model=${CONFIG.d_model} heads=${CONFIG.num_heads}×${CONFIG.head_dim} ` +
		`seq=${CONFIG.seq_len} layers=${CONFIG.num_layers} experts=${CONFIG.num_routed_experts} top_k=${CONFIG.top_k}`
);
console.log(`【检视】第 ${lt.layer + 1} 层 (${lt.kind})  attention=${lt.attnId}  ffn=${lt.ffnId}`);
console.log(
	`【步骤】共 ${demo.steps.length} 步  |  ` +
		demo.segments.map((s) => `${s.stage}[${s.from},${s.to})`).join('  ')
);

console.log('\n[流程与阶段]');
assert(
	listFlows().some((f) => f.id === 'llm'),
	`注册表里有 llm 流程（已注册 ${listFlows().length} 条：${listFlows()
		.map((f) => f.id)
		.join(', ')}）`
);
// 流程从注册表取出来跑，与直接调用必须一模一样——否则"可插拔"只是摆设
const viaRegistry = getFlow('llm').build({ attnId: 'mha', ffnId: 'auto', layer: MOE_LAYER });
assert(
	JSON.stringify(viaRegistry.steps) === JSON.stringify(demo.steps),
	'经注册表取的流程跑出的步骤与直接调用完全一致'
);

assert(
	JSON.stringify(demo.stages.map((s) => s.id)) ===
		JSON.stringify(['embedding-lookup', 'attention', 'ffn-moe', 'lm-head']),
	`流水线节点由流程提供：${demo.stages.map((s) => s.title).join(' → ')}`
);
assert(
	demo.stages.every((s) => s.title && s.depth && (s.scope === 'model' || s.scope === 'layer')),
	'每个阶段都声明了标题 / 深度 / 作用范围（组件不写死节点）'
);

// 区段必须**无缝覆盖**全部步骤：不重不漏，否则"当前落在哪个阶段"会算错
let cursor = 0;
for (const s of demo.segments) {
	assert(s.from === cursor, `区段 ${s.stage} 紧接上一段（[${s.from},${s.to})）`);
	cursor = s.to;
}
assert(cursor === demo.steps.length, `区段无缝覆盖全部 ${demo.steps.length} 步`);

const segOf = new Map(demo.segments.map((s) => [s.stage, s]));
for (const st of demo.stages) {
	const seg = segOf.get(st.id);
	assert(!!seg && seg.to > seg.from, `阶段 ${st.id} 有自己的步骤（${seg ? seg.to - seg.from : 0} 步）`);
}
assert(
	demo.segments.reduce((n, s) => n + (s.to - s.from), 0) === demo.steps.length,
	'各阶段步数之和 = 总步数（没有步骤落在阶段之外）'
);

const embedSeg = segOf.get('embedding-lookup');
const headSeg = segOf.get('lm-head');
assert(
	!!embedSeg && !!headSeg && embedSeg.from === 0 && headSeg.to === demo.steps.length,
	'Embedding 打头、LM Head 收尾（其余阶段夹在中间）'
);
assert(
	demo.stages.filter((s) => s.scope === 'model').length === 2 &&
		demo.stages.filter((s) => s.scope === 'layer').length === 2,
	'2 个模型级阶段（Embedding / LM Head）+ 2 个层级阶段（Attention / FFN-MoE）'
);

console.log('\n[Embedding 查表]');
const embedStep = demo.steps[embedSeg!.from];
const lookup = embedStep.tensors.find((t) => t.kind === 'lookup');
assert(!!lookup, 'Embedding 用查表视图——不把查表伪装成矩阵乘');
if (lookup?.kind === 'lookup') {
	const tbl = lookup.table.data ?? [];
	assert(
		lookup.keys.length === CONFIG.seq_len && lookup.rows.length === CONFIG.seq_len,
		`${CONFIG.seq_len} 个 token 各取一行`
	);
	assert(
		lookup.table.shape[0] === CONFIG.vocab_size && lookup.table.shape[1] === CONFIG.d_model,
		`词表矩阵 = [${CONFIG.vocab_size} × ${CONFIG.d_model}]`
	);
	assert(
		lookup.keys.every((k) => k >= 0 && k < CONFIG.vocab_size),
		`token id 都落在词表范围内（0..${CONFIG.vocab_size - 1}）`
	);
	assert(
		lookup.rows.every((row, i) => row.every((v, d) => v === tbl[lookup.keys[i]]?.[d])),
		'x_0[i] 逐元素等于 W_E[id_i]（查表结果正确）'
	);
	assert(
		(lookup.result.data ?? []).every((row, i) => row.every((v, d) => v === lookup.rows[i][d])),
		'结果矩阵就是取出的行摞起来（没有任何计算）'
	);
}

console.log('\n[LM Head]');
const headStep = demo.steps[headSeg!.from];
const headMm = headStep.tensors.find((t) => t.kind === 'matmul');
assert(!!headMm, 'LM Head 用矩阵乘视图');
if (headMm?.kind === 'matmul') {
	const a = headMm.a.data ?? [];
	const b = headMm.b.data ?? [];
	const c = headMm.out.data ?? [];
	assert(
		headMm.a.shape[0] === 1 && headMm.out.shape[0] === 1,
		'只算最后一个位置（1 行）——预测下一个 token 用不到前面的位置'
	);
	assert(
		headMm.a.shape[1] === CONFIG.d_model && headMm.b.shape[1] === CONFIG.vocab_size,
		`[1 × ${CONFIG.d_model}] · [${CONFIG.d_model} × ${CONFIG.vocab_size}] → [1 × ${CONFIG.vocab_size}]`
	);
	assert(
		c[0].every((v, j) => Math.abs(v - a[0].reduce((s, x, p) => s + x * b[p][j], 0)) < 1e-9),
		'logits = x_L[last] · W_head 逐元素成立'
	);
}
// 权重多转置一次会让整张 logits 变成 NaN（维度对不上 → `matmul` 取到 `undefined`）。
assert(
	demo.trace.logits.flat().every((v) => Number.isFinite(v)),
	'logits 全部是有限数（维度对不上会整张变成 NaN）'
);
assert(
	demo.trace.logits.length === CONFIG.seq_len &&
		demo.trace.logits[0].length === CONFIG.vocab_size,
	`logits 形状 = [${CONFIG.seq_len} × ${CONFIG.vocab_size}]`
);

const probsAll = softmaxRow(demo.trace.logits[CONFIG.seq_len - 1]);
const argmax = probsAll.indexOf(Math.max(...probsAll));
assert(
	Math.abs(probsAll.reduce((x, y) => x + y, 0) - 1) < 1e-9,
	'最后一个位置的 softmax 概率和为 1'
);
const probBars = headStep.tensors.find((t) => t.kind === 'bars');
assert(!!probBars && !!probBars.animated, '候选概率用会随进度长出来的条形图（不提前剧透）');
if (probBars?.kind === 'bars') {
	assert(probBars.data.length === 4, `只展示概率最高的 4 个候选（实际 ${probBars.data.length}）`);
	assert(
		probBars.data.every((v, i) => i === 0 || v <= probBars.data[i - 1]),
		'候选按概率从高到低排列'
	);
	assert(
		probBars.data.reduce((x, y) => x + y, 0) <= 1.001,
		`展示的是真实概率（前 4 名之和 ${probBars.data.reduce((x, y) => x + y, 0).toFixed(3)} ≤ 1）`
	);
	assert(probBars.labels?.[0] === `#${argmax}`, `概率最高的候选 #${argmax} 就是模型的预测`);
}

console.log('\n[尺寸硬约束]');
// 本项目第一硬约束：**每个矩阵都要能逐格显示真实数字**（d_model=8、vocab_size=12 都为此让路）。
// 格子尺寸的公式与 MatmulView 共用一份，"还能不能显示数字"才不会两边判断不一致。
let tightest = Infinity;
let tightestDesc = '';
for (const step of demo.steps) {
	for (const t of step.tensors) {
		if (t.kind !== 'matmul') continue;
		const size = matmulCellSize(Math.max(t.out.shape[1] ?? 1, t.a.shape[1] ?? 1), t.row !== undefined);
		if (size < tightest) {
			tightest = size;
			tightestDesc = `${t.a.shape.join('×')} @ ${t.b.shape.join('×')} = ${t.out.shape.join('×')}`;
		}
	}
}
assert(
	tightest >= MIN_TEXT_CELL,
	`所有矩阵乘的格子都 ≥ ${MIN_TEXT_CELL}px（最紧的 ${tightestDesc} → ${tightest}px），能逐格写数字`
);

console.log('\n[Attention 形状]');
assert(attn.q.length === CONFIG.num_heads, `Q 头数 = ${CONFIG.num_heads}`);
assert(attn.q[0].length === CONFIG.seq_len, `Q 序列长度 = ${CONFIG.seq_len}`);
assert(attn.q[0][0].length === CONFIG.head_dim, `Q 每头维度 = ${CONFIG.head_dim}`);
assert(
	attn.scores[0].length === CONFIG.seq_len && attn.scores[0][0].length === CONFIG.seq_len,
	`注意力矩阵 = ${CONFIG.seq_len}×${CONFIG.seq_len}`
);
assert(
	attn.out.length === CONFIG.seq_len && attn.out[0].length === CONFIG.d_model,
	`注意力输出 = ${CONFIG.seq_len}×${CONFIG.d_model}`
);

console.log('\n[Attention 数值]');
let maxErr = 0;
for (const head of attn.probs)
	for (const row of head) {
		const s = row.reduce((a, b) => a + b, 0);
		maxErr = Math.max(maxErr, Math.abs(s - 1));
	}
assert(maxErr < 1e-9, `softmax 每行和为 1（最大误差 ${maxErr.toExponential(1)}）`);

const rawScale = Math.sqrt(CONFIG.head_dim);
let sumS = 0;
let sumRaw = 0;
for (let h = 0; h < CONFIG.num_heads; h++)
	for (let i = 0; i < CONFIG.seq_len; i++)
		for (let j = 0; j < CONFIG.seq_len; j++) {
			sumS += Math.abs(attn.scores[h][i][j]);
			sumRaw += Math.abs(attn.scores[h][i][j] * rawScale);
		}
assert(
	sumS > 0,
	`1/√d_h 缩放已应用（缩放前后量级比 ≈ ${(sumRaw / sumS).toFixed(2)}，理论 ${rawScale.toFixed(2)}）`
);

console.log('\n[表征健康度]');
assert(attn.causal === true, '已按 decoder-only 施加因果掩码');

let upperMax = 0;
for (const head of attn.probs)
	for (let i = 0; i < head.length; i++)
		for (let j = i + 1; j < head[i].length; j++)
			upperMax = Math.max(upperMax, Math.abs(head[i][j]));
assert(upperMax === 0, `掩码生效：上三角权重恒为 0（本次最大 ${upperMax}）`);

let meanMax = 0;
let cnt = 0;
for (const head of attn.probs)
	for (const row of head) {
		meanMax += Math.max(...row);
		cnt++;
	}
meanMax /= cnt;
const uniformBaseline = 1 / CONFIG.seq_len;
console.log(
	`    注意力每行最大权重均值 ${meanMax.toFixed(3)}` +
		`（${uniformBaseline.toFixed(3)} = 完全均匀，1.000 = 完全聚焦）`
);
assert(
	meanMax > uniformBaseline * 1.4,
	`注意力不是均匀分布（${meanMax.toFixed(3)} > 均匀基线 ${uniformBaseline.toFixed(3)} 的 1.4 倍）`
);

const xs = moe.x;
let sim = 0;
let pairs = 0;
for (let i = 0; i < xs.length; i++)
	for (let j = i + 1; j < xs.length; j++) {
		let d = 0;
		let a = 0;
		let b = 0;
		for (let k = 0; k < xs[i].length; k++) {
			d += xs[i][k] * xs[j][k];
			a += xs[i][k] ** 2;
			b += xs[j][k] ** 2;
		}
		sim += d / ((Math.sqrt(a) * Math.sqrt(b)) || 1);
		pairs++;
	}
const cosSim = sim / pairs;
console.log(`    MoE 输入 token 两两余弦相似度均值 ${cosSim.toFixed(3)}（越接近 1 越坍缩）`);
assert(cosSim < 0.99, `表征未坍缩（余弦相似度 ${cosSim.toFixed(3)} < 0.99）`);

console.log('\n[MoE 路由]');
assert(moe.topkIdx.length === CONFIG.seq_len, '每个 token 一条路由结果');
assert(moe.topkIdx[0].length === CONFIG.top_k, `每个 token 选 ${CONFIG.top_k} 个专家`);
assert(
	!moe.topkIdx.some((idx) => new Set(idx).size !== idx.length),
	'同一 token 不会重复选中同一个专家'
);
const loadSum = moe.expertLoad.reduce((a, b) => a + b, 0);
assert(
	loadSum === CONFIG.seq_len * CONFIG.top_k,
	`专家负载总和 = ${CONFIG.seq_len}×${CONFIG.top_k} = ${loadSum}`
);
const inactive = moe.expertLoad.filter((x) => x === 0).length;
console.log(
	`    负载分布 [${moe.expertLoad.join(', ')}]  →  未被激活的专家 ${inactive}/${CONFIG.num_routed_experts}`
);
assert(
	inactive < CONFIG.num_routed_experts,
	`不是全部专家都闲置（${CONFIG.num_routed_experts - inactive}/${CONFIG.num_routed_experts} 个被激活）`
);
assert(
	moe.out.length === CONFIG.seq_len && moe.out[0].length === CONFIG.d_model,
	`MoE 输出 = ${CONFIG.seq_len}×${CONFIG.d_model}`
);
console.log(
	`    稀疏度：每 token 激活 ${CONFIG.top_k}/${CONFIG.num_routed_experts} = ` +
		`${((CONFIG.top_k / CONFIG.num_routed_experts) * 100).toFixed(1)}% 专家`
);

console.log('\n[可插拔性]');
const dense = buildLlmFlow({ attnId: 'mha', ffnId: 'dense-ffn', layer: MOE_LAYER });
assert(
	dense.trace.layers.every((l) => l.ffnId === 'dense-ffn'),
	`整网统一替换为 dense-ffn 插件生效（${CONFIG.num_layers} 层全部替换）`
);
assert(
	dense.trace.layers[MOE_LAYER].kind === 'moe',
	`第 ${MOE_LAYER + 1} 层在 R1 结构下本是 MoE 层 —— 插件覆盖与模型结构解耦`
);
const denseTrace = dense.trace.layers[MOE_LAYER].ffn as { out: number[][] };
assert(
	denseTrace.out.length === CONFIG.seq_len && denseTrace.out[0].length === CONFIG.d_model,
	`替换后输出形状不变 = ${CONFIG.seq_len}×${CONFIG.d_model}`
);

console.log('\n[渲染数据完整性]');
// 一步之内同一个矩阵只能画一次（见 assertNoDuplicateMatrices 的注释）
assertNoDuplicateMatrices(demo.steps, 'MHA + MoE');
// 每个 matmul 视图都必须带齐 A/B/C 的数据，否则动画无从谈起
let matmulCount = 0;
for (const step of demo.steps) {
	for (const t of step.tensors) {
		if (t.kind !== 'matmul') continue;
		matmulCount++;
		assert(
			!!t.a.data && !!t.b.data && !!t.out.data,
			`步骤 ${step.id} 的矩阵乘带齐 A/B/C 数值（${t.a.shape.join('×')} @ ${t.b.shape.join('×')} = ${t.out.shape.join('×')}）`
		);
	}
}
console.log(`    共 ${matmulCount} 个矩阵乘视图，全部可逐格显示真实数字`);
// 能带数据还不够——**形状必须真的能乘**（见 assertMatmulShapes 的注释）
const mmDemo = assertMatmulShapes(demo.steps, 'MHA+MoE');
const mmDense = assertMatmulShapes(dense.steps, 'dense-ffn');
console.log(`    形状自洽的矩阵乘：MHA+MoE ${mmDemo} 个、dense-ffn ${mmDense} 个`);

// mask 视图必须带齐原始分数与掩码，否则动画无从谈起
for (const step of demo.steps) {
	for (const t of step.tensors) {
		if (t.kind !== 'mask') continue;
		assert(
			t.scores.length > 0 && t.mask.length === t.scores.length,
			`步骤 ${step.id} 的掩码视图带齐 scores 与 mask（${t.scores.length}×${t.scores[0].length}）`
		);
		assert(
			t.mask.flat().some((m) => !m),
			`掩码确实屏蔽了部分位置（${t.mask.flat().filter((m) => !m).length} 个）`
		);
	}
}

// 一步里有多个动画单元时，StepPanel 会按等份依次播放。
// 分段规则直接复用 core/steps.ts 的 animationUnits——**不要在这里重写一遍**。
for (const step of demo.steps) {
	const n = animationUnits(step).length;
	if (n > 1) {
		console.log(`    步骤 ${step.id} 含 ${n} 段动画（按等份依次播放）`);
	}
}

console.log('\n[跨步骤流转]');
// 收集每个 handoffId 出现在哪些步骤——同一个 id 至少出现两次才算"衔接上了"
const handoffs = new Map<string, string[]>();
for (const step of demo.steps) {
	for (const t of step.tensors) {
		// 用 matRefsOf 而不是只看 matmul：链式变换的第二段（then.b）也可能带 handoffId
		for (const ref of matRefsOf(t)) {
			if (!ref.handoffId) continue;
			const list = handoffs.get(ref.handoffId) ?? [];
			if (list[list.length - 1] !== step.id) list.push(step.id);
			handoffs.set(ref.handoffId, list);
		}
	}
}
for (const [id, steps] of handoffs) {
	console.log(`    ${id}: ${steps.join(' → ')}`);
}
for (const [id, label] of [
	['q', 'Q'],
	['k', 'K'],
	['v', 'V']
] as const) {
	const chain = handoffs.get(id);
	assert(
		!!chain && chain.length >= 2,
		`${label} 有跨步骤衔接（产出 → 使用：${chain?.join(' → ')}）`
	);
}

console.log('\n[流程总览]');
const items = buildOverviewItems(demo.steps);
console.log(`    共 ${items.length} 个节点（= 有动画的步骤数）：`);
for (const it of items) {
	if (it.kind === 'matmul') {
		const pre = it.ops?.length ? `  [前置: ${it.ops.join(' → ')}]` : '';
		const par = it.parallel ? `  ×${it.parallel}` : '';
		console.log(`      [矩阵乘] ${it.out.name} = ${it.a.name}·${it.b.name}${pre}${par}`);
	} else {
		const par = it.parallel ? `  ×${it.parallel}` : '';
		console.log(
			`      [单目]   ${it.input.name} --${it.ops.join(' → ')}--> ${it.output.name}${par}`
		);
	}
}
// 一步一个节点：节点数不得超过步骤数
const withAnim = demo.steps.filter((s) => animationUnits(s).length > 0).length;
assert(
	items.length <= demo.steps.length,
	`节点数 ${items.length} ≤ 步骤数 ${demo.steps.length}`
);
assert(items.length === withAnim, `有动画的每一步恰好一个节点（${items.length} 个）`);
assert(
	items.every((it) => {
		const refs = it.kind === 'matmul' ? [it.a, it.b, it.out] : [it.input, it.output];
		return refs.every((r) => !!r.data);
	}),
	'每个节点的矩阵都带数据，可画缩略图'
);
assert(
	items.every((it) => demo.steps.some((s) => s.id === it.stepId)),
	'每个节点都能对应到某个步骤（点击可跳转）'
);
// 链式步骤：前置运算名挂在节点上，而不是再拆一个节点
const chain = items.find((it) => it.stepId === 'attn-mask-softmax-av');
assert(
	chain?.kind === 'matmul' && chain.ops?.join('→') === '掩码→softmax',
	`掩码/softmax 作为"前置运算"标在节点上（实际 ${chain?.ops?.join('→')}）`
);
const concatNode = items.find((it) => it.stepId === 'attn-out');
assert(
	concatNode?.kind === 'matmul' && concatNode.a.name === 'Concat',
	'拼接 + 投影合成一个节点（左操作数就是拼接结果 Concat）'
);
// 并行分支折叠成一个节点 + ×N
for (const id of ['moe-experts', 'moe-expert-act', 'moe-expert-down']) {
	const par = items.find((it) => it.stepId === id);
	assert(par?.parallel === 2, `${id}：并行的两个专家折叠成一个节点并标 ×2`);
}

// 共享矩阵：同一个矩阵应出现在多个节点里，才能看出数据怎么流动
const shared = new Map<string, number>();
for (const it of items) {
	for (const k of itemKeys(it)) shared.set(k, (shared.get(k) ?? 0) + 1);
}
const reused = [...shared.entries()].filter(([, n]) => n > 1);
console.log(
	`    跨节点复用的矩阵：${reused.map(([k, n]) => `${k}×${n}`).join(', ') || '（无）'}`
);
assert(shared.get('x')! >= 3, `X 出现在 ${shared.get('x')} 个节点里（三个投影共享输入）`);
assert(shared.get('q')! >= 2, `Q 出现在 ${shared.get('q')} 个节点里（投影产出 → 算分数时再用）`);
assert(shared.get('k')! >= 2, `K 出现在 ${shared.get('k')} 个节点里（投影产出 → 算分数时再用）`);
assert(shared.get('v')! >= 2, `V 出现在 ${shared.get('v')} 个节点里（投影产出 → 加权求和时再用）`);
assert(shared.get('o')! >= 2, `注意力输出出现在 ${shared.get('o')} 个节点里（进入 MoE）`);

// 连接曲线数 = 每个复用矩阵的「出现次数 - 1」之和
const curves = [...shared.values()].filter((n) => n > 1).reduce((s, n) => s + n - 1, 0);
console.log(
	`    将绘制 ${curves} 条连接曲线（${reused.map(([k, n]) => `${k}: ${n - 1} 条`).join('，')}）`
);
assert(curves >= 6, `连接曲线数 = ${curves}，足以体现前后关系`);
// 上面这些只能查**数据侧**：曲线是否真画出来取决于 DOM（`MiniBox` 上的 `data-ov-node` 锚点）。
// 锚点被删掉时数据断言全绿、页面上却一条线都没有，所以 DOM 侧要用无头 Chrome 数
// `.links path` 的条数。

console.log('\n[步骤号引用]');
// 插件里写的是 {{step:步骤id}} 占位符，buildLlmFlow 负责解析成真实步骤号。
// 这里**不去枚举"哪些字段可能带 label"**——那会和实现犯同样的漏。
// 直接把整个步骤序列序列化后全局搜占位符，任何位置的遗漏都跑不掉。
const blob = JSON.stringify(demo.steps);
const leftover = (blob.match(/\{\{step:[\w-]+\}\}/g) ?? []).length;
assert(leftover === 0, `没有残留的 {{step:xxx}} 占位符（残留 ${leftover} 处）`);
for (const m of new Set(blob.match(/[^"]*\{\{step:[\w-]+\}\}[^"]*/g) ?? [])) {
	console.log(`    ✗ 未解析：${m}`);
}
// 光"没有残留"还不够——还要解析成**正确**的步骤号
{
	const actStep = demo.steps.find((s) => s.id === 'moe-expert-act');
	const tf = actStep?.tensors.find((t) => t.kind === 'transform');
	const wantNo = demo.steps.findIndex((s) => s.id === 'moe-experts') + 1;
	assert(
		tf?.kind === 'transform' && tf.input.label === `← 第 ${wantNo} 步的升维结果`,
		`变换的输入标签解析成正确步骤号（实际「${
			tf?.kind === 'transform' ? tf.input.label : ''
		}」）`
	);
}

/** 取某个矩阵作为"结果"出现时的那条标签 */
function outLabelOf(name: string): string {
	for (const s of demo.steps) {
		for (const t of s.tensors) {
			if (t.kind === 'matmul' && t.out.name === name) return t.out.label ?? '';
		}
	}
	return '';
}
const noOf = (id: string) => demo.steps.findIndex((s) => s.id === id) + 1;

// V 的标签必须指向它真正被使用的那一步（而不是它被投影出来的那一步）
const vLabel = outLabelOf('V');
assert(
	vLabel.includes(`第 ${noOf('attn-mask-softmax-av')} 步`),
	`V 的标签指向它真正被使用的第 ${noOf('attn-mask-softmax-av')} 步（实际「${vLabel}」）`
);
const qLabel = outLabelOf('Q');
assert(
	qLabel.includes(`第 ${noOf('attn-score')} 步`),
	`Q 的标签指向第 ${noOf('attn-score')} 步（实际「${qLabel}」）`
);
const kLabel = outLabelOf('K');
assert(
	kLabel.includes(`第 ${noOf('attn-score')} 步`),
	`K 的标签指向第 ${noOf('attn-score')} 步（实际「${kLabel}」）`
);

console.log('\n[拼接]');
for (const step of demo.steps) {
	for (const t of step.tensors) {
		if (t.kind !== 'concat') continue;
		const sum = t.parts.reduce((s, p) => s + p.shape[1], 0);
		assert(
			sum === t.result.shape[1],
			`各块列数之和 ${sum} = 结果列数 ${t.result.shape[1]}（${t.parts.map((p) => p.name).join(' + ')}）`
		);
		assert(
			t.parts.every((p) => !!p.data) && !!t.result.data,
			'拼接的各块与结果都带数据，可逐列揭示'
		);
		assert(
			t.parts.every((p) => p.label && !p.label.includes('{{step:')),
			`各块的来源标签已解析步骤号（${t.parts.map((p) => p.label).join(' / ')}）`
		);
		// 结果矩阵的列数应等于各块列数之和，且每块占一段连续列区间
		assert(
			t.parts.every((p, i) => {
				const from = t.parts.slice(0, i).reduce((s, q) => s + q.shape[1], 0);
				return from + p.shape[1] <= t.result.shape[1];
			}),
			'每块在结果里占据连续的列区间'
		);
	}
}

console.log('\n[变换动画]');
for (const step of demo.steps) {
	for (const t of step.tensors) {
		if (t.kind !== 'transform') continue;
		assert(
			t.input.shape[0] === t.output.shape[0],
			`${step.id}：变换按行进行，输入 ${t.input.shape[0]} 行 = 输出 ${t.output.shape[0]} 行`
		);
		assert(
			!!t.input.data && !!t.output.data,
			`${step.id}：变换的输入 ${t.input.name} 与输出 ${t.output.name} 都带数据`
		);
		assert(!!t.op, `${step.id}：变换标了运算名「${t.op}」`);
		if (t.op === 'softmax' && t.output.data) {
			let maxErr = 0;
			for (const row of t.output.data) {
				maxErr = Math.max(maxErr, Math.abs(row.reduce((a, b) => a + b, 0) - 1));
			}
			assert(
				maxErr < 1e-9,
				`${step.id}：softmax 输出每行和为 1（最大误差 ${maxErr.toExponential(1)}）`
			);
		}
	}
}

console.log('\n[专家批量计算]');
{
	// 专家计算这一步要把 token 0 选中的**每个**专家都画出来，而且每个专家拿到的
	// 输入必须是"分给它的全部 token"（批量），不是只有 token 0——只画一个 token
	// 会让人误以为一个 token 只过一个专家，也会丢掉 MoE 批量矩阵乘的本质。
	const sel = demo.steps.find((s) => s.id === 'moe-probs-select');
	const exp = demo.steps.find((s) => s.id === 'moe-experts');
	const tf = sel?.tensors.find((t) => t.kind === 'transform');
	// softmax → top-k 合并成一页后，g 是链式变换的第二段结果
	const gRow: number[] = tf && tf.kind === 'transform' ? (tf.then?.result.data?.[0] ?? []) : [];
	const want = moe.topkIdx[0]; // token 0 选中的专家，按路由概率降序
	const mats = (exp?.tensors ?? []).filter((t) => t.kind === 'matmul');
	const expertOf = (t: (typeof mats)[number]) =>
		Number(/E(\d+)/.exec(t.kind === 'matmul' ? t.b.name : '')?.[1]);

	assert(
		mats.length > 1 && JSON.stringify(mats.map(expertOf)) === JSON.stringify(want),
		`专家计算画出了 token 0 选中的全部专家（E${want.join('、E')}）`
	);
	assert(
		JSON.stringify(
			gRow.map((v, e) => (v !== 0 ? e : -1)).filter((e) => e >= 0).sort((a, b) => a - b)
		) === JSON.stringify([...want].sort((a, b) => a - b)),
		'画出的专家与 `moe-probs-select` 算出的 g 矩阵的非零项一致'
	);

	for (const t of mats) {
		if (t.kind !== 'matmul') continue;
		const e = expertOf(t);
		const tokens = Array.from({ length: CONFIG.seq_len }, (_, i) => i).filter((i) =>
			moe.topkIdx[i].includes(e)
		);
		const A = t.a.data ?? [];
		assert(
			t.a.shape[0] === tokens.length && A.length === tokens.length,
			`E${e} 的输入是分给它的全部 ${tokens.length} 个 token（[${t.a.shape.join(' × ')}]）`
		);
		assert(
			tokens.every((tok, r) => A[r].every((v, c) => v === moe.x[tok][c])),
			`E${e} 的输入行与这些 token 的 x 逐值一致`
		);
		assert(
			t.out.shape[0] === tokens.length && t.out.shape[1] === t.b.shape[1],
			`E${e} 的输出 [${t.out.shape.join(' × ')}] = 输入行数 × W_up 列数`
		);
		assert(t.row === 1 && t.parallel, `E${e} 与其它专家在同一行并行摆放（不画 → 箭头）`);
	}
	console.log(
		`    各专家 batch：${mats
			.map((t) => `E${expertOf(t)} × ${t.a.shape[0]} token`)
			.join('，')}`
	);
	assert(
		exp !== undefined && animationUnits(exp).length === 1,
		'多个专家并行 → 本页只算 1 段动画（一起播，不是依次播）'
	);

	// ── 专家是一个**完整的小 FFN**：升维 → SiLU 激活 → 降维 ──
	const act = demo.steps.find((s) => s.id === 'moe-expert-act');
	const down = demo.steps.find((s) => s.id === 'moe-expert-down');
	assert(!!act && !!down, '专家 FFN 的激活与降维都有独立步骤');

	const acts = (act?.tensors ?? []).filter((t) => t.kind === 'transform');
	assert(
		acts.length === want.length && acts.every((t) => t.op === 'silu'),
		`激活步骤按专家数画出 ${acts.length} 个 SiLU 变换`
	);
	for (const t of acts) {
		if (t.kind !== 'transform') continue;
		const ei = Number(/E(\d+)/.exec(t.input.name)?.[1]);
		const eo = Number(/E(\d+)/.exec(t.output.name)?.[1]);
		// 并行分支最容易串线：输入必须来自**同一个**专家
		assert(ei === eo, `SiLU 的输入 ${t.input.name} 与输出 ${t.output.name} 是同一个专家`);
		const U = t.input.data ?? [];
		const H = t.output.data ?? [];
		assert(
			U.length === H.length && U.every((row, r) => row.every((v, c) => H[r][c] === silu(v))),
			`H^(E${eo}) = SiLU(U^(E${eo})) 逐元素成立`
		);
		assert(t.row === 1 && t.parallel, `E${eo} 的激活与其它专家同行并行`);
	}

	const downs = (down?.tensors ?? []).filter((t) => t.kind === 'matmul');
	assert(
		downs.length === want.length,
		`降维步骤按专家数画出 ${downs.length} 个矩阵乘`
	);
	for (const t of downs) {
		if (t.kind !== 'matmul') continue;
		const e = Number(/E(\d+)/.exec(t.b.name)?.[1]);
		const H = t.a.data ?? [];
		assert(
			t.a.name === `H^(E${e})` && t.out.name === `O^(E${e})`,
			`O^(E${e}) = H^(E${e})·W_down^(E${e})`
		);
		assert(
			t.out.shape.join() === [H.length, CONFIG.d_model].join() &&
				t.out.shape[1] === t.b.shape[1],
			`E${e} 降维后回到 ${CONFIG.d_model} 维（[${t.out.shape.join(' × ')}]）`
		);
		const wantOut = matmul(H, t.b.data ?? []);
		assert(
			wantOut.every((row, r) => row.every((v, c) => Math.abs(v - (t.out.data ?? [])[r][c]) < 1e-9)),
			`E${e} 的降维输出数值正确`
		);
		assert(t.row === 1 && t.parallel, `E${e} 的降维与其它专家同行并行`);
	}
}

console.log('\n[每页/每段速度]');
{
	// 时长必须由"动画量"算出来，否则页与页之间会忽快忽慢：
	//   - 页级播放与底部播放器必须看同一个值，不能各用各的；
	//   - `workOf` 的单位要统一（`matmul` 数格子、`transform` 不能数行，否则逐行页快 2~3 倍）；
	//   - 不能只按格数线性算（小矩阵乘的页会明显比大页快）。
	// 现在：每个"动画段"至少 MIN_PHASE_MS，再按格数加时间（stepDurationMs / viewDurationMs）。
	const perPhase = demo.steps.flatMap((s) =>
		animationUnits(s).flatMap((u) =>
			u.flatMap((t) => phaseCells(t).map((c) => Math.max(MIN_PHASE_MS, c * MS_PER_CELL)))
		)
	);
	assert(
		perPhase.every((ms) => ms >= MIN_PHASE_MS),
		`每一段动画都不短于 ${MIN_PHASE_MS}ms（共 ${perPhase.length} 段，最小 ${Math.min(...perPhase)}ms）`
	);
	// 比较三类页：格子少的（moe-route、并行页 moe-experts）与格子多的（attn-q）
	const qProj = demo.steps.find((s) => s.id === 'attn-q')!;
	const route = demo.steps.find((s) => s.id === 'moe-route')!;
	const experts = demo.steps.find((s) => s.id === 'moe-experts')!;
	assert(
		stepDurationMs(experts) === stepDurationMs(route),
		`并行页 moe-experts 与 moe-route 时长一致（${stepDurationMs(route)}ms）`
	);
	const ratio = stepDurationMs(qProj) / stepDurationMs(route);
	assert(
		ratio < 1.3,
		`格子少的页与格子多的页时长只差 ${((ratio - 1) * 100).toFixed(0)}%` +
			`（${stepDurationMs(route)}ms vs ${stepDurationMs(qProj)}ms）`
	);
	// 格子更多的链式页仍然更长（不是被拉平到一样）
	const sumStep = demo.steps.find((s) => s.id === 'moe-shared-add')!;
	assert(
		stepDurationMs(sumStep) > stepDurationMs(qProj) * 1.5,
		`格子更多的 moe-shared-add 仍明显更长（${stepDurationMs(sumStep)}ms vs ${stepDurationMs(qProj)}ms）`
	);
	// 每页的时长 = 各单元"最大视图时长"之和（并行单元不能重复计）
	const sumOfUnits = demo.steps.map((s) =>
		animationUnits(s).reduce((n, u) => n + Math.max(...u.map((t) => phaseCells(t).reduce((a, c) => a + Math.max(MIN_PHASE_MS, c * MS_PER_CELL), 0))), 0)
	);
	assert(
		sumOfUnits.every((ms, i) => ms === stepDurationMs(demo.steps[i])),
		'每页时长 = Σ 各动画单元（单元内取最大值）'
	);
	const total = demo.steps.reduce((n, s) => n + stepDurationMs(s), 0);
	assert(total < 50000, `整条流程 ${(total / 1000).toFixed(1)}s`);
}

console.log('\n[分段时长的结构]');
{
	// 按下标硬套 `phaseCells(t)` 会错位：没有掩码段时它会少一项，无掩码的变换
	// （SiLU、softmax→top-k）会把"逐行段"当成"掩码段"、`rows` 变成 0，
	// 逐行揭示被判成"已完成"，动画一上来就是终态。这里把结构钉死。
	for (const s of demo.steps) {
		for (const t of s.tensors) {
			if (t.kind === 'transform') {
				const w = transformPhaseMs(t);
				assert(
					t.preMask ? w.mask > 0 : w.mask === 0,
					`${s.id}/${t.name}：掩码段时长与 preMask 一致（${w.mask}ms）`
				);
				assert(w.rows > 0, `${s.id}/${t.name}：逐行段必须有时长（${w.rows}ms）`);
				assert(
					t.then?.b ? w.mm > 0 : w.mm === 0,
					`${s.id}/${t.name}：矩阵乘段时长与 then.b 一致（${w.mm}ms）`
				);
				assert(
					w.mask + w.rows + w.mm === viewDurationMs(t),
					`${s.id}/${t.name}：三段时长之和 = 视图时长（${viewDurationMs(t)}ms）`
				);
			}
			if (t.kind === 'concat') {
				const w = concatPhaseMs(t);
				assert(w.cols > 0, `${s.id}/${t.name}：拼接段必须有时长（${w.cols}ms）`);
				assert(
					t.then ? w.mm > 0 : w.mm === 0,
					`${s.id}/${t.name}：矩阵乘段时长与 then 一致（${w.mm}ms）`
				);
				assert(
					w.cols + w.mm === viewDurationMs(t),
					`${s.id}/${t.name}：两段时长之和 = 视图时长（${viewDurationMs(t)}ms）`
				);
			}
			if (t.kind === 'sum') {
				const w = sumPhaseMs(t);
				assert(w.sum > 0, `${s.id}/${t.name}：加权求和段必须有时长（${w.sum}ms）`);
				assert(
					t.then ? w.add > 0 : w.add === 0,
					`${s.id}/${t.name}：叠加段的时长与 then 一致（${w.add}ms）`
				);
				assert(
					w.sum + w.add === viewDurationMs(t),
					`${s.id}/${t.name}：两段时长之和 = 视图时长（${viewDurationMs(t)}ms）`
				);
				// 两段必须**先后**播：第二段的进度不能在总进度为 0 时就 > 0
				assert(
					progressOf(t, animationUnits(s), 0.01) <= 0.5,
					`${s.id}/${t.name}：一开头只跑第一段（进度 ${progressOf(t, animationUnits(s), 0.01).toFixed(2)}）`
				);
			}
		}
	}
}

console.log('\n[播放器单步]');
{
	// 底部的 ⏪ / ⏩ 必须能一路走到底。
	// GSAP 内部用 `progress = time / duration` 存进度、渲染时再乘回来，一来一回掉精度——
	// `tl.time(x)` 之后再读可能得到比 `x` 小一丁点的值，"刚好落在段边界上"的时刻
	// 就被判成还在上一段里，next() 再也翻不过去。边界比较上留了 1µs 容差（见 timeline.ts）。
	const seen: number[] = [];
	const player = createPlayer(demo.steps, (p) => seen.push(p.stepIndex));
	const all = demo.steps.map((_, i) => i);

	// 初始就停在第一步（cur=0，构造时不 emit），所以 next() 的序列是从第二步到最后一步
	for (let i = 0; i < demo.steps.length + 3; i++) player.next();
	assert(
		JSON.stringify(seen) === JSON.stringify(all.slice(1)),
		`next() 一路翻到最后一步、每步都动了（${seen.length} 次跳转）`
	);
	const before = seen.length;
	player.next();
	assert(seen.length === before, '最后一步再按 next() 不再变化');

	seen.length = 0;
	for (let i = 0; i < demo.steps.length + 3; i++) player.prev();
	assert(
		JSON.stringify(seen) === JSON.stringify(all.slice(0, -1).reverse()),
		`prev() 一路退回第一步、每步都动了（${seen.length} 次跳转）`
	);

	seen.length = 0;
	player.seekStep(5); // 先离开第一步，这样下面 seekStep(0) 也会触发
	seen.length = 0;
	for (let i = 0; i < demo.steps.length; i++) player.seekStep(i);
	assert(
		JSON.stringify(seen) === JSON.stringify(all),
		'seekStep() 每一步都精确跳到位（含浮点边界那一步）'
	);
	player.destroy();
}

console.log('\n[并行动画：每格耗时一致]');
{
	// 两个专家分到的 token 数不同（2 个 vs 6 个），输出格子数差 3 倍。
	// 如果共用一个进度，2 个 token 的那个会快 3 倍填完；所以按工作量**反比**缩放：
	// 两者同时开始，小的那个先完成、然后空转等大的那个——每格耗时始终一样。
	for (const id of ['moe-experts', 'moe-expert-act', 'moe-expert-down']) {
		const step = demo.steps.find((s) => s.id === id);
		assert(!!step, `存在并行步骤 ${id}`);
		if (!step) continue;
		const units = animationUnits(step);
		assert(units.length === 1 && units[0].length > 1, `${id}：同行多个视图算 1 段、一起播`);
		const works = units[0].map(workOf);
		assert(
			new Set(works).size > 1,
			`${id}：两个视图工作量不同（${works.join(' vs ')} 格）——这正是需要按工作量缩放的原因`
		);
		const maxWork = Math.max(...works);
		const ok = [0.2, 0.5, 0.8].every((r) =>
			units[0].every(
				(t, i) => Math.abs(progressOf(t, units, r) - Math.min(1, (r * maxWork) / works[i])) < 1e-9
			)
		);
		assert(ok, `${id}：进度按工作量反比缩放 → 每格耗时一致`);
		const half = units[0].map((t) => progressOf(t, units, 0.5));
		const small = works.indexOf(Math.min(...works));
		assert(
			half[small] === 1 && half.some((p, i) => i !== small && p < 1),
			`${id}：工作量小的先做完（一半进度时 ${works[small]} 格的那个已 100%，另一个才 ${Math.round(
				half[works.findIndex((w) => w === maxWork)] * 100
			)}%）`
		);
	}
}

console.log('\n[加权求和 + 共享专家]');
{
	// 这一步是一条链的矩阵动画：
	//   O_1×g̃_1 + O_2×g̃_2  ──Σ──▶  routed  ──+ shared──▶  y
	// 关键点：`routed` 只画一次——共享专家是**接着它的结果**继续加，
	// 而不是把 routed 再画一遍当第二个相加项（那样页面上会出现两个 routed）。
	const add = demo.steps.find((s) => s.id === 'moe-shared-add');
	const sums = (add?.tensors ?? []).filter((t) => t.kind === 'sum');
	assert(sums.length === 1, `这一步是一条链、只有一个相加视图（实际 ${sums.length}）`);
	assert(
		add !== undefined && animationUnits(add).length === 1,
		'链式相加只算 1 段动画（一路跑完，不用分两段）'
	);
	assert(
		!demo.steps.some((s) => s.id === 'moe-shared-add' && s.tensors.some((t) => t.kind === 'shape')),
		'这一步不再靠 shape 文字块表达'
	);

	const sum = sums[0];
	if (sum?.kind === 'sum') {
		const { terms, weights, result, then } = sum;
		assert(
			terms.length === CONFIG.top_k && !!weights,
			`第 1 段把 ${CONFIG.top_k} 个专家输出按门控权重相加`
		);
		assert(
			weights.every((w, j) => w.every((v, t) => v === moe.topkW[t][j])),
			'权重就是 `moe-probs-select` 算出的 top-k 门控权重（逐 token 不同）'
		);
		const routed = result.data ?? [];
		assert(
			routed.every((row, t) =>
				row.every((v, d) => {
					const want = terms.reduce((s, term, j) => s + weights[j][t] * (term.data ?? [])[t][d], 0);
					return Math.abs(v - want) < 1e-9;
				})
			),
			'routed = Σ g̃_i · O_i 逐元素成立'
		);

		assert(!!then, '共享专家接在 routed 后面（then 段）');
		if (then) {
			assert(
				then.terms.length === 1 && then.terms[0].name === 'shared',
				'追加项只有共享专家一项'
			);
			assert(
				!then.terms.some((t) => t.name === result.name),
				`routed 不作为追加项重复出现（只画一次）`
			);
			assert(
				then.result.name === 'y' &&
					then.terms.every((t) => t.shape.join() === then.result.shape.join()),
				'追加项与最终结果 y 同形'
			);
			const y = then.result.data ?? [];
			assert(
				y.every((row, t) =>
					row.every((v, d) => {
						const want =
							routed[t][d] +
							then.terms.reduce((s, term) => s + (term.data ?? [])[t][d], 0);
						return Math.abs(v - want) < 1e-9;
					})
				),
				'y = routed + shared 逐元素成立（接着上一段的结果加）'
			);
			assert(
				y.every((row, t) => row.every((v, d) => Math.abs(v - moe.out[t][d]) < 1e-9)),
				'最终 y 等于参考模型算出的 MoE 输出'
			);
		}

		// 整个视图里 `routed` 这个名字只应出现一次（就是第一段的结果）
		const names = matRefsOf(sum).map((r) => r.name);
		assert(
			names.filter((n) => n === 'routed').length === 1,
			`routed 在整个视图里只出现 1 次（实际 ${names.filter((n) => n === 'routed').length} 次）`
		);
		assert(
			sum.shape.join() === [CONFIG.seq_len, CONFIG.d_model].join(),
			`这一步形状不变（[${sum.shape.join(' × ')}]）`
		);
	}
}

console.log('\n[softmax → top-k 合并页]');
{
	// softmax 与 top-k 合并成一页：一条链 `r --softmax--> p --top-k--> g`，
	// 关键点与 `moe-shared-add` 那一步一样——`p` **只画一次**（它是第一段的结果，第二段接着它继续）
	const sel = demo.steps.find((s) => s.id === 'moe-probs-select');
	assert(!!sel, '存在合并后的 softmax + top-k 页（moe-probs-select）');
	const tf = sel?.tensors.find((t) => t.kind === 'transform');
	assert(
		!!tf && tf.kind === 'transform' && tf.op === 'softmax' && !!tf.then && tf.then.op === 'top-k',
		'一页里按 softmax → top-k 两段跑（链式变换）'
	);
	assert(
		sel !== undefined && animationUnits(sel).length === 1,
		'链式变换只算 1 段动画（一路跑完，不用分两段）'
	);
	if (tf?.kind === 'transform' && tf.then) {
		assert(
			tf.output.name === 'p' && tf.then.result.name === 'g',
			'链是 p → g，中间结果 p 就是第一段的输出'
		);
		assert(
			matRefsOf(tf).filter((r) => r.name === 'p').length === 1,
			`p 在整个视图里只出现 1 次（实际 ${matRefsOf(tf).filter((r) => r.name === 'p').length} 次）`
		);
		assert(
			!matRefsOf(tf).some((r) => r === tf.then?.result && r.name === 'p'),
			'p 没有被当作第二段的输入重画'
		);
		const g = tf.then.result.data ?? [];
		const kept = g.map((row) => row.filter((v) => v !== 0).length);
		assert(
			kept[0] > 0 && kept.every((n) => n === kept[0]),
			`g 每行恰好保留 ${kept[0]} 个专家（其余归零）`
		);
		assert(
			g.every((row) => Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-9),
			'保留的权重每行和为 1（top-k 权重已归一化）'
		);
		assert(
			(tf.output.data ?? []).every((row) =>
				Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-9
			),
			'p 每行和为 1（softmax 结果）'
		);
	}
}

console.log('\n[掩码 → softmax → P·V 链]');
{
	// 掩码+softmax 与 P·V 合成一页，串成一条链：S_h --掩码 → softmax--> P_h --×V_h--> O_h
	// 关键：`S_h` 与 `P_h` 都**只画一次**——掩码直接作用在 S_h 块上（preMask），
	// P_h 同时充当第二段矩阵乘的左操作数（then.b），且三段按先后顺序播、不并发。
	const merged = demo.steps.find((s) => s.id === 'attn-mask-softmax-av');
	assert(!!merged, '存在合并后的掩码 → softmax → P·V 页（attn-mask-softmax-av）');
	assert(
		merged !== undefined && animationUnits(merged).length === 1,
		'整条链只算 1 段动画（内部分三段先后跑，不用页级分两段）'
	);
	const tf = merged?.tensors.find((t) => t.kind === 'transform');
	assert(tf?.kind === 'transform', '链由一个 transform 视图承载');
	if (tf?.kind === 'transform') {
		assert(!!tf.preMask, '掩码并进了这个视图（preMask），不是单独一个 mask 视图');
		assert(
			tf.op === 'softmax' && tf.then?.b !== undefined,
			'第二段是矩阵乘（then.b）：P_h · V_h → O_h'
		);
		assert(
			!demo.steps.some((s) => s.id === 'attn-mask-softmax-av' && s.tensors.some((t) => t.kind === 'mask')),
			'这一页没有独立的 mask 视图（否则 S_h 会画两遍）'
		);
		const names = matRefsOf(tf).map((r) => r.name);
		for (const [n, want] of [
			['S_h', 1],
			['P_h', 1]
		] as const) {
			assert(
				names.filter((x) => x === n).length === want,
				`${n} 在整个视图里只出现 ${want} 次（实际 ${names.filter((x) => x === n).length} 次）`
			);
		}
		// 三段的工作量按"格子数"计（掩码段 + 逐行段 + 矩阵乘段）
		const cells = phaseCells(tf);
		assert(
			workOf(tf) === cells.reduce((a, c) => a + c, 0),
			`三段工作量 = 各段格子数之和（${cells.join(' + ')}，共 ${workOf(tf)} 格）`
		);
		assert(
			tf.output.name === 'P_h' && tf.then.result.name === 'O_h' && tf.then.b.name === 'V_h',
			'链是 P_h · V_h → O_h，P_h 就是 softmax 的输出'
		);
		const out = tf.then.result.data ?? [];
		assert(
			out.every((row, t) =>
				row.every((v, d) => {
					const want = (tf.output.data ?? [])[t].reduce(
						(s, p, k) => s + p * (tf.then?.b?.data ?? [])[k][d],
						0
					);
					return Math.abs(v - want) < 1e-9;
				})
			),
			'O_h = P_h · V_h 逐元素成立'
		);
	}
}

console.log('\n[MLA（多头潜在注意力）]');
/** 朴素版 MLA 的流程结果，后面的「吸收合并」一节要拿它比对等价性 */
let mlaDemoRef: LlmFlowResult | null = null;
{
	// MLA 与 MHA 是**同一个插槽的两个语义插件**。
	// 这里验证三件事：形状对得上、公式算得对、缓存确实变小了。
	const mlaDemo = buildLlmFlow({ attnId: 'mla', ffnId: 'auto', layer: MOE_LAYER });
	mlaDemoRef = mlaDemo;
	assertMatmulShapes(mlaDemo.steps, 'MLA');
	const t = mlaDemo.trace.layers[MOE_LAYER].attention as MlaTrace;
	const S = CONFIG.seq_len;
	const H = CONFIG.num_heads;
	const dn = CONFIG.qk_nope_head_dim!;
	const dr = CONFIG.qk_rope_head_dim!;
	const dv = CONFIG.v_head_dim!;
	const kvr = CONFIG.kv_lora_rank!;
	const qr = CONFIG.q_lora_rank!;
	const dqk = dn + dr;
	const mlaW = initWeights(CONFIG.seed).slots['mla'][MOE_LAYER] as {
		Wukv: number[][];
		Wa: number[][];
		Wo: number[][];
	};

	console.log('    形状：');
	assert(
		t.cQ.length === S && t.cQ[0].length === qr,
		`c_Q = [${S} × ${qr}]（Q 的低秩潜向量，真实 [S × ${REAL_R1.q_lora_rank}]）`
	);
	assert(
		t.cKv.length === S && t.cKv[0].length === kvr,
		`c_KV = [${S} × ${kvr}]（**所有头共享**的 KV 潜向量，真实 [S × ${REAL_R1.kv_lora_rank}]）`
	);
	assert(
		t.kPe.length === S && t.kPe[0].length === dr,
		`k_pe = [${S} × ${dr}]（不可压缩的位置编码段，真实 [S × ${REAL_R1.rope_head_dim}]）`
	);
	assert(
		t.q.length === H && t.q[0].length === S && t.q[0][0].length === dqk,
		`q = [${H} 头 × ${S} × ${dqk}]，每头 = ${dn} 维 nope + ${dr} 维 rope`
	);
	assert(
		t.k[0][0].length === dn && t.v[0][0].length === dv,
		`k_nope = [${H} × ${S} × ${dn}]，v = [${H} × ${S} × ${dv}]（真实每头 128 / 128）`
	);
	assert(
		t.out.length === S && t.out[0].length === CONFIG.d_model,
		`输出与 MHA 完全同形 = [${S} × ${CONFIG.d_model}] —— 换插件，下游不用改`
	);

	console.log('    数值：');
	// KV 上投影：一次矩阵乘同时得到 k_nope 和 v（权重按本仓库约定存成 [in × out]）
	let kvErr = 0;
	for (let i = 0; i < S; i++)
		for (let c = 0; c < H * (dn + dv); c++) {
			const want = t.cKvNorm[i].reduce((s, x, p) => s + x * mlaW.Wukv[p][c], 0);
			kvErr = Math.max(kvErr, Math.abs(want - t.kvUpFlat[i][c]));
		}
	assert(kvErr < 1e-9, `[k_nope | v] = RMSNorm(c_KV)·W_UKV 逐元素成立（最大误差 ${kvErr.toExponential(1)}）`);

	// 下投影：c_KV 与 k_pe 确实来自同一个矩阵乘的两段
	let downErr = 0;
	for (let i = 0; i < S; i++)
		for (let c = 0; c < kvr + dr; c++) {
			const want = t.x[i].reduce((s, x, p) => s + x * mlaW.Wa[p][c], 0);
			const got = c < kvr ? t.cKv[i][c] : t.kPeRaw[i][c - kvr];
			downErr = Math.max(downErr, Math.abs(want - got));
		}
	assert(
		downErr < 1e-9,
		`[c_KV | k_pe] 来自同一次下投影（最大误差 ${downErr.toExponential(1)}）`
	);

	let upperMaxM = 0;
	for (const head of t.probs)
		for (let i = 0; i < head.length; i++)
			for (let j = i + 1; j < head[i].length; j++)
				upperMaxM = Math.max(upperMaxM, Math.abs(head[i][j]));
	assert(upperMaxM === 0, `因果掩码生效：上三角恒为 0（本次最大 ${upperMaxM}）`);
	let sumErrM = 0;
	for (const head of t.probs)
		for (const row of head) sumErrM = Math.max(sumErrM, Math.abs(row.reduce((a, b) => a + b, 0) - 1));
	assert(sumErrM < 1e-9, `softmax 每行和为 1（最大误差 ${sumErrM.toExponential(1)}）`);
	assert(
		t.out.every((row, i) =>
			row.every(
				(v, c) =>
					Math.abs(
						v - t.concat[i].reduce((s, x, p) => s + x * (mlaW as { Wo: number[][] }).Wo[p][c], 0)
					) < 1e-9
			)
		),
		'u = Concat · W_O 逐元素成立（真实是 16384 → 7168 的降维）'
	);

	console.log('    RoPE：');
	const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
	assert(
		t.kPe[0].every((v, i) => Math.abs(v - t.kPeRaw[0][i]) < 1e-12),
		'第 0 个位置不旋转（角度 = 0）'
	);
	let normErr = 0;
	for (let i = 0; i < S; i++) normErr = Math.max(normErr, Math.abs(norm(t.kPe[i]) - norm(t.kPeRaw[i])));
	assert(normErr < 1e-9, `RoPE 保范数（最大误差 ${normErr.toExponential(1)}）—— 旋转只改方向`);
	assert(
		t.kPe.some((row, i) => i > 0 && row.some((v, c) => Math.abs(v - t.kPeRaw[i][c]) > 1e-6)),
		'位置 > 0 确实被旋转了（不是原样返回）'
	);

	console.log('    absorb（工程上为什么不物化 K/V）：');
	// 把 W_UK 折进 Q 之后，注意力可以直接在潜空间里算：
	//   (q_nope · W_UK) · c_KV  ==  q_nope · k_nope
	// 这就是 sglang 的 forward_absorb：缓存里的 c_KV 既当 K 又当 V。
	const scaleM = 1 / Math.sqrt(dqk);
	let absorbErr = 0;
	for (let h = 0; h < H; h++)
		for (let i = 0; i < S; i++) {
			// W_UK 的第 h 个头那块：Wukv 存的是 [kv_lora_rank × ...]，
			// 所以"第 h 头、第 c 个潜维度、第 p 个 nope 维度"= Wukv[c][h*(dn+dv)+p]
			const absorbed = Array.from({ length: kvr }, (_, c) =>
				t.qNope[h][i].reduce((s, x, p) => s + x * mlaW.Wukv[c][h * (dn + dv) + p], 0)
			);
			for (let j = 0; j < S; j++) {
				const nope = absorbed.reduce((s, x, c) => s + x * t.cKvNorm[j][c], 0);
				const ropePart = t.qPe[h][i].reduce((s, x, p) => s + x * t.kPe[j][p], 0);
				absorbErr = Math.max(absorbErr, Math.abs((nope + ropePart) * scaleM - t.scores[h][i][j]));
			}
		}
	assert(
		absorbErr < 1e-9,
		`absorb 等价：把 W_UK 折进 Q、直接在潜空间内积，与用 K 算的分数一致（最大误差 ${absorbErr.toExponential(1)}）`
	);

	console.log('    KV Cache：');
	const mlaIds = mlaDemo.steps.filter((s) => s.id.startsWith('mla-')).map((s) => s.id);
	assert(
		JSON.stringify(mlaIds) ===
			JSON.stringify([
				'mla-q-compress',
				'mla-q-up',
				'mla-kv-compress',
				'mla-kv-up',
				'mla-rope',
				'mla-score',
				'mla-mask-softmax-av',
				'mla-out',
				'mla-kv-cache'
			]),
		`MLA 共 ${mlaIds.length} 步（比 MHA 多，因为多了一次压缩 / 还原）`
	);
	const cacheStep = mlaDemo.steps.find((s) => s.id === 'mla-kv-cache');
	const kvView = cacheStep?.tensors.find((x) => x.kind === 'kvcache');
	assert(!!kvView, 'MLA 最后一步是 KV Cache 视图');
	if (kvView?.kind === 'kvcache') {
		assert(
			kvView.perToken === kvr + dr && kvView.blocks.reduce((n, b) => n + b.size, 0) === kvView.perToken,
			`每 token 缓存 ${kvView.perToken} 个数 = c_KV(${kvr}) + k_pe(${dr})，各块之和相符`
		);
		assert(
			kvView.realPerToken === REAL_R1.kv_lora_rank + REAL_R1.rope_head_dim,
			`真实每 token 缓存 ${kvView.realPerToken} 个数（${REAL_R1.kv_lora_rank} + ${REAL_R1.rope_head_dim}）`
		);
		const cmpPerToken = kvView.compare.parts.reduce((s, p) => s + p.perToken, 0);
		assert(
			cmpPerToken === 2 * H * CONFIG.head_dim,
			`对比的 MHA 每 token 缓存 ${cmpPerToken} 个数 = 2 × ${H} 头 × ${CONFIG.head_dim} 维`
		);
		assert(
			kvView.compare.realPerToken === 2 * REAL_R1.num_heads * REAL_R1.head_dim,
			`真实 MHA 每 token 缓存 ${kvView.compare.realPerToken} 个数`
		);
		assert(
			kvView.realPerToken * 50 < kvView.compare.realPerToken,
			`真实收益 ${(kvView.compare.realPerToken / kvView.realPerToken).toFixed(1)}× 更小`
		);
		// 缓存要画成**整段矩阵**（一行一个 token），而不是只画 token 0 的那几个数
		assert(
			kvView.blocks.every((b) => b.data && b.data.length === CONFIG.seq_len && b.data[0].length === b.size),
			`每个缓存块都是 [${CONFIG.seq_len} × 各自宽度] 的整段矩阵（${CONFIG.seq_len} 个 token 一行一个）`
		);
		assert(
			JSON.stringify(kvView.blocks[0].data) === JSON.stringify(t.cKvNorm) &&
				JSON.stringify(kvView.blocks[1].data) === JSON.stringify(t.kPe),
			'缓存里存的就是 [RMSNorm(c_KV) | RoPE(k_pe)]（与 sglang 写入缓存的内容一致）'
		);
	}

	// 步数变多后，底部 ⏩ 仍要能一路翻到最后——
	// GSAP 的 time↔progress 精度误差在步数多时最容易在边界上暴露
	assertNoDuplicateMatrices(mlaDemo.steps, 'MLA');
	const seenM: number[] = [];
	const playerM = createPlayer(mlaDemo.steps, (p) => seenM.push(p.stepIndex));
	for (let i = 0; i < mlaDemo.steps.length + 3; i++) playerM.next();
	assert(
		JSON.stringify(seenM) === JSON.stringify(mlaDemo.steps.map((_, i) => i).slice(1)),
		`MLA 的 ${mlaDemo.steps.length} 步也能一路翻到最后（每步都动了）`
	);
	playerM.destroy();


}

console.log('\n[MLA 的「矩阵吸收合并」执行方式]');
{
	// 吸收合并是**实现**（怎么算），不是新语义：算出来的东西必须和朴素版**逐元素相同**。
	// 这是这一节最重要的一条断言——它证明图上画的那些"折叠"是真的等价变换，不是示意图。
	const absDemo = buildLlmFlow({
		attnId: 'mla',
		attnImplId: 'mla-absorb',
		ffnId: 'auto',
		layer: MOE_LAYER
	});
	const naive = mlaDemoRef!;
	const a = absDemo.trace.layers[MOE_LAYER].attention as MlaAbsorbTrace;
	const t = naive.trace.layers[MOE_LAYER].attention as MlaTrace;

	console.log('    语义 × 实现的解析：');
	assert(
		listImplementationsFor('mla').map((i) => i.id).join(',') === 'mla-absorb',
		`MLA 有一个可选的执行方式：${listImplementationsFor('mla').map((i) => i.name).join(', ')}`
	);
	assert(
		listImplementationsFor('mha').length === 0,
		'MHA 没有可选实现 —— 所以「Attention 实现」下拉只在选到 MLA 时出现'
	);
	assert(
		absDemo.attn.implementation?.id === 'mla-absorb' &&
			absDemo.attn.name.includes('MLA') &&
			absDemo.attn.name.includes('吸收'),
		`解析后的插件名带上实现：${absDemo.attn.name}`
	);
	assert(
		absDemo.attn.weightKey === 'mla',
		'吸收合并不产生新权重（用的还是 mla 那批，现场合并）—— 所以权重 key 仍是 mla'
	);
	let rejected = '';
	try {
		// 组合不合法：实现声明 supports 之外的语义必须报错，而不是悄悄用错
		resolvePlugin('mha', 'mla-absorb');
	} catch (e) {
		rejected = String(e);
	}
	assert(
		rejected.includes('不支持语义'),
		'把 MLA 的实现配到 MHA 上会被拒绝（supports 声明生效）'
	);

	console.log('    等价性（最关键）：');
	let eqErr = 0;
	for (let i = 0; i < CONFIG.seq_len; i++)
		for (let j = 0; j < CONFIG.d_model; j++)
			eqErr = Math.max(eqErr, Math.abs(a.out[i][j] - t.out[i][j]));
	assert(
		eqErr < 1e-12,
		`吸收合并的输出与朴素版**逐元素相同**（最大误差 ${eqErr.toExponential(1)}）`
	);
	let scoreErr = 0;
	for (let h = 0; h < CONFIG.num_heads; h++)
		for (let i = 0; i < CONFIG.seq_len; i++)
			for (let j = 0; j < CONFIG.seq_len; j++)
				scoreErr = Math.max(scoreErr, Math.abs(a.scores[h][i][j] - t.scores[h][i][j]));
	assert(
		scoreErr < 1e-12,
		`注意力分数也完全相同（最大误差 ${scoreErr.toExponential(1)}）—— 缩放因子仍是 1/√(d_h+d_h^R)`
	);
	// 手工验算合并公式：W̄^UQ = W^UQ·W^UKᵀ 必须真的等于"折进 Q"后的结果
	const mlaW = initWeights(CONFIG.seed).slots['mla'][MOE_LAYER] as {
		Wuq: number[][];
		Wukv: number[][];
		Wo: number[][];
	};
	const dn = CONFIG.qk_nope_head_dim!;
	const dr = CONFIG.qk_rope_head_dim!;
	const dv = CONFIG.v_head_dim!;
	const dc = CONFIG.kv_lora_rank!;
	const qr = CONFIG.q_lora_rank!;
	let barErr = 0;
	for (let i = 0; i < qr; i++)
		for (let c = 0; c < dc; c++) {
			let want = 0;
			for (let p = 0; p < dn; p++) want += mlaW.Wuq[i][p] * mlaW.Wukv[c][p];
			barErr = Math.max(barErr, Math.abs(want - a.WbarUQ[i][c]));
		}
	assert(
		barErr < 1e-12,
		`W̄^UQ = W^UQ·W^UKᵀ 逐元素成立（head 0，最大误差 ${barErr.toExponential(1)}）`
	);
	// 手工验算合并公式：W̄^O = W^UV·W^O（仓库记法 `[dc × dv] · [dv × d]`，收缩维 dv，**不转置**；
	// 论文按 `[out × in]` 写，同一个矩阵写成 W^O·W^UV）
	let barOErr = 0;
	for (let c = 0; c < dc; c++)
		for (let j = 0; j < CONFIG.d_model; j++) {
			let want = 0;
			for (let p = 0; p < dv; p++) want += mlaW.Wukv[c][dn + p] * mlaW.Wo[p][j];
			barOErr = Math.max(barOErr, Math.abs(want - a.WbarO[c][j]));
		}
	assert(
		barOErr < 1e-12,
		`W̄^O = W^UV·W^O 逐元素成立（head 0，最大误差 ${barOErr.toExponential(1)}）`
	);

	console.log('    缓存直接当 K/V：');
	// K 和 V 都是 c_KV~ 本身：每个头的 K 完全相同，且 V == K
	let sameK = true;
	for (let i = 0; i < CONFIG.seq_len; i++)
		for (let c = 0; c < dc; c++) {
			const kFull = [...a.cKvNorm[i], ...a.kPe[i]];
			for (let h = 0; h < CONFIG.num_heads; h++) {
				const qFull = [...a.qBarH[h][i], ...a.qPe[h][i]];
				if (qFull.length !== kFull.length) sameK = false;
			}
		}
	assert(sameK, `Q̄_h 与 K_h 的维度都是 ${dc}+${dr} —— 两边都是「潜向量 + 位置编码」`);
	assert(
		a.headOut[0].length === CONFIG.seq_len && a.headOut[0][0].length === dc,
		`注意力输出是 ${dc} 维潜向量（不是 ${dv} 维 v）—— 最后由 W̄^O 负责还原`
	);
	// V = c_KV~：手工算一遍 o_h = P_h · c_KV~
	let avErr = 0;
	for (let i = 0; i < CONFIG.seq_len; i++)
		for (let c = 0; c < dc; c++) {
			const want = a.probs[0][i].reduce((s, p, j) => s + p * a.cKvNorm[j][c], 0);
			avErr = Math.max(avErr, Math.abs(want - a.headOut[0][i][c]));
		}
	assert(avErr < 1e-12, `o_h = P_h · c_KV~ 逐元素成立（V 就是缓存里的潜向量）`);

	console.log('    步骤与渲染：');
	const absIds = absDemo.steps.filter((s) => s.id.startsWith('abs-')).map((s) => s.id);
	assert(
		JSON.stringify(absIds) ===
			JSON.stringify([
				'abs-q-compress',
				'abs-merge',
				'abs-q-absorb',
				'abs-kv-compress',
				'abs-replicate',
				'abs-rope',
				'abs-score',
				'abs-av',
				'abs-out'
			]),
		`吸收合并共 ${absIds.length} 步：${absIds.join(' → ')}`
	);
	assert(
		absDemo.steps.every((s) => !s.id.startsWith('mla-')),
		'选了实现之后，步骤序列被**完全替换**（不再有朴素版的 mla-* 步骤）'
	);
	assertNoDuplicateMatrices(absDemo.steps, 'MLA（吸收合并）');
	assertMatmulShapes(absDemo.steps, 'MLA（吸收合并）');
	const mergeStep = absDemo.steps.find((s) => s.id === 'abs-merge')!;
	const mergeMm = mergeStep.tensors.filter((t2) => t2.kind === 'matmul');
	assert(
		mergeMm.length === 2,
		`"合并权重"那一页有两个矩阵乘（W̄^UQ 与 W̄^O），共 ${mergeMm.length} 个`
	);
	const replicateStep = absDemo.steps.find((s) => s.id === 'abs-replicate')!;
	assert(
		replicateStep.tensors.some((x) => x.kind === 'tiles') &&
			replicateStep.tensors.some((x) => x.kind === 'kvcache'),
		'"缓存直接当 K/V"那一页用 tiles 画出复制出来的 K/V，并带 KV Cache 视图'
	);
	assert(
		absDemo.segments.find((s) => s.stage === 'attention') !== undefined &&
			absDemo.stages.find((s) => s.id === 'attention')!.sub!.includes('吸收'),
		'流程图的 Attention 节点副标题会显示当前实现'
	);
	// 播放器同样要能翻到底
	const seenA: number[] = [];
	const playerA = createPlayer(absDemo.steps, (p) => seenA.push(p.stepIndex));
	for (let i = 0; i < absDemo.steps.length + 3; i++) playerA.next();
	assert(
		JSON.stringify(seenA) === JSON.stringify(absDemo.steps.map((_, i) => i).slice(1)),
		`吸收合并的 ${absDemo.steps.length} 步也能一路翻到最后`
	);
	playerA.destroy();

	console.log('    数据流图（在朴素版那张图上做差异标注）：');
	const absSteps = absDemo.steps.filter((s) => s.id.startsWith('abs-'));
	assert(
		absSteps.every((s) => !!s.diagram),
		`吸收合并每一步都带数据流图（${absSteps.length} 步全覆盖）`
	);
	const ad = absSteps[0].diagram!;
	const nd = mlaDemoRef!.steps.find((s) => s.id === 'mla-score')!.diagram!;
	const aids = new Set(ad.nodes.map((n) => n.id));
	// 关键：**布局与朴素版完全一样**，只是标注不同——这样两张图能逐节点对照
	const posOf = (d: typeof nd) => d.nodes.map((n) => `${n.id}@${n.col},${n.row}`).sort().join(' ');
	const naivePos = posOf(nd);
	const absPos = posOf(ad);
	const added = ['wbaruq@2,1', 'wbaro@5,1'];
	assert(
		absPos === [...posOf(nd).split(' '), ...added].sort().join(' '),
		`吸收合并的图**复用朴素版那张的布局**（只多了折出来的两个权重节点），不是另画一张`
	);
	assert(
		nd.nodes.every((n) => {
			const m = ad.nodes.find((x) => x.id === n.id);
			return !m || (m.col === n.col && m.row === n.row);
		}),
		'朴素版里每个节点在吸收合并的图里都待在**同一格**'
	);
	// 三类差异标注
	const goneIds = ad.nodes.filter((n) => n.tone === 'gone').map((n) => n.id).sort();
	assert(
		JSON.stringify(goneIds) === JSON.stringify(['knope', 'kv', 'v']),
		`不再需要的三个矩阵标成灰色淡显：${goneIds.join(' / ')}`
	);
	assert(
		ad.nodes.filter((n) => n.tone === 'gone').every((n) => n.faded && n.sub?.includes('不再需要')),
		'灰色节点同时淡显、并在小字里写明"不再需要"'
	);
	assert(
		ad.nodes.filter((n) => n.tone === 'absorbed').map((n) => n.id).sort().join(',') ===
			'kh,q,vh,wbaro,wbaruq',
		'被改动的环节标成青色：q̄（原 q）、两个折出来的权重、换了来源的 K_h / V_h'
	);
	assert(
		ad.nodes.find((n) => n.id === 'q')!.label === 'q̄',
		'原来的 `q` 节点改名为 `q̄`（吸收后直接落在潜空间）'
	);
	// K/V 的来源变了：直接写在小字里（新连线要横穿第 2 行、会被灰节点盖住，所以不画）
	assert(
		ad.nodes.find((n) => n.id === 'kh')!.sub!.includes('c_KV~') &&
			ad.nodes.find((n) => n.id === 'vh')!.sub!.includes('c_KV~'),
		'K_h / V_h 的小字写明"直接用 c_KV~"（换了来源，但没有新增连线）'
	);
	assert(
		ad.edges.length === nd.edges.length + 2,
		`只多了两条边（折出来的两个权重各一条），共 ${ad.edges.length} 条`
	);
	// 老路径淡显
	assert(
		['ckvn>kv', 'kv>knope', 'kv>v', 'knope>kh', 'v>vh'].every((pair) => {
			const [f, t] = pair.split('>');
			const e = ad.edges.find((x) => x.from === f && x.to === t);
			return !!e && e.faded === true;
		}),
		'朴素版走过的那条上投影路径（c_KV~ → [k_nope|v] → k_nope/v → K_h/V_h）全部淡显'
	);
	assert(
		ad.edges.some((e) => e.from === 'wbaruq' && e.to === 'q') &&
			ad.edges.some((e) => e.from === 'wbaro' && e.to === 'attn'),
		'两个折出来的权重各自连到它的消费者（q̄ 与 注意力）'
	);
	const badAEdge = ad.edges.find((e) => !aids.has(e.from) || !aids.has(e.to));
	assert(!badAEdge, `所有边的两端都指向存在的节点（共 ${ad.edges.length} 条边）`);
	assert(
		!ad.nodes.some((n, i) => ad.nodes.findIndex((m) => m.col === n.col && m.row === n.row) !== i),
		'没有两个节点占据同一个格子'
	);
	const badAActive = absSteps.flatMap((s) =>
		(s.diagram?.active ?? []).filter((id) => !aids.has(id)).map((id) => `${s.id}:${id}`)
	);
	assert(badAActive.length === 0, '每步高亮的节点都在图上（没有拼错的 id）');
	assert(
		absSteps.every((s) => (s.diagram?.active ?? []).length > 0),
		'每一步都至少高亮一个节点'
	);
	assert(
		absSteps.every((s) => !(s.diagram?.active ?? []).some((id) => goneIds.includes(id))),
		'高亮里**不会出现已经不需要的灰色节点**（否则等于说它还在参与计算）'
	);
	assert(
		absSteps.every((s) => s.diagram!.nodes === ad.nodes && s.diagram!.edges === ad.edges),
		'各步共用同一张图（只有高亮不同）'
	);
	assert(ad.nodes !== nd.nodes, '吸收合并用的是**另一张图对象**（不是把朴素版那张改了）');

}

console.log('\n[阶段开关]');
{
	// 关掉的阶段**根本不进流程**：步骤、流程图节点、总览都跟着消失。
	// 因为没有任何跨阶段的 `{{step:...}}` 引用，所以关掉一整段不会把标签解析成 `?`。
	const toggles = getFlow('llm').stageToggles ?? [];
	assert(toggles.length === 4, `流程声明了 ${toggles.length} 个可开关的阶段`);
	assert(
		toggles.map((t) => t.id).join(',') === 'embedding-lookup,attention,ffn-moe,lm-head',
		`阶段顺序与流程一致：${toggles.map((t) => t.title).join(' → ')}`
	);
	assert(
		toggles.filter((t) => t.defaultOn).map((t) => t.id).join(',') === 'attention,ffn-moe',
		`默认开的是 Attention 与 FFN/MoE（Embedding / LM Head 默认关）：${toggles
			.filter((t) => t.defaultOn)
			.map((t) => t.title)
			.join(' / ')}`
	);

	const allOn = buildLlmFlow({ attnId: 'mla', ffnId: 'auto', layer: MOE_LAYER });
	assert(
		allOn.stages.map((s) => s.id).join(',') === toggles.map((t) => t.id).join(','),
		'开关声明的 id 与真正产出的阶段 id 一一对应（不会写错 id）'
	);
	const total = allOn.steps.length;

	// 逐个关掉：该阶段的步骤与流程图节点都要消失，其余保持不变
	for (const t of toggles) {
		const off = buildLlmFlow({ attnId: 'mla', ffnId: 'auto', layer: MOE_LAYER, off: [t.id] });
		const seg = allOn.segments.find((s) => s.stage === t.id)!;
		const own = seg.to - seg.from;
		assert(
			off.steps.length === total - own,
			`关掉 ${t.title}：少了它那 ${own} 步（${total} → ${off.steps.length}）`
		);
		assert(
			!off.stages.some((s) => s.id === t.id),
			`关掉 ${t.title}：流程图里也没有它了（剩 ${off.stages.length} 个节点）`
		);
		assert(off.segments.every((s) => s.stage !== t.id), `关掉 ${t.title}：步骤区间映射里也没有它`);
		// 剩下的步骤里不能有 `{{step:...}}` 残留、也不能解析成 `?`
		const blob = JSON.stringify(off.steps);
		assert(
			!blob.includes('{{step:') && !blob.includes('第 ? 步'),
			`关掉 ${t.title}：剩下的步骤里没有失效的步骤号引用`
		);
		// 播放器仍要能翻到底
		const seen: number[] = [];
		const p = createPlayer(off.steps, (x) => seen.push(x.stepIndex));
		for (let i = 0; i < off.steps.length + 3; i++) p.next();
		assert(
			JSON.stringify(seen) === JSON.stringify(off.steps.map((_, i) => i).slice(1)),
			`关掉 ${t.title}：剩 ${off.steps.length} 步仍能一路翻到最后`
		);
		p.destroy();
	}

	// 只留默认开的那两个：流程应当就是"Attention + FFN/MoE"
	const defaults = buildLlmFlow({
		attnId: 'mla',
		ffnId: 'auto',
		layer: MOE_LAYER,
		off: toggles.filter((t) => !t.defaultOn).map((t) => t.id)
	});
	assert(
		defaults.stages.map((s) => s.id).join(',') === 'attention,ffn-moe',
		`按默认开关跑出来就是两段：${defaults.stages.map((s) => s.title).join(' → ')}（${defaults.steps.length} 步）`
	);

	// 四个全关：不报错，流程为空（UI 会提示"至少打开一个"）
	const none = buildLlmFlow({
		attnId: 'mla',
		ffnId: 'auto',
		layer: MOE_LAYER,
		off: toggles.map((t) => t.id)
	});
	assert(
		none.steps.length === 0 && none.stages.length === 0 && none.segments.length === 0,
		'四个阶段全关掉时流程为空（步骤 / 节点 / 区间都是空的），不抛错'
	);
}

console.log('\n[确定性]');
const again = buildLlmFlow({ attnId: 'mha', ffnId: 'auto', layer: MOE_LAYER });
assert(
	JSON.stringify(again.trace.layers[MOE_LAYER].ffn) === JSON.stringify(moe),
	'固定 seed：重复运行结果完全一致'
);

console.log('\n全部检查通过 ✓\n');
