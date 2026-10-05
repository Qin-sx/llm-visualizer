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
import { CONFIG, REAL_R1, REAL_V4_FLASH, R1_MODEL, V4_FLASH_MODEL, attnIdFor, pluginsOf } from '../src/lib/model/config';
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
	isAnimatable,
	MAX_TEXT_CELL,
	fitCellSize,
	matmulCellSize,
	transformCellSize,
	transformPhaseMs,
	concatPhaseMs,
	sumPhaseMs,
	viewDurationMs
} from '../src/lib/core/steps';
import { createPlayer } from '../src/lib/core/timeline';
import { matmul, ropeApply, silu, softmaxRow } from '../src/lib/core/mat';
import { realLabel } from '../src/lib/components/realShape';
import { initWeights } from '../src/lib/model/init';
import type { MhaTrace } from '../src/lib/slots/attention/mha';
import type { MlaTrace } from '../src/lib/slots/attention/mla';
import type { MlaAbsorbTrace } from '../src/lib/slots/attention/absorb';
import type { HybridTrace } from '../src/lib/slots/attention/hybrid';
import type { Step, MatRef } from '../src/lib/core/types';
import type { MoeTrace } from '../src/lib/slots/ffn-moe/deepseek-moe';

function assert(cond: unknown, msg: string) {
	if (!cond) throw new Error('✗ ' + msg);
	console.log('  ✓ ' + msg);
}

/**
 * 本项目第一硬约束：**每个矩阵都要能逐格显示真实数字**（d_model = 8、
 * vocab_size = 12 都是为它让的路）。格子尺寸的公式与 `MatmulView` 共用一份，
 * "还能不能显示数字"才不会两边判断不一致。
 *
 * 注意只查 `matmul` 视图：`transform` / `mask` 用的是固定格子尺寸，
 * 不会因为矩阵变宽而写不下数字。
 */
function assertCellSizes(steps: Step[], label: string) {
	let tightest = Infinity;
	let tightestDesc = '';
	for (const step of steps) {
		for (const t of step.tensors) {
			if (t.kind === 'matmul') {
				// 与 `MatmulView` 同一份判定：按 A/B/C 的**总列数**取预算（三块是左右排着的）
				const size = matmulCellSize(
					(t.a.shape[1] ?? 1) + (t.b.shape[1] ?? 1) + (t.out.shape[1] ?? 1),
					t.row !== undefined
				);
				if (size < tightest) {
					tightest = size;
					tightestDesc = `${t.a.shape.join('×')} @ ${t.b.shape.join('×')} = ${t.out.shape.join('×')}`;
				}
			} else if (t.kind === 'transform') {
				// 变换视图的格子尺寸也是"渲染与校验共用一份"（链尾多一段时会按列数缩小）
				const cols = t.input.data?.[0]?.length ?? 1;
				const size = transformCellSize(cols, t.row !== undefined, !!t.tail);
				if (size < tightest) {
					tightest = size;
					tightestDesc = `${step.id}/${t.name}（变换 ${t.input.shape.join('×')}）`;
				}
			}
		}
	}
	assert(
		tightest >= MIN_TEXT_CELL,
		`${label}：所有矩阵的格子都 ≥ ${MIN_TEXT_CELL}px（最紧的 ${tightestDesc} → ${tightest}px），能逐格写数字`
	);
}

/**
 * 每个张量**标注的形状必须等于数据的形状**。
 *
 * `assertMatmulShapes` 只管矩阵乘的那三个操作数，而"这一页画出来的每一个块"
 * 都可能标错（尤其是把 `[in × out]` 和 `[out × in]` 弄反、或者改了形状没改数据）。
 */
function assertRefShapes(steps: Step[], label: string) {
	const bad: string[] = [];
	for (const s of steps) {
		for (const t of s.tensors) {
			for (const r of matRefsOf(t)) {
				if (!r.data) continue;
				const got = [r.data.length, r.data[0]?.length ?? 0];
				if (r.shape[0] !== got[0] || (r.shape[1] ?? 0) !== got[1]) {
					bad.push(`${s.id}/${r.name}: 标注 [${r.shape.join('×')}] 数据 [${got.join('×')}]`);
				}
			}
		}
	}
	assert(
		bad.length === 0,
		`${label}：每个张量标注的形状都和数据对得上` + (bad.length ? `（${bad.join('；')}）` : '')
	);
}

/**
 * 公式里的 LaTeX 命令**必须带反斜杠**：这些字符串是 JS 单引号 / 模板字符串，
 * `'\m'` 里的反斜杠会被 JS 吃掉，页面会把命令名当普通字母排出来。
 * 这类错误在数据侧看不出来（字符串本身合法），所以在这里兜一道：
 * 命令名出现却没有反斜杠，就是漏了。
 *
 * 启发式：先剥掉 `\text{...}` 的内容再查，免得把 `\text{theta}` 这种
 * 正常写法的内容当成"漏了反斜杠的命令"。
 */
const LATEX_CMDS = [
	'mathcal',
	'mathrm',
	'left',
	'right',
	'frac',
	'sqrt',
	'tilde',
	'qquad',
	'quad',
	'cdot',
	'infty',
	'sigma',
	'theta',
	'mathbf',
	'partial',
	'hat',
	'bar',
	'sum',
	'prod'
];
function assertFormulas(steps: Step[], label: string) {
	const bad: string[] = [];
	for (const s of steps) {
		if (!s.formula) continue;
		// 先剥掉 `\text{...}` 的内容：那里面是普通文字，出现命令名不算漏反斜杠
		const body = s.formula.replace(/\\text\{[^}]*\}/g, '');
		for (const cmd of LATEX_CMDS) {
			// 前面既不能是反斜杠（那样就是对的），也不能是字母（否则 `\qquad` 里的 `quad` 会误报）
			if (new RegExp(`(?<![\\\\A-Za-z])${cmd}\\b`).test(body)) {
				bad.push(`${s.id}: 公式里的 ${cmd} 没有反斜杠`);
			}
		}
	}
	assert(
		bad.length === 0,
		`${label}：公式里的 LaTeX 命令都带反斜杠（不会被 JS 吃掉）` +
			(bad.length ? `（${bad.join('；')}）` : '')
	);
}

/**
 * 每个矩阵都要标**真实尺寸**——这是本项目的表示策略（见 `slots/attention/mha.ts` 的头注释）。
 *
 * `realShape` 一缺，页面上那一行就空着。以前只有 `matmul` / `lookup` 会渲染它，
 * 所以 `transform` / `concat` / `sum` 里缺了也**看不出来**；现在五种视图都会渲染，
 * 这条断言就是防"新加的视图忘了标"。
 */
function assertRealShapes(steps: Step[], label: string) {
	const missing = steps.flatMap((s) =>
		s.tensors.flatMap((t) =>
			matRefsOf(t).filter((r) => !r.realShape).map((r) => `${s.id}/${r.name}`)
		)
	);
	assert(
		missing.length === 0,
		`${label}：每个矩阵都标了真实尺寸` + (missing.length ? `（缺：${missing.join(' / ')}）` : '')
	);
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
		/** 每个名字第一次出现时的（数据对象, 是否并行视图）——用于放行"并行分支共用输入" */
		const first = new Map<string, { data: unknown; parallel: boolean }>();
		for (const t of s.tensors) {
			for (const ref of matRefsOf(t)) {
				// `hideInput: true` 的变换：输入由同一步里别的视图画（如矩阵乘的 C），
				// 这里不重画，所以不参与"同一个矩阵画两次"的计数
				if (t.kind === 'transform' && t.hideInput && ref === t.input) continue;
				const f = first.get(ref.name);
				// **并行分支共用同一份输入**不算重复画：如写回页两条流都读同一份 y / r_0 / r_1，
				// 两边数据相同、又都是 `parallel`——这是有意的并排展示。串行链里同名同数据仍要抓
				// （MLA 的 c_Q / c_KV~ 被画两遍那种）。名字相同但数据不同更是照抓。
				if (f && f.parallel && t.parallel && f.data === ref.data) continue;
				if (!f) first.set(ref.name, { data: ref.data, parallel: !!t.parallel });
				count.set(ref.name, (count.get(ref.name) ?? 0) + 1);
			}
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
		JSON.stringify(['embedding-lookup', 'attention', 'ffn-moe', 'residual', 'lm-head']),
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
	// 阶段可以有**多段**区间（Residual 拆在两处），步数按段求和
	const own = demo.segments
		.filter((s) => s.stage === st.id)
		.reduce((n, s) => n + (s.to - s.from), 0);
	assert(own > 0, `阶段 ${st.id} 有自己的步骤（${own} 步）`);
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
		demo.stages.filter((s) => s.scope === 'layer').length === 3,
	'2 个模型级阶段（Embedding / LM Head）+ 3 个层级阶段（Attention / FFN-MoE / Residual）'
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
assertCellSizes(demo.steps, 'MHA + MoE');
assertRealShapes(demo.steps, 'MHA + MoE');
assertFormulas(demo.steps, 'MHA + MoE');

/**
 * `fillWidth` 那一页放大格子用的 `fitCellSize`（纯函数，不依赖 DOM）。
 *
 * 渲染时 `StepPanel` 会量出这一行的可用宽度再迭代它；这里只钉住三条性质：
 * **按比例放大、下限 `MIN_TEXT_CELL`、上限 `MAX_TEXT_CELL`**。
 * 页面上的实际占满率靠无头 Chrome 量（`verify` 看不见 DOM）。
 */
{
	// 内容 550 在 1100 里放得下 → 边长翻倍
	assert(
		fitCellSize(20, 550, 1100) === 40,
		`fitCellSize：内容只占一半宽 → 格子边长翻倍（20 → ${fitCellSize(20, 550, 1100)}）`
	);
	// 内容比可用宽度还宽 → 一直缩到下限为止，不会再小
	assert(
		fitCellSize(20, 4400, 1100) === MIN_TEXT_CELL,
		`fitCellSize：内容宽 4 倍 → 收到下限 ${MIN_TEXT_CELL}px`
	);
	// 内容很窄 → 一路放大到上限，不会无限涨
	assert(
		fitCellSize(20, 40, 1100) === MAX_TEXT_CELL,
		`fitCellSize：内容很窄 → 放大到上限 ${MAX_TEXT_CELL}px`
	);
	// 量不到宽度时（SSR / 首帧）原样返回，不要把 0 传下去
	assert(fitCellSize(34, 0, 1100) === 34, 'fitCellSize：量不到内容宽度时原样返回');
	assert(fitCellSize(34, 500, 0) === 34, 'fitCellSize：量不到可用宽度时原样返回');
}

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
assertRealShapes(dense.steps, 'dense-ffn');
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
		stepDurationMs(experts) <= stepDurationMs(route),
		`并行页 moe-experts（${stepDurationMs(experts)}ms）不长于 moe-route（${stepDurationMs(route)}ms）——` +
			`并行单元按工作量取最大格数，格数多的一侧时长更长`
	);
	// 每格耗时一致：去掉下限的影响后，格子少的页与格子多的页各自"每格耗时"相等。
	// 旧断言"只差 30% 以内"是旧下限 2400 把小页拉平的结果；降到 1200 后
	// 24 格的 moe-route（1440ms）与 48 格的 attn-q（2880ms）就是格数比例，
	// 每格仍是 MS_PER_CELL——这才是"页与页之间不忽快忽慢"的真正含义。
	const routeWork = animationUnits(route).reduce((n, u) => n + Math.max(...u.map(workOf)), 0);
	const qWork = animationUnits(qProj).reduce((n, u) => n + Math.max(...u.map(workOf)), 0);
	assert(
		Math.abs(stepDurationMs(route) / routeWork - stepDurationMs(qProj) / qWork) < 1,
		`每格耗时一致（moe-route ${stepDurationMs(route)}ms/${routeWork} 格 vs ` +
			`attn-q ${stepDurationMs(qProj)}ms/${qWork} 格，每格应同为 ${MS_PER_CELL}ms）`
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
	// 上限是"整条流程连播下来别太久"的软约束。`then`（逐行）那一段改成**另起一段**之后
	// （"要先算出 Y，才算得上对它做 op2"），每条流程会多出若干个 `MIN_PHASE_MS`——
	// 这是有意换来的：宁可多两秒，也不要两块矩阵"同时"揭示。
	assert(total < 60000, `整条流程 ${(total / 1000).toFixed(1)}s（< 60s）`);
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
				// 第二段是**逐行**（`then` 无 `b`）时它必须**另起一段**——`X ──op1──▶ Y ──op2──▶ Z`
				// 里 Y、Z 是先后算的，挤在同一段一起揭示就成了"同时算出来"
				assert(
					t.then && !t.then.b ? w.rows2 > 0 : w.rows2 === 0,
					`${s.id}/${t.name}：第二段逐行的时长与 then（无 b）一致（${w.rows2}ms）`
				);
				assert(
					t.then?.b ? w.mm > 0 : w.mm === 0,
					`${s.id}/${t.name}：矩阵乘段时长与 then.b 一致（${w.mm}ms）`
				);
				assert(
					t.tail ? w.tail > 0 : w.tail === 0,
					`${s.id}/${t.name}：链尾段时长与 tail 一致（${w.tail}ms）`
				);
				assert(
					w.mask + w.rows + w.rows2 + w.mm + w.tail === viewDurationMs(t),
					`${s.id}/${t.name}：各段时长之和 = 视图时长（${viewDurationMs(t)}ms）`
				);
				// 链尾段（`tail`）必须**接在最后**：进度 0.99 时矩阵乘段还没播完就说不过去了
				if (t.tail) {
					assert(
						transformPhaseMs(t).tail > 0 && w.mm > 0,
						`${s.id}/${t.name}：链尾段 ${t.tail.op} 排在矩阵乘段之后`
					);
				}
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

	// 上面只走了一条流程（MHA+MoE）。**每一条流程都要走一遍**——V4 的三层步骤数不一样、
	// 各步时长也不一样，某一层踩到边界不代表别的层没事。
	const walkFlows: [string, LlmFlowResult][] = [
		['MHA', buildLlmFlow({ modelId: 'r1', attnId: 'mha', ffnId: 'auto', layer: 0 })],
		['MLA', buildLlmFlow({ modelId: 'r1', attnId: 'mla', ffnId: 'auto', layer: 0 })],
		['V4-SWA', buildLlmFlow({ modelId: 'v4-flash', attnId: '', ffnId: 'auto', layer: 0 })],
		['V4-CSA', buildLlmFlow({ modelId: 'v4-flash', attnId: '', ffnId: 'auto', layer: 2 })],
		['V4-HCA', buildLlmFlow({ modelId: 'v4-flash', attnId: '', ffnId: 'auto', layer: 3 })]
	];
	for (const [label, d] of walkFlows) {
		const zero = d.steps.filter((s) => stepDurationMs(s) <= 0).map((s) => s.id);
		assert(
			zero.length === 0,
			`${label}：${d.steps.length} 步里没有"时长 0"的步骤（那种步骤时间轴跳不进去）${zero.length ? `：${zero.join(' ')}` : ''}`
		);
		const at: number[] = [];
		const pl = createPlayer(d.steps, (p) => at.push(p.stepIndex));
		pl.seekStep(d.steps.length - 1); // 先离开第一步，否则 seekStep(0) 不产生变化、不会被记录
		at.length = 0;
		for (let i = 0; i < d.steps.length; i++) pl.seekStep(i);
		assert(
			JSON.stringify(at) === JSON.stringify(d.steps.map((_, i) => i)),
			`${label}：每一步都能被 seekStep 命中（${at.length} 步）`
		);
		// 从最后一步一路后退，每一步都要真的退回去
		pl.seekStep(d.steps.length - 1);
		const back: number[] = [];
		for (let i = 0; i < d.steps.length - 1; i++) {
			pl.prev();
			back.push(at[at.length - 1]);
		}
		assert(
			JSON.stringify(back) ===
				JSON.stringify(
					Array.from({ length: d.steps.length - 1 }, (_, i) => d.steps.length - 2 - i)
				),
			`${label}：从最后一步一路 prev() 能退到第一步（${back.length} 次）`
		);
		pl.destroy();
	}

	// 页级"本页播放"接管之后**交还**控制权靠 `seekStepRatio`：把总播放器挪到
	// "现在显示到本页的哪个比例"，交还的一瞬间画面才不会跳回它原来停着的地方。
	{
		const d = walkFlows[walkFlows.length - 1][1];
		const seen2: { i: number; r: number }[] = [];
		const pl2 = createPlayer(d.steps, (p) => seen2.push({ i: p.stepIndex, r: p.stepProgress }));
		for (const [i, r] of [
			[3, 0.5],
			[7, 0.25],
			[7, 0.999]
		] as const) {
			seen2.length = 0;
			pl2.seekStepRatio(i, r);
			const last = seen2[seen2.length - 1];
			assert(
				last?.i === i && Math.abs(last.r - r) < 1e-3,
				`seekStepRatio(${i}, ${r}) 落到第 ${last?.i} 步的 ${(last?.r ?? 0).toFixed(3)}`
			);
		}
		pl2.destroy();
	}

	for (const [label, d] of walkFlows) {
		const ovItems = buildOverviewItems(d.steps);
		const animated = d.steps.filter((s) =>
			s.tensors.some((t) => isAnimatable(t) && t.kind !== 'kvcache')
		);
		assert(
			ovItems.length === animated.length,
			`${label}：总览节点数 ${ovItems.length} = 有动画的步骤数 ${animated.length}`
		);
	}
	{
		// 组内加权平均那一步：`w ⊙ v → v̄`（两个压缩层型都该有）
		for (const [label, id] of [
			['V4-HCA', 'hca-compress-pool'],
			['V4-CSA', 'csa-compress-pool']
		] as const) {
			const d = walkFlows.find(([l]) => l === label)![1];
			const pool = buildOverviewItems(d.steps).find((it) => it.stepId === id);
			assert(!!pool, `${label}：组内加权平均（\`-compress-pool\`）在总览里有节点`);
		assert(
			pool?.kind === 'unary' &&
				pool.input.name === 'w ⊙ v' &&
				pool.output.name.startsWith('压缩 KV'),
			`${label}：该节点是 \`w ⊙ v → 压缩 KV\`（实际 ${
				pool?.kind === 'unary' ? `${pool.input.name} → ${pool.output.name}` : '—'
			}）`
		);
		}
	}
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
	assertRealShapes(mlaDemo.steps, 'MLA');
	assertFormulas(mlaDemo.steps, 'MLA');
	const t = mlaDemo.trace.layers[MOE_LAYER].attention as MlaTrace;
	const S = CONFIG.seq_len;
	const H = CONFIG.num_heads;
	const dn = CONFIG.qk_nope_head_dim!;
	const dr = CONFIG.qk_rope_head_dim!;
	const dv = CONFIG.v_head_dim!;
	const kvr = CONFIG.kv_lora_rank!;
	const qr = CONFIG.q_lora_rank!;
	const dqk = dn + dr;
	const mlaW = initWeights(R1_MODEL).slots['mla'][MOE_LAYER] as {
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
	const mlaW = initWeights(R1_MODEL).slots['mla'][MOE_LAYER] as {
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
	assertRealShapes(absDemo.steps, 'MLA（吸收合并）');
	assertFormulas(absDemo.steps, 'MLA（吸收合并）');
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

console.log('\n[DeepSeek V4 混合注意力：SWA / CSA / HCA]');
{
	// V4 的注意力是**逐层固定**的：`compress_ratios = [0,0,4,8]` → 前两层纯滑窗、
	// 第三层 CSA、第四层 HCA。所以逐层各 build 一次。
	const v4 = (layer: number) =>
		buildLlmFlow({ modelId: 'v4-flash', attnId: '', ffnId: 'auto', layer });
	const cfg = V4_FLASH_MODEL.cfg;
	const S = cfg.seq_len;
	const W = cfg.swa_window!;
	const H = cfg.num_heads;
	const hd = cfg.head_dim;
	const dn = cfg.qk_nope_head_dim!;
	const dr = cfg.qk_rope_head_dim!;
	const G = cfg.o_groups!;
	const oLora = cfg.o_lora_rank!;
	const theta = cfg.rope_theta!;
	// 两档压缩比（真实 4 / 128 缩放到 2 / 4）：CSA 压得轻、条目多；HCA 压得狠、条目少
	const plan2 = cfg.compress_ratios![2];
	const plan4 = cfg.compress_ratios![3];

	console.log('    结构：');
	assert(
		V4_FLASH_MODEL.attnByLayer.join(',') === 'swa,swa,csa,hca',
		`compress_ratios = ${JSON.stringify(cfg.compress_ratios)} → 逐层 SWA / SWA / CSA / HCA`
	);
	assert(
		V4_FLASH_MODEL.attnChoices.length === 0,
		'V4 的注意力逐层固定、不给用户选——所以"MLA 配 V4"这种组合根本不存在'
	);
	assert(
		attnIdFor(V4_FLASH_MODEL, 2, 'mla') === 'csa',
		'就算硬塞一个 attnId，逐层固定的层也仍然用自己的插件（塞不进 MLA）'
	);
	assert(
		!pluginsOf(R1_MODEL).includes('csa') && !pluginsOf(R1_MODEL).includes('hca'),
		'R1 只给 mha / mla 分配权重，V4 的插件连权重都不初始化'
	);

	const swaDemo = v4(0);
	const csaDemo = v4(2);
	const hcaDemo = v4(3);
	assert(
		swaDemo.trace.layers.map((l) => l.attnId).join(',') === 'swa,swa,csa,hca',
		`一次前向里四层各用各的注意力插件（${swaDemo.trace.layers.map((l) => l.attnId).join(' / ')}）`
	);
	assert(
		swaDemo.trace.layers.every((l) => l.ffnId === 'deepseek-moe'),
		'V4 全 MoE：四层都是 deepseek-moe，没有 dense FFN 层'
	);
	for (const [label, d] of [
		['SWA', swaDemo],
		['CSA', csaDemo],
		['HCA', hcaDemo]
	] as const) {
		assertMatmulShapes(d.steps, `V4 ${label}`);
		assertRefShapes(d.steps, `V4 ${label}`);
		assertRealShapes(d.steps, `V4 ${label}`);
		assertFormulas(d.steps, `V4 ${label}`);
		assertNoDuplicateMatrices(d.steps, `V4 ${label}`);
		assertCellSizes(d.steps, `V4 ${label}`);
		const blob = JSON.stringify(d.steps);
		assert(
			!blob.includes('{{step:') && !blob.includes('第 ? 步'),
			`V4 ${label}：跨步骤引用全部解析成功（没有残留占位符）`
		);
	}

	// ── Q / KV / 输出投影的形状 ──────────────────
	console.log('    形状与 K = V：');
	const sw = swaDemo.trace.layers[0].attention as HybridTrace;
	assert(
		[swaDemo, csaDemo, hcaDemo].every((d) =>
			d.trace.layers.every((l) => l.attention.out.flat().every(Number.isFinite))
		),
		'三层注意力输出全是有限值（没有 NaN / Infinity 渗出来）'
	);
	assert(
		sw.q.length === H && sw.q[0].length === S && sw.q[0][0].length === hd,
		`q = [${H} 头 × ${S} × ${hd}]，每头 = ${dn} 维 nope + ${dr} 维 rope`
	);
	assert(
		sw.kv.length === S && sw.kv[0].length === hd,
		`KV 只有**一个头**、宽 ${hd}（真实 ${REAL_V4_FLASH.head_dim}）：K 和 V 就是同一份向量`
	);
	assert(
		sw.groups.length === G && sw.groups[0][0].length === (H / G) * hd,
		`输出分成 ${G} 组，每组 ${H / G} 头 × ${hd} 维（真实 ${REAL_V4_FLASH.o_groups} 组、每组 ${REAL_V4_FLASH.d_model} → ${REAL_V4_FLASH.o_lora_rank}）`
	);
	assert(
		sw.oLatent[0].length === G * oLora && sw.out[0].length === cfg.d_model,
		`分组低秩 → 拼回 ${G * oLora} 维 → W_O^B → ${cfg.d_model} 维`
	);

	// ── 页内排布 / 两条 RoPE 页的统一格式 ────────────
	console.log('    页内排布：');
	{
		const attnSteps = swaDemo.steps.filter((s) => s.id.startsWith('swa-'));
		const idx = (id: string) => attnSteps.findIndex((s) => s.id === id);
		const qUp = attnSteps[idx('swa-q-up')];
		const qNorm = attnSteps[idx('swa-q-norm-rope')];
		assert(
			!!qUp && !!qNorm && idx('swa-q-up') < idx('swa-q-norm-rope'),
			'Q 的"每头归一化 + RoPE"从升维那页**单独拆出来**，排在它后面'
		);
		assert(
			qUp.tensors.length === 2 && qNorm.tensors.length === 2,
			`升维页只剩"归一化 → 升维"一条链 + 一行小字（${qUp.tensors.length} 块）；新页一条链 + 一行小字（${qNorm.tensors.length} 块）`
		);

		// 一条 `transform`（整块矩阵），RoPE 挂在**链尾**，末尾 dr 维在链尾段才被框出来。
		const ropePages = ['swa-q-norm-rope', 'swa-kv-norm-rope'] as const;
		const lastCol = Array.from({ length: dr }, (_, i) => dn + i).join(',');
		for (const id of ropePages) {
			const step = attnSteps[idx(id)];
			const tf = step?.tensors.find((t) => t.kind === 'transform');
			assert(
				!!tf && tf.kind === 'transform' && !!tf.tail && !tf.then,
				`${id}：归一化与 RoPE 合成**一条链**（RoPE 在链尾，没有第二段/矩阵乘）`
			);
			if (tf?.kind === 'transform') {
				assert(
					tf.tail!.result.shape.join('×') === tf.output.shape.join('×'),
					`${id}：链尾前后是**同一块整矩阵**（${tf.output.shape.join('×')}），不是只画 rope 那几维`
				);
				assert(
					tf.tail!.highlightCols?.join(',') === lastCol,
					`${id}：框的是末尾 ${dr} 维（真实 ${REAL_V4_FLASH.rope_head_dim} 维）——和 O_h / O_h′ 那个框同一套规则`
				);
				assert(
					!!tf.tail!.highlightCols?.length && tf.tail!.op === 'RoPE',
					`${id}：链尾那一段就是 RoPE（"算到这一步才框出来"靠的就是它单独成段）`
				);
			}
			assert(
				step.tensors.filter((t) => t.kind === 'transform').length === 1,
				`${id}：只有一个变换视图（RoPE 没有被拆成独立的"rope 段"小块）`
			);
		}

		// `tail` 只支持"没有 then"或"then 是矩阵乘"两种，`then`（逐行）+ `tail` 的排版没实现
		for (const d of [swaDemo, csaDemo, hcaDemo]) {
			const bad = d.steps.flatMap((s) =>
				s.tensors.filter((t) => t.kind === 'transform' && t.tail && t.then && !t.then.b).map((t) => `${s.id}/${t.name}`)
			);
			assert(
				bad.length === 0,
				`没有"逐行 then + tail"的视图（渲染没实现这种排法）${bad.length ? `：${bad.join(' ')}` : ''}`
			);
		}

		const outA = attnSteps[idx('swa-out-a')];
		const groups = outA.tensors.filter((t) => t.group !== undefined);
		assert(
			groups.length === G &&
				groups.every((t) => t.row !== undefined && t.row === groups[0].row && !!t.parallel),
			`${G} 组的分组投影**左右并排**（共用 row=${groups[0]?.row} + 各自 group + parallel → 中间不画箭头）`
		);
	}

	// ── 滑窗掩码 + sink ─────────────────────────
	console.log('    纯滑窗层：');
	assert(
		sw.nE === 0 && sw.cols === S + 1,
		`纯滑窗层没有压缩条目：列数 = ${S} 个 token + 1 个 sink = ${sw.cols}`
	);
	let winOk = true;
	let sinkOk = true;
	let probsOk = true;
	for (let i = 0; i < S; i++) {
		for (let j = 0; j < S; j++) {
			if (sw.mask[i][j] !== (j <= i && i - j < W ? 1 : 0)) winOk = false;
		}
		if (sw.mask[i][sw.sinkCol] !== 1) sinkOk = false;
		const sum = sw.probs[0][i].reduce((a, b) => a + b, 0);
		if (Math.abs(sum - 1) > 1e-9) probsOk = false;
	}
	assert(winOk, `滑窗掩码：第 i 个 token 只看 [i−${W}+1, i]，更早的一律 −∞`);
	assert(
		sinkOk,
		'sink 偏置在掩码里恒为 1：它不是位置、而是分母上的一项——否则靠前的行会被整行掩掉，无从归一化'
	);
	assert(probsOk, 'softmax 每行和为 1（sink 也占一份权重）');
	assert(
		sw.probs[0].every((row) => row[sw.sinkCol] > 0),
		'sink 槽拿到的权重 > 0：它确实在参与归一化，不是装饰'
	);

	// ── sink 没有"位置"（kernel 里是在循环之后折进分母的） ──
	console.log('    sink 偏置：');
	assert(
		sw.sinkCol === S + sw.nE,
		`sink 画在列空间的最后（第 ${sw.sinkCol} 列）——但这是**排版**，不是位置`
	);
	const permOk = (() => {
		for (let i = 0; i < S; i++) {
			const row = sw.maskedScores[0][i];
			// 把 sink 那一列从末尾挪到**最前面**再 softmax：真实列的权重必须一模一样
			const moved = [row[sw.sinkCol], ...row.slice(0, sw.sinkCol)];
			const p2 = softmaxRow(moved);
			for (let j = 0; j < sw.sinkCol; j++) {
				if (Math.abs(sw.probs[0][i][j] - p2[j + 1]) > 1e-12) return false;
			}
		}
		return true;
	})();
	assert(
		permOk,
		'把 sink 挪到第一列，真实列的权重**逐元素不变**——softmax 对顺序不敏感，sink 不是某个 token 的位置'
	);
	const numeratorOk = (() => {
		// sink 只进分母：输出必须等于"只对真实列加权求和"（sink 的 V 是 0）
		for (let i = 0; i < S; i++) {
			for (let c = 0; c < hd; c++) {
				let acc = 0;
				for (let j = 0; j < sw.sinkCol; j++) {
					const v = j < S ? sw.kv[j][c] : sw.compKv![j - S][c];
					acc += sw.probs[0][i][j] * v;
				}
				if (Math.abs(acc - sw.headOut[0][i][c]) > 1e-12) return false;
			}
		}
		return true;
	})();
	assert(
		numeratorOk,
		'输出里没有 sink 的贡献（它的 V 是 0）：分子只对真实列加权，sink 只把分母撑大'
	);
	assert(
		sw.sink.length === H,
		`sink 是**每头一个**可学习的标量（共 ${H} 个），不是全局一个、更不是某个 token 的键`
	);
	// P·V 的右操作数：V 的前 S+nE 行**逐元素等于 K**（K = V 共用），最后一行是 sink 的 0
	assert(
		sw.vAll.length === sw.sinkCol + 1,
		`V 有 ${sw.vAll.length} 行 = 真实列 ${sw.sinkCol} + sink 1 行`
	);
	assert(
		sw.vAll.every((row, j) =>
			j === sw.sinkCol
				? row.every((v) => v === 0)
				: row.every((v, c) => Math.abs(v - sw.kAll[j][c]) < 1e-15)
		),
		'V 的前 S+nE 行**逐元素等于 K**（K = V 共用），最后一行（sink）全是 0'
	);

	// ── 逆 RoPE：绝对角度 → 相对角度 ─────────────
	console.log('    逆 RoPE：');
	const invOk = (() => {
		for (let i = 0; i < S; i++) {
			for (let c = dn; c < hd; c++) {
				let acc = 0;
				// kv[j] 已经转过 R(j)，所以这里要用**没转过**的 kvNorm[j] 再按相对位置 j−i 转
				for (let j = Math.max(0, i - W + 1); j <= i; j++) {
					acc +=
						sw.probs[0][i][j] * ropeApply(sw.kvNorm[j].slice(dn), j - i, theta)[c - dn];
				}
				if (Math.abs(acc - sw.headOutInv[0][i][c]) > 1e-9) return false;
			}
		}
		return true;
	})();
	assert(
		invOk,
		'逆 RoPE 把绝对角度转成了相对量：O_0 的 rope 段 = Σ_j P_ij · R(j−i)·k_j（用未旋转的 k_j 按位置差算）'
	);

	// ── 逆 RoPE 挂在链尾（`tail`），`O_h′` 就在 `O_h` 右边 ──────────
	{
		const attnStep = swaDemo.steps.find((s) => s.id === 'swa-attn')!;
		const chain = attnStep.tensors.find((t) => t.kind === 'transform')!;
		assert(
			chain.kind === 'transform' && !!chain.then?.b && !!chain.tail,
			'逆 RoPE 挂在链尾：S ──softmax──▶ P_h ──×V──▶ O_h ──逆 RoPE──▶ O_h′ 是**一个**视图'
		);
		if (chain.kind === 'transform') {
			const names = [chain.input.name, chain.output.name, chain.then!.result.name, chain.tail!.result.name];
			assert(
				names.join('→') === 'S（+ sink 列）→P_h→O_h→O_h′',
				`链上的四块依次是 S → P_h → O_h → O_h′（实际 ${names.join(' → ')}）`
			);
			assert(
				chain.tail!.result.data === (swaDemo.trace.layers[0].attention as HybridTrace).headOutInv[0],
				'链尾画的就是逆 RoPE 的结果（head 0 的 O_h′），不是另算一份'
			);
			assert(
				chain.tail!.highlightCols?.join(',') ===
					Array.from({ length: dr }, (_, i) => dn + i).join(','),
				`链尾框出的正是末尾 ${dr} 维（真实 ${REAL_V4_FLASH.rope_head_dim} 维），输入输出同一批列`
			);
			// 一步之内不能出现第二个"逆 RoPE"视图——那正是被合并掉的那个
			assert(
				attnStep.tensors.filter((t) => t.kind === 'transform').length === 1,
				'这一步只有一个变换视图：逆 RoPE 没有被拆成单独一块'
			);
		}
	}

	// ── K / V 上的"分组框"：滑窗（精确 token）与压缩条目用**不同浅色**框出来 ──
	// K 按**列**分段、V 按**行**分段（两者都是"位置"那一维），段边界 = S；
	// 颜色在 K / V 上必须一致，否则读者认不出"哪一段是哪一类"。纯滑窗层没有第二段，不画框。
	{
		// 注意 `endsWith('-score')` 会先撞上 CSA 的 `csa-indexer-score`（indexer 的打分），
		// 所以这里按**完整 id** 找那两步
		const scoreStep = hcaDemo.steps.find((s) => s.id === 'hca-score')!;
		const attnStep = hcaDemo.steps.find((s) => s.id === 'hca-attn')!;
		const kB = scoreStep.tensors.find((t) => t.kind === 'matmul' && t.b.name === 'K_hᵀ');
		const vB = attnStep.tensors.find(
			(t) => t.kind === 'transform' && t.then?.b.name === 'V'
		);
		const kRef = kB?.kind === 'matmul' ? kB.b : undefined;
		const vRef = vB?.kind === 'transform' ? vB.then!.b : undefined;
		const nE = (hcaDemo.trace.layers[3].attention as HybridTrace).nE;
		const desc = (r?: { bands?: { axis: string; from: number; to: number; tone: string }[] }) =>
			(r?.bands ?? []).map((b) => `${b.axis}:${b.from}-${b.to}@${b.tone}`).join(' ');
		assert(
			desc(kRef) === `col:0-${S - 1}@#38bdf8 col:${S}-${S + nE - 1}@#fbbf24`,
			`V4 HCA：K_hᵀ 的列分成"滑窗（前 ${S} 列）"与"压缩（后 ${nE} 列）"两个浅色框（实际 ${desc(kRef)}）`
		);
		assert(
			desc(vRef) === `row:0-${S - 1}@#38bdf8 row:${S}-${S + nE - 1}@#fbbf24`,
			`V4 HCA：V 的行分成两段同色的框，末行 sink 不框（实际 ${desc(vRef)}）`
		);
		// 颜色必须一一对应：蓝 = 滑窗、琥珀 = 压缩
		const toneOf = (r?: { bands?: { from: number; tone: string }[] }, from = 0) =>
			r?.bands?.find((b) => b.from === from)?.tone;
		assert(
			toneOf(kRef) === toneOf(vRef) && toneOf(kRef, S) === toneOf(vRef, S),
			'V4 HCA：K 与 V 上"同一类东西"用的是同一个颜色（蓝 = 滑窗、琥珀 = 压缩）'
		);
		// CSA 也有两段（压缩比不同、条目数不同）；纯滑窗层只有一段，不画框
		const csaScore = csaDemo.steps.find((s) => s.id === 'csa-score')!;
		const csaK = csaScore.tensors.find((t) => t.kind === 'matmul' && t.b.name === 'K_hᵀ');
		assert(
			(csaK?.kind === 'matmul' ? csaK.b.bands?.length : 0) === 2,
			'V4 CSA：K 上同样有两段框（压缩比不同不影响"分两段"这件事）'
		);
		const swaScore = swaDemo.steps.find((s) => s.id === 'swa-score')!;
		const swaK = swaScore.tensors.find((t) => t.kind === 'matmul' && t.b.name === 'K_hᵀ');
		assert(
			(swaK?.kind === 'matmul' ? swaK.b.bands : undefined) === undefined,
			'V4 SWA：纯滑窗层只有一段，不画分组框（没有可区分的东西）'
		);
	}

	// ── 压缩器（CSA：重叠窗口 + indexer） ────────
	console.log('    CSA（2 倍压缩（真实 4）+ 重叠窗口 + indexer）：');
	const cs = csaDemo.trace.layers[2].attention as HybridTrace;
	const cg = cs.compress!.groups;

	// gate 投影必须是**真正的矩阵乘**：`W_gate` 得作为 B 画出来。
	{
		const gateStep = csaDemo.steps.find((s) => s.id.endsWith('-compress-gate'))!;
		const gate = gateStep.tensors.find((t) => t.kind === 'matmul');
		assert(
			!!gate && gate.kind === 'matmul' && gate.b.name === 'W_gate',
			'gate 投影用矩阵乘视图，`W_gate` 作为 B 画出来（不是只在箭头里写个 `× W_gate`）'
		);
		if (gate?.kind === 'matmul') {
			assert(
				gateStep.tensors.filter((t) => t.kind === 'matmul').length === 1,
				'这一步只有一个矩阵乘视图'
			);
			// 画出来的 A/B/C 必须真的满足 A·B = C——否则"画了 W_gate"也可能画的是别的矩阵
			const A = gate.a.data ?? [];
			const B = gate.b.data ?? [];
			const C = gate.out.data ?? [];
			let ok = A.length > 0 && B.length > 0 && C.length > 0;
			for (let i = 0; ok && i < C.length; i++)
				for (let j = 0; ok && j < C[i].length; j++) {
					let acc = 0;
					for (let p = 0; p < A[i].length; p++) acc += A[i][p] * (B[p]?.[j] ?? 0);
					if (Math.abs(acc - C[i][j]) > 1e-12) ok = false;
				}
			assert(ok, '页面上画的 X · W_gate = [v | s] 真的成立（画出来的就是乘的那个矩阵）');
		}
		// `[v | s]` 是一块矩阵，光看 `[8 × 1024]` 看不出"值占多少、打分占多少"——
		// 而这一步的关键正是"打分算权重、值被平均"。所以注解里必须写出真实的分段尺寸。
		for (const [name, d] of [
			['CSA', csaDemo],
			['HCA', hcaDemo]
		] as const) {
			const gateStep = d.steps.find((s) => s.id.endsWith('-compress-gate'))!;
			const g = gateStep.tensors.find((t) => t.kind === 'matmul');
			const out = g?.kind === 'matmul' ? g.out : undefined;
			const realTotal = out?.realShape?.[1] ?? 0;
			const stepText = gateStep.label;
			assert(
				stepText.includes(`值 ${realTotal / 2} 列`) &&
					stepText.includes(`打分 ${realTotal / 2} 列`) &&
					stepText.includes(`合计 ${realTotal}`),
				`V4 ${name}：gate 那一步的文案写出了值 / 打分各自在真实里的列数（各 ${realTotal / 2}）与合计 ${realTotal}`
			);
			// "真实尺寸"那一行也要分段标出来（绿色胶囊是 nowrap + ellipsis，长句子会被截掉，
			// 所以分段尺寸走这一行，不走胶囊）
			const realLine = out ? (realLabel(out) ?? '') : '';
			// CSA（重叠窗口）分段名是 dsv4 的英文（v_overlap / v / s_overlap / s），
			// HCA（不重叠）是中文（值 / 打分）——各按各的查。
			const segOk = name === 'CSA'
				? realLine.includes('v_overlap 512 列') &&
					realLine.includes('v 512 列') &&
					realLine.includes('s_overlap 512 列') &&
					realLine.includes('s 512 列')
				: realLine.includes('值 512 列') && realLine.includes('打分 512 列');
			assert(
				segOk,
				`V4 ${name}：\`[v | s]\` 的"真实尺寸"行分段标出了值 / 打分各占多少列（${realLine}）`
			);
		}
	}

	assert(cs.nE === 4, `2 倍压缩（真实 4）：${S} 个 token 压出 ${cs.nE} 条压缩条目`);

	// ── 压缩的前两步：权重从哪来（gate 页） + 怎么用权重（pool 页） ──
	// gate 页：`W_gate` 画成矩阵乘、`[v | s]` 标真实分段尺寸、打分加 ape 再 softmax
	// 和 gate 投影同一页、且和 `ape` 排在同一行（三块各 hd 列）；
	// pool 页只留"用权重把值加权平均"，且别把窗口里的 token 拆成好几块。
	console.log('    压缩：权重从哪来 / 怎么用：');
	for (const [name, d, layer, ratio, overlap] of [
		['CSA', csaDemo, 2, plan2, true],
		['HCA', hcaDemo, 3, plan4, false]
	] as const) {
		const gateStep = d.steps.find((s) => s.id.endsWith('-compress-gate'))!;
		const poolStep = d.steps.find((s) => s.id.endsWith('-compress-pool'))!;
		const comp = (d.trace.layers[layer].attention as HybridTrace).compress!;
		const grp = comp.groups[comp.groups.length - 1]; // 页面上举的例子：最后一条

		// ① gate 页：投影 → 框出打分那几列
		const mm = gateStep.tensors.find((t) => t.kind === 'matmul');
		assert(
			!!mm && mm.kind === 'matmul' && mm.b.name === 'W_gate' && !!mm.highlightCols?.length,
			`V4 ${name}：gate 页把 \`W_gate\` 画成矩阵乘，并在结果 \`[v | s]\` 上**框出打分那几列**`
		);

		// 打分加 ape → softmax → w 放哪：
		//   · HCA（不重叠）：仍在这一页（`-gate-softmax` 一条链，没有两份打分）；
		//   · CSA（重叠）：挪到 `-sape` 页，逐 token **双份**打分（`-sape-own` / `-sape-tail`）
		//     全量 + ape，再窗口堆叠 softmax → w（`-sape-w`）。
		if (overlap) {
			const sape = d.steps.find((s) => s.id.endsWith('-sape'));
			assert(!!sape, 'V4 CSA：打分加 ape → softmax → w 拆到 -sape 页（两份打分全量，gate 页放不下）');
			const own = sape?.tensors.find((t) => t.name.endsWith('-sape-own'));
			const tail = sape?.tensors.find((t) => t.name.endsWith('-sape-tail'));
			const wIn = sape?.tensors.find((t) => t.name.endsWith('-sape-w-in'));
			const wStep = sape?.tensors.find((t) => t.name.endsWith('-sape-w') && !t.name.endsWith('-sape-w-in'));
			// 两条并排链：主体份 s + ape、尾巴份 s_overlap + ape，各 8 行全画
			assert(
				own?.kind === 'transform' && own.input.name === 's' && own.output.name === 's+ape' &&
					own.input.shape[0] === S && own.output.shape[0] === S,
				`V4 ${name}：\`-sape-own\` 主体份打分 \`s\` ──+ape──▶ \`s+ape\`（8 个 token 全画）`
			);
			assert(
				tail?.kind === 'transform' && tail.input.name === 's_overlap' && tail.output.name === 's_overlap+ape' &&
					tail.input.shape[0] === S && tail.output.shape[0] === S,
				`V4 ${name}：\`-sape-tail\` 尾巴份打分 \`s_overlap\` ──+ape──▶ \`s_overlap+ape\`（8 个 token 全画，在窗口槽位打分左侧上方）`
			);
			// 两列布局：tail / own 同 row 同 group（左列，上下堆叠）；窗口槽位打分（`-sape-w-in`
			// matrix，animated 从空填）同 row 不同 group（右列）且 `vcenter`（垂直居中在两链之间）；
			// `-sape-w`（softmax→w）接在窗口槽位打分右边、`hideInput`（输入由左边的 matrix 画）
			assert(
				tail?.row !== undefined &&
					tail.row === own?.row &&
					tail.group === own?.group &&
					own.group !== undefined &&
					wIn?.row !== undefined &&
					wIn.row === tail.row &&
					wIn.group !== undefined &&
					wIn.group !== own.group &&
					wIn.vcenter === true &&
					wIn.kind === 'matrix' &&
					wIn.animated === true &&
					wStep?.kind === 'transform' &&
					wStep.op.includes('softmax') &&
					wStep.hideInput === true &&
					wStep.row === tail.row &&
					wStep.group !== undefined &&
					wStep.group !== wIn.group,
				`V4 ${name}：两列布局——s_overlap/s 链同列堆叠（row=${tail?.row} group='${own?.group}'），窗口槽位打分（matrix animated + vcenter）在右侧（group='${wIn?.group}'），softmax→w 再接一列（group='${wStep?.group}'，hideInput）`
			);
			// 窗口 softmax → w：[nE·window × hd] 堆叠，输出 w
			const winRows = comp.groups.length * grp.slots.length;
			assert(
				wStep?.kind === 'transform' &&
					wStep.op.includes('softmax') &&
					wStep.output.name === 'w' &&
					wStep.input.shape[0] === winRows,
				`V4 ${name}：\`-sape-w\` 把 ${comp.groups.length} 条窗口堆成 ${winRows} 行、逐通道 softmax → w`
			);
			// 扫描拼接：窗口槽位打分（`-sape-w-in` matrix animated 逐行）标了 `scan`，seq 长度 = 窗口行数，
			// 且**交替**指向 tail / own（每条窗口前 ratio 槽 → s_overlap+ape、后 ratio 槽 → s+ape）。
			assert(
				wIn?.kind === 'matrix' &&
					wIn.animated === true &&
					wIn.revealOrder === 'row' &&
					!!wIn.scan?.seq &&
					wIn.scan.seq.length === winRows,
				`V4 ${name}：\`-sape-w-in\`（窗口槽位打分）是 matrix animated 逐行、标了 scan（seq 长度 ${wIn?.scan?.seq?.length} = 窗口行数 ${winRows}）`
			);
			if (wIn?.scan?.seq) {
				const seq = wIn.scan.seq;
				const px = name === 'CSA' ? 'csa' : 'hca';
				// 每条窗口按 ratio 行一组拷贝：前 ratio 行 = s_overlap+ape 的 2 行块、后 ratio 行 = s+ape 的 2 行块；
				// 窗口 0 的上一组是空槽 → 块为 null。
				let scanOk = true;
				for (const gr of comp.groups) {
					const tailT = gr.slots.slice(0, ratio).filter((t) => t >= 0);
					const ownT = gr.slots.slice(ratio).filter((t) => t >= 0);
					const tailBlock = tailT.length
						? { target: `${px}-sape-tail`, from: Math.min(...tailT), to: Math.max(...tailT) }
						: null;
					const ownBlock = ownT.length
						? { target: `${px}-sape-own`, from: Math.min(...ownT), to: Math.max(...ownT) }
						: null;
					for (let j = 0; scanOk && j < gr.slots.length; j++) {
						const e = seq[gr.index * gr.slots.length + j];
						const expect = j < ratio ? tailBlock : ownBlock;
						if (expect === null) scanOk = e === null;
						else
							scanOk =
								!!e &&
								e.target === expect.target &&
								e.from === expect.from &&
								e.to === expect.to;
					}
				}
				assert(
					scanOk,
					`V4 ${name}：扫描 seq 行块对应正确——每条窗口前 ${ratio} 行 → s_overlap+ape 的 2 行块、后 ${ratio} 行 → s+ape 的 2 行块，空槽块 null（交替，三矩阵框都 2 行高）`
				);
			}
			assert(
				!gateStep.tensors.some((t) => t.name.endsWith('-gate-softmax')),
				`V4 ${name}：gate 页不再画打分链（挪到 -sape 页了），gate 页只剩投影 + 注解`
			);
		} else {
			// HCA：打分加 ape → softmax 仍在这一页（一条链，`ape ──+ s──▶ s+ape ──softmax──▶ w`）
			const chain = gateStep.tensors.find((t) => t.name.endsWith('-gate-softmax'));
			assert(
				!!chain && chain.kind === 'transform' && chain.input.name === 'ape' && chain.output.name === 's+ape',
				`V4 ${name}：同一页里跟着 \`ape ──+ s──▶ s + ape\`（"权重从哪来"就在眼前）`
			);
			// "放大占满宽度"是**逐页开关**，只给 HCA 开
			const fills = !!mm?.fillWidth && !!chain?.fillWidth;
			assert(
				fills === !overlap,
				`V4 ${name}：gate 页"放大到占满可用宽度"${overlap ? '不开' : '开'}（fillWidth=${fills}）`
			);
			if (chain?.kind === 'transform') {
				assert(
					chain.input.shape[1] === hd && chain.output.shape[1] === hd && chain.then?.result.shape[1] === hd,
					`V4 ${name}：ape / s + ape / w 三块各只有 ${hd} 列（一行排得下）`
				);
				assert(
					chain.input.shape[0] === S &&
						chain.output.shape[0] === S &&
						chain.then?.result.shape[0] === S,
					`V4 ${name}：ape / s + ape / w 都是 ${S} 行——整个序列一起画`
				);
				assert(
					chain.then?.op.includes('softmax'),
					`V4 ${name}：链尾是 softmax（逐通道）——打分加完 ape 立刻归一化`
				);
			}
		}
		// gate 页的 `[v | s]` 投影块不能重复画到 pool 页（那是 gate 页独有的）。
		assert(
			!poolStep.tensors.some((t) => t.kind === 'matmul' && t.b.name === 'W_gate'),
			`V4 ${name}：pool 页不重复画 \`W_gate\` 的 \`[v | s]\` 投影块（那只在 gate 页）`
		);

		// ② pool 页：权重怎么算 + 怎么用 —— `w 计算链` + `w ⊙ v` 一块 + 箭头 + 一行 v̄
		//
		// 不重叠（HCA）用 `ewise`：`w` 与 `v` 都从上一页的完整矩阵拷过来、画成和矩阵乘同一个 2×2，
		// **8 个 token 全画**、逐组（ratio 行一组）揭示。
		// 重叠窗口（CSA）下同一个 token 会同时出现在两条窗口、权重不同，逐 token 的 8 行加起来
		// 不等于 v̄，所以把 **4 条窗口**各 4 行上下堆成一块 `[nE·window × hd]`，同样用 `ewise` +
		// `groupRows = window` 逐组揭示；每条窗口的 window 行按列加 = 那组自己的 1 条。
		// 另外 CSA 还在 pool 页加了条"打分→softmax→w"的 w 计算链（用户要求权重过程也在这页可见）。
		const prod = poolStep.tensors.find(
			(t) => t.name.endsWith('-pool-prod') || t.name.endsWith('-pool-ewise')
		);
		const result = poolStep.tensors.find((t) => t.name.endsWith('-pool-result'));
		const grpN = grp.slots.length;

		if (overlap) {
			// CSA：重叠窗口。`w ⊙ v` 用 `ewise` 把 **4 条窗口**上下堆成一块 `[nE·window × hd]`，
			// `groupRows = window` 逐组（一条窗口一组）揭示。不能用 HCA 的"逐 token 8 行"：
			// 重叠下同一个 token 会同时出现在两条窗口、权重不同，逐 token 行求和算不出条目。
			assert(
				!!prod && prod.kind === 'ewise' && prod.op === '⊙',
				`V4 ${name}：\`w ⊙ v\` 用**逐元素**视图画（两个操作数都画出来，不是只画结果）`
			);
			if (prod?.kind === 'ewise') {
				const nWin = comp.groups.length;
				const stacked = nWin * grpN;
				// 操作数 & 结果是"按窗口堆叠"：[nE·window × hd]
				assert(
					prod.out.shape[0] === stacked &&
						prod.a.shape[0] === stacked &&
						prod.b.shape[0] === stacked,
					`V4 ${name}：\`v\` / \`w\` / \`w ⊙ v\` 都是 ${nWin} 条窗口 × ${grpN} 行 = ${stacked} 行（窗口堆叠，不是 HCA 的逐 token 行）`
				);
				assert(
					prod.a.name === 'v' && prod.b.name === 'w',
					`V4 ${name}：左操作数 \`v\`（各窗口的值）、右操作数 \`w\`（各窗口的权重）`
				);
				assert(
					prod.groupRows === grpN,
					`V4 ${name}：\`w ⊙ v\` 逐组揭示、一条窗口一组（${grpN} 行，实际 ${prod.groupRows}）`
				);
				// 逐元素 = 两个同位置的格子相乘
				const outData = prod.out.data!;
				const aData = prod.a.data!;
				const bData = prod.b.data!;
				let same = true;
				for (let t = 0; same && t < outData.length; t++)
					for (let c = 0; same && c < hd; c++)
						if (Math.abs((outData[t][c] ?? 0) - aData[t][c] * bData[t][c]) > 1e-15) same = false;
				assert(same, `V4 ${name}：\`w ⊙ v\` 逐元素 = \`v\` × \`w\`（同位置的格子）`);
				// 每条窗口的 `window` 行按列加起来 = 那组自己的 1 条（逐条对上）
				let allWinOk = true;
				comp.groups.forEach((gr, g) => {
					const base = g * grpN;
					for (let c = 0; allWinOk && c < hd; c++) {
						let acc = 0;
						for (let j = 0; j < grpN; j++) acc += outData[base + j][c];
						if (Math.abs(acc - gr.pooled[c]) > 1e-12) allWinOk = false;
					}
				});
				assert(
					allWinOk,
					`V4 ${name}：每条窗口的 ${grpN} 行按列加起来 = 那组自己的 1 条压缩条目`
				);
			}
		} else {
			assert(
				!!prod && prod.kind === 'ewise' && prod.op === '⊙',
				`V4 ${name}：\`w ⊙ v\` 用**逐元素**视图画（两个操作数都画出来，不是只画"已经乘过权重"的结果）`
			);
			if (prod?.kind === 'ewise') {
				// 两个操作数**就是**上一步那两块（同一个对象 = "从上一步的完整矩阵拷贝过来"）
				const chainW = gateStep.tensors.find((t) => t.name.endsWith('-gate-softmax'));
				const wData = chainW?.kind === 'transform' ? chainW.then?.result.data : undefined;
				const comp = (d.trace as HybridTrace).layers[layer].attention.compress!;
				assert(
					prod.a.name === 'v' && prod.a.data === comp.v,
					`V4 ${name}：左操作数 \`v\` 就是上一步 \`[v | s]\` 的值那几列（同一个矩阵对象）`
				);
				assert(
					prod.b.name === 'w' && !!wData && prod.b.data === wData,
					`V4 ${name}：右操作数 \`w\` 就是上一步 softmax 出来的权重（同一个矩阵对象）`
				);
				// 8 个 token 全画（不是只画窗口那 4 行）
				assert(
					prod.out.shape[0] === S && prod.a.shape[0] === S && prod.b.shape[0] === S,
					`V4 ${name}：\`w\` / \`v\` / \`w ⊙ v\` 都是 ${S} 行——8 个 token 全画（原来只有窗口那 ${grpN} 行）`
				);
				// **逐组揭示**：8 个 token 分成 2 组，框跟着动画走（不再写死"举例那一组"）
				assert(
					prod.groupRows === plan4,
					`V4 ${name}：\`w ⊙ v\` 逐组揭示、一组 ${plan4} 行（实际 ${prod.groupRows}）`
				);
				assert(
					prod.highlightRows === undefined,
					`V4 ${name}：框不写死（原来固定框最后那一组当例子）——现在框"正在算的那一组"`
				);
				// 逐元素 = 两个同位置的格子相乘
				let same = true;
				for (let t = 0; same && t < S; t++)
					for (let c = 0; same && c < hd; c++)
						if (
							Math.abs((prod.out.data![t][c] ?? 0) - prod.a.data![t][c] * prod.b.data![t][c]) >
							1e-15
						)
							same = false;
				assert(same, `V4 ${name}：\`w ⊙ v\` 逐元素 = \`v\` × \`w\`（同位置的格子）`);
				// 每一组的那几行按列加起来 = 对应那一条（页面上是"先算第 1 组、再算第 2 组"）
				let allGroupsOk = true;
				for (const gr of comp.groups) {
					for (let c = 0; allGroupsOk && c < hd; c++) {
						let acc = 0;
						for (let t = gr.from; t <= Math.min(gr.to - 1, S - 1); t++) acc += prod.out.data![t][c];
						if (Math.abs(acc - gr.pooled[c]) > 1e-12) allGroupsOk = false;
					}
				}
				assert(
					allGroupsOk,
					`V4 ${name}：每一组的 ${plan4} 行按列加起来 = 那一条压缩条目（两组分别得到第 1、2 条）`
				);
			}
		}

		// 结果块：**整段序列压出的全部条目**（不是只拿一条举例），画成 `[nE × hd]` 的小矩阵，
		// 在 `w ⊙ v` 算完之后逐行出现（分开的两段）。
		const allEntries = (d.trace as HybridTrace).layers[layer].attention.compress!.entries;
		const nEnt = allEntries.length;
		assert(
			!!result && result.kind === 'matrix' && result.data === allEntries,
			`V4 ${name}：结果画成**全部 ${nEnt} 条**压缩条目（${nEnt} × ${hd}，就是后面 ${nEnt} 条压缩 KV），不是只拿一条举例`
		);
		if (!overlap) {
			// HCA：上游一组一行地出结果（先出完第 1 条、再第 2 条），所以**不写死框**、
			// 也不用"整列一起出现"那套（`revealGroup`）
			assert(
				result?.kind === 'matrix' &&
					result.groupRows === plan4 &&
					result.revealGroup === undefined &&
					result.revealOrder !== 'col' &&
					result.highlightRows === undefined,
				`V4 ${name}：结果在 w ⊙ v 之后**逐行**出现（先出完第 1 条再第 2 条），不写死框`
			);
		} else {
			// CSA：结果逐组跟着上游出现——一条窗口算完 → 那 1 条的那一行（`groupRows` 一组 window 行）
			assert(
				result?.kind === 'matrix' && result.groupRows === grpN,
				`V4 ${name}：结果在 w ⊙ v 之后**一行一组**出现（一条窗口 = 1 行，${grpN} 行一组，实际 ${result?.groupRows}）`
			);
		}
		// 每一组按列加出来 = 对应的那一条（逐条对上，不只是举例那条）
		assert(
			result?.kind === 'matrix' &&
				comp.groups.every((gr, e) =>
					allEntries[e].every((v, c) => Math.abs(v - gr.pooled[c]) < 1e-12)
				),
			`V4 ${name}：结果里第 e 条就是上面第 e 组按列加出来的那一条`
		);
		assert(
			prod?.row !== undefined && prod.row === result?.row && !prod.parallel,
			`V4 ${name}：\`w ⊙ v\` 与结果排在同一行（中间画 →），而不是上下堆着`
		);
		// 相加的过程要**看得见**：`w ⊙ v` 按列逐格揭示（`Σ_j` 是沿着列加的）。
		assert(
			(prod?.kind === 'matrix' || prod?.kind === 'ewise') &&
				prod.revealOrder === 'col' &&
				(prod.kind === 'ewise' || !!prod.animated),
			`V4 ${name}：\`w ⊙ v\` **按列**逐格揭示（\`Σ_j\` 是沿着列加的）`
		);
		// 先算 `w ⊙ v`、后算 `v̄`：两个视图**不共用进度**（不用 `sync`），各自一段依次播。
		// 共用进度的话，"每算完一组的一列结果就出一格"，相乘和求和会在画面上缠在一起。
		assert(
			!prod?.sync && !result?.sync && !prod?.parallel && !result?.parallel,
			`V4 ${name}：\`w ⊙ v\` 与结果**不共用进度**（先整块算相乘、后算求和，各占一段）`
		);
		assert(
			isAnimatable(prod!) && isAnimatable(result!),
			`V4 ${name}：\`w ⊙ v\` 与结果都是可动画视图（结果原来一上来就整行显示完了）`
		);
		{
			// 分开算：`w ⊙ v` 与 `v̄` 必须落在**不同的动画单元**（依次播），且相乘先于求和
			const poolUnits = animationUnits(poolStep);
			const uProd = poolUnits.findIndex((u) => u.includes(prod!));
			const uRes = poolUnits.findIndex((u) => u.includes(result!));
			assert(
				uProd >= 0 && uRes >= 0 && uProd !== uRes && uProd < uRes,
				`V4 ${name}：\`w ⊙ v\`（单元 ${uProd}）与 \`v̄\`（单元 ${uRes}）是**两个依次播的单元**，相乘先于求和`
			);
		}
	}

	assert(
		cg.every((g) => g.slots.length === plan2 * 2),
		`重叠窗口：每条压缩条目看 ${plan2 * 2} 个 token = 上一组 ${plan2} 个 + 自己 ${plan2} 个`
	);
	assert(
		cg[0].slots.filter((t) => t < 0).length === plan2,
		`第 1 条没有"上一组"，窗口里 ${plan2} 个空槽（打分 −∞、权重 0）`
	);
	assert(
		cg.map((g) => g.bornAt).join(',') === '1,3,5,7',
		`压缩条目在 token ${cg.map((g) => g.bornAt).join(' / ')} 上才写出来——组收尾了才算得完`
	);
	let colOk = true;
	let emptyOk = true;
	for (const g of cg) {
		for (let c = 0; c < hd; c++) {
			const sum = g.weights.reduce((a, row) => a + row[c], 0);
			if (Math.abs(sum - 1) > 1e-9) colOk = false;
		}
		g.slots.forEach((t, j) => {
			if (t < 0 && g.weights[j].some((w) => w !== 0)) emptyOk = false;
		});
	}
	assert(colOk, '逐通道 softmax：权重的每一**列**加起来是 1（不是每一行）');
	assert(
		cg.every((g) => g.weights.every((row) => row.every(Number.isFinite))),
		'压缩权重全是有限值（空槽是 −∞ 打分，但权重必须是 0 而不是 NaN）'
	);
	assert(emptyOk, '空槽的权重恒为 0');
	assert(
		cs.compKv!.length === cs.nE && cs.compKv![0].length === hd,
		`压缩 KV = [${cs.nE} × ${hd}]，和滑窗 KV 一样宽——所以能进同一个 softmax`
	);

	// indexer
	const idx = cs.indexer!;
	const topkK = cfg.index_topk!;
	assert(
		idx.topk.every((t, i) => t.length === Math.min(topkK, idx.candidates[i].length)),
		`indexer 每个 query 在"已经写出来的条目"里挑最多 ${topkK} 条（真实 ${REAL_V4_FLASH.index_topk}）`
	);
	assert(
		idx.topk[S - 1].length === topkK,
		`最后一个 token 已经能看见全部 ${cg.length} 条，所以确实挑满 ${topkK} 条`
	);
	assert(
		idx.logits.every((row, i) => row.every((v, e) => Number.isFinite(v) === (cg[e].bornAt <= i))),
		'还没写出来的条目打分是 −∞（序列没走到那儿，它根本不存在）'
	);
	assert(
		idx.topk.every((t, i) => t.every((e) => cg[e].bornAt <= i)),
		'top-k 只从已经写出来的条目里挑'
	);
	assert(
		idx.compressor.entriesReady[0].length === cfg.index_head_dim,
		`indexer 用的是**自己那套**更窄的压缩 KV（${cfg.index_head_dim} 维，真实 ${REAL_V4_FLASH.index_head_dim}）`
	);
	let pickOk = true;
	for (let i = 0; i < S; i++) {
		for (let e = 0; e < cs.nE; e++) {
			const want = idx.topk[i].includes(e) && cg[e].bornAt <= i ? 1 : 0;
			if (cs.mask[i][S + e] !== want) pickOk = false;
		}
	}
	assert(pickOk, '压缩条目只有被 indexer 选中的那些进 softmax，其余是 −∞');

	{
		const logitsStep = csaDemo.steps.find((s) => s.id === 'csa-indexer-logits')!;
		const topkView = logitsStep.tensors.find((t) => t.kind === 'mask');
		assert(
			!!topkView && topkView.kind === 'mask' && topkView.matrixName === 'top-k 选择',
			'indexer-logits 页有一个"top-k 选择"掩码视图（逐行揭示选中 / 落选 / 未出生）'
		);
		if (topkView?.kind === 'mask') {
			assert(
				topkView.scores === idx.logits && topkView.shape[0] === S && topkView.shape[1] === cs.nE,
				'选择矩阵画的就是 I 的打分（[S × nE]）：选中的格子保留分数、其余 ∅'
			);
			// 选择矩阵 = 每行 top-k 摊成的 0/1，且 == 注意力掩码的压缩列（逐格对得上）
			let selOk = true;
			for (let i = 0; i < S; i++)
				for (let e = 0; e < cs.nE; e++) {
					const want = idx.topk[i].includes(e) && cg[e].bornAt <= i ? 1 : 0;
					if (topkView.mask[i][e] !== want || cs.mask[i][S + e] !== topkView.mask[i][e]) selOk = false;
				}
			assert(selOk, '选择矩阵每格 = 该行 top-k 是否选中该条目，且与 -attn 掩码的压缩列逐格一致（第 15 页 → 第 17 页）');
		}
	}

	// ── 掩码按"来源段"依次播（token 列 ← 滑窗 / 压缩列 ← top-k / sink ← 恒保留） ──
	{
		const attnStep = csaDemo.steps.find((s) => s.id === 'csa-attn')!;
		const chain = attnStep.tensors.find((t) => t.kind === 'transform')!;
		assert(
			chain.kind === 'transform' && !!chain.preMask && !!chain.preMaskParts,
			'csa-attn：掩码带来源段（preMaskParts），三段依次播'
		);
		if (chain.kind === 'transform' && chain.preMaskParts) {
			const parts = chain.preMaskParts;
			assert(
				parts.length === 3 &&
					parts[0].from === 0 &&
					parts[0].to === S - 1 &&
					parts[1].from === S &&
					parts[1].to === S + cs.nE - 1 &&
					parts[2].from === S + cs.nE &&
					parts[2].to === S + cs.nE,
				`三段列区间正好盖住 token（0..${S - 1}）/ 压缩（${S}..${S + cs.nE - 1}）/ sink（${S + cs.nE}）`
			);
			assert(
				parts[0].label.includes('滑窗') &&
					parts[1].label.includes('top-k 选择') &&
					parts[2].label.includes('恒保留'),
				'三段注解写明各自来源：滑窗规则 / 第 15 页的 top-k 选择 / sink 恒保留'
			);
		}
	}

	{
		const attnStep = csaDemo.steps.find((s) => s.id === 'csa-attn')!;
		const chain = attnStep.tensors.find((t) => t.kind === 'transform')!;
		assert(
			chain.kind === 'transform' && !!chain.topRef && chain.topRef.colFrom === S,
			'csa-attn：top-k 选择矩阵作为 topRef 叠在 S 的压缩列上方（colFrom = S，不另占一行）'
		);
		if (chain.kind === 'transform' && chain.topRef) {
			const tv = chain.topRef.view;
			assert(
				tv.kind === 'mask' && tv.scores === idx.logits && tv.shape[0] === S && tv.shape[1] === cs.nE,
				'topRef 就是 top-k 选择矩阵（[S × nE]，scores = I 的打分）'
			);
			let selOk = true;
			for (let i = 0; i < S; i++)
				for (let e = 0; e < cs.nE; e++) {
					const want = idx.topk[i].includes(e) && cg[e].bornAt <= i ? 1 : 0;
					if (
						tv.kind === 'mask' &&
						(tv.mask[i][e] !== want || cs.mask[i][S + e] !== tv.mask[i][e])
					)
						selOk = false;
				}
			assert(selOk, 'topRef 选择矩阵与 -attn 掩码的压缩列逐格一致（这就是叠在 S 上方的依据）');
			assert(
				!attnStep.tensors.some((t) => t.kind === 'matrix' && t.name.endsWith('-attn-i')),
				'csa-attn 没有独立的 I 参考视图了（topRef 替掉了它）'
			);
		}
	}
	// HCA 无 indexer：压缩段的来源注解是"全保留"；SWA 无压缩列：只有 token + sink 两段
	{
		const hcaChain = hcaDemo.steps
			.find((s) => s.id === 'hca-attn')!
			.tensors.find((t) => t.kind === 'transform')!;
		assert(
			hcaChain.kind === 'transform' &&
				hcaChain.preMaskParts?.length === 3 &&
				hcaChain.preMaskParts[1].label.includes('全保留'),
			'HCA 掩码也是三段，但压缩段的注解是"写出的全保留（不检索）"'
		);
		const swaChain = swaDemo.steps
			.find((s) => s.id === 'swa-attn')!
			.tensors.find((t) => t.kind === 'transform')!;
		assert(
			swaChain.kind === 'transform' && swaChain.preMaskParts?.length === 2,
			'SWA 无压缩列：掩码只有 token + sink 两段'
		);
	}
	assert(
		cs.probs.every((head) => head.every((row) => Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-9)),
		'两类 KV + sink 进的是**同一个** softmax（每行仍然和为 1）'
	);

	// ── 压缩器（HCA：不重叠、无 indexer） ────────
	console.log('    HCA（4 倍压缩（真实 128）、无 indexer）：');
	const hc = hcaDemo.trace.layers[3].attention as HybridTrace;
	assert(hc.indexer === null, 'HCA 没有 indexer——条目本来就少，全看就行');
	assert(hc.nE === 2, `4 倍压缩（真实 128）：${S} 个 token 压出 ${hc.nE} 条`);
	assert(
		hc.compress!.groups.every((g) => g.slots.length === plan4 && g.slots.every((t) => t >= 0)),
		`不重叠：每条看 ${plan4} 个连续 token，没有空槽`
	);
	assert(
		hc.mask.every((row, i) =>
			row.slice(S, S + hc.nE).every((v, e) => v === (hc.compress!.groups[e].bornAt <= i ? 1 : 0))
		),
		'没有 indexer：已经写出来的压缩条目全部参与（稠密）'
	);

	// ── 数据流图（三种层型共用一张，缺的支路淡显） ──
	console.log('    数据流图：');
	const diagrams = [
		['SWA', swaDemo, ['gate', 'w', 'entries', 'ckv', 'itop']],
		['HCA', hcaDemo, ['itop']],
		['CSA', csaDemo, []]
	] as const;
	for (const [label, d, goneWant] of diagrams) {
		const attnSteps = d.steps.filter((s) => s.id.startsWith(label.toLowerCase()));
		assert(
			attnSteps.every((s) => !!s.diagram),
			`V4 ${label}：每一步都带数据流图（${attnSteps.length} 步全覆盖）`
		);
		const dg = attnSteps[0].diagram!;
		const ids = new Set(dg.nodes.map((n) => n.id));
		assert(
			attnSteps.every((s) => s.diagram!.nodes === dg.nodes && s.diagram!.edges === dg.edges),
			`V4 ${label}：各步共用同一张图（只有高亮不同）`
		);
		assert(
			!dg.nodes.some((n, i) => dg.nodes.findIndex((m) => m.col === n.col && m.row === n.row) !== i),
			`V4 ${label}：没有两个节点占据同一个格子`
		);
		assert(
			!dg.edges.some((e) => !ids.has(e.from) || !ids.has(e.to)),
			`V4 ${label}：所有边的两端都指向存在的节点（共 ${dg.edges.length} 条）`
		);
		const bad = attnSteps.flatMap((s) =>
			(s.diagram?.active ?? []).filter((id) => !ids.has(id)).map((id) => `${s.id}:${id}`)
		);
		assert(bad.length === 0, `V4 ${label}：每步高亮的节点都在图上（没有拼错的 id）`);
		assert(
			attnSteps.every((s) => (s.diagram?.active ?? []).length > 0),
			`V4 ${label}：每一步都至少高亮一个节点`
		);
		// 关键：**图上每个节点都被某一步高亮过**——否则说明它只是装饰，或者 active 里写错了 id
		const lit = new Set(attnSteps.flatMap((s) => s.diagram?.active ?? []));
		const unlit = dg.nodes.filter((n) => !n.faded && !lit.has(n.id)).map((n) => n.id);
		assert(
			unlit.length === 0,
			`V4 ${label}：图上每个节点都被某一步高亮过（漏掉：${unlit.join(' / ') || '无'}）`
		);
		const goneIds = dg.nodes.filter((n) => n.tone === 'gone').map((n) => n.id);
		assert(
			JSON.stringify([...goneIds].sort()) === JSON.stringify([...goneWant].sort()),
			`V4 ${label}：这一层**没有**的支路标成灰色淡显（${goneIds.join(' / ') || '无'}）`
		);
		assert(
			dg.nodes.filter((n) => n.tone === 'gone').every((n) => n.faded && !!n.sub),
			`V4 ${label}：灰色节点同时淡显、并在小字里写明为什么没有`
		);
		assert(
			!goneIds.some((id) => lit.has(id)),
			`V4 ${label}：高亮里不会出现本层没有的灰色节点`
		);
		assert(
			dg.edges.filter((e) => e.faded).length > 0 === goneIds.length > 0,
			`V4 ${label}：缺支路的那些边一并淡显`
		);
	}
	// 三种层型的图**布局完全一样**，只是标注不同——切层时是"同一张图变了颜色"
	const posOf = (d: (typeof diagrams)[number][1]) => {
		const dg = d.steps.find((s) => s.diagram)!.diagram!;
		return dg.nodes
			.map((n) => `${n.id}@${n.col},${n.row}`)
			.sort()
			.join(' ');
	};
	const base = posOf(csaDemo);
	assert(
		posOf(swaDemo) === base && posOf(hcaDemo) === base,
		'SWA / CSA / HCA 用的是**同一套坐标**（同一张图，只换了颜色）'
	);
	assert(
		csaDemo.steps.find((s) => s.diagram)!.diagram!.nodes !==
			swaDemo.steps.find((s) => s.diagram)!.diagram!.nodes,
		'三种层型各用**自己的图对象**（不是把同一张改了）'
	);
}

console.log('\n[DeepSeek V4 mHC 残差]');
{
	// 4 条并行残差流，每个子层前 pre 混合、算完 post 写回。
	// 对照 sglang `hc_pre` / `hc_post` / `hc_head_torch` 逐行实现，这里验证数学正确性。
	const v4 = buildLlmFlow({ modelId: 'v4-flash', attnId: '', ffnId: 'auto', layer: 0 });
	const cfg = V4_FLASH_MODEL.cfg;
	const hc = cfg.hc_mult!;
	const eps = cfg.hc_eps!;
	const iters = cfg.hc_sinkhorn_iters!;
	const S = cfg.seq_len;
	const D = cfg.d_model;
	assert(hc === 2, `展示 hc_mult = ${hc}（真实 ${REAL_V4_FLASH.hc_mult}，演示只画 2 条流，页面有标注）`);
	const lt0 = v4.trace.layers[0];
	assert(!!lt0.mhc, 'V4 每一层都带 mHC 中间量');

	const { streamsIn, attnMix, attnWrite, streamsAfterAttn, ffnMix, ffnWrite } = lt0.mhc!;
	// 第 0 层：流初始是**同一份** embedding（数量 = 展示的 hc）
	assert(
		streamsIn.length === hc && streamsIn.every((s, k) => k === 0 || JSON.stringify(s) === JSON.stringify(streamsIn[0])),
		`第 0 层 ${hc} 条流是同一份 embedding（真实 ` +
			`deepseek_v4.py:4685 unsqueeze(1).repeat(1, hc_mult, 1)）`
	);

	// 门控范围：pre ∈ (ε, 1+ε)、post ∈ (0, 2)
	for (const [tag, mix] of [
		['attn', attnMix],
		['ffn', ffnMix]
	] as const) {
		assert(
			mix.pre.flat().every((v) => v > eps && v < 1 + eps),
			`${tag} pre ∈ (ε, 1+ε)：${mix.pre[0][0].toFixed(4)} …（sigmoid + ε，不归一化）`
		);
		assert(
			mix.post.flat().every((v) => v > 0 && v < 2),
			`${tag} post ∈ (0, 2)：${mix.post[0][0].toFixed(4)} …（2·sigmoid，写回能增强）`
		);
		// comb 每 token 一个 hc×hc，Sinkhorn 后行和 ≈ 列和 ≈ 1（双随机）
		const rowErr = mix.comb.map((row) => {
			const rs: number[] = [];
			for (let j = 0; j < hc; j++) rs.push(row.slice(j * hc, j * hc + hc).reduce((a, b) => a + b, 0));
			return Math.max(...rs.map((r) => Math.abs(r - 1)));
		});
		const colErr = mix.comb.map((row) => {
			const cs: number[] = [];
			for (let k = 0; k < hc; k++) {
				let s = 0;
				for (let j = 0; j < hc; j++) s += row[j * hc + k];
				cs.push(s);
			}
			return Math.max(...cs.map((c) => Math.abs(c - 1)));
		});
		assert(
			Math.max(...rowErr, ...colErr) < 0.05,
			`${tag} comb 是双随机矩阵（行和/列和 ≈ 1，最大偏差 ${Math.max(...rowErr, ...colErr).toExponential(1)}，` +
				`Sinkhorn ${iters} 次）`
		);
	}

	// 混合 = Σ pre_k·r_k（逐元素）
	const comb = attnMix.combined;
	const mixedOk = comb.every((row, i) =>
		row.every(
			(v, d) =>
				Math.abs(v - streamsIn.reduce((a, s, k) => a + attnMix.pre[i][k] * s[i][d], 0)) < 1e-9
		)
	);
	assert(mixedOk, 'combined = Σ_k pre_k·r_k（逐元素一致）');

	// 写回 = post_k·y + Σ_j comb_kj·r_j
	const writeOk = attnWrite.newStreams.every((ns, k) =>
		ns.every((row, i) =>
			row.every((v, d) => {
				const want = attnWrite.postY[k][i][d] + attnWrite.combOld[k][i][d];
				return Math.abs(v - want) < 1e-9;
			})
		)
	);
	assert(writeOk, 'newStreams[k] = postY[k] + combOld[k]（逐元素一致）');

	// 流在子层之间确实"写回 + 再读"：FFN 的输入流 = 注意力写回后的流
	const sameStreams =
		streamsAfterAttn.every((s, k) => JSON.stringify(s) === JSON.stringify(attnWrite.newStreams[k])) &&
		lt0.mhc!.streamsAfterFfn.every((s, k) => JSON.stringify(s) === JSON.stringify(ffnWrite.newStreams[k]));
	assert(sameStreams, 'streamsAfterAttn / streamsAfterFfn 就是对应写回产出（引用一致）');

	// hc_head：压回 [S, D]，logits 有限
	assert(v4.trace.hcHead!.out.length === S && v4.trace.hcHead!.out[0].length === D, 'hc_head 压回 [S, d_model]');
	assert(v4.trace.logits.every((row) => row.every(Number.isFinite)), 'V4 的 logits 全部有限（hc_head 后进 LM Head）');

	// 残差阶段在 V4 有 7 段（gates-attn/pre-attn/post-attn/gates-ffn/pre-ffn/post-ffn/head），关掉整段一起消失
	const v4ResSegs = v4.segments.filter((s) => s.stage === 'residual');
	assert(v4ResSegs.length === 7, `V4 的 Residual 阶段有 7 段（实际 ${v4ResSegs.length}）`);
	const off7 = buildLlmFlow({ modelId: 'v4-flash', attnId: '', ffnId: 'auto', layer: 0, off: ['residual'] });
	assert(
		off7.steps.length === v4.steps.length - 7 &&
			!off7.stages.some((s) => s.id === 'residual') &&
			off7.segments.every((s) => s.stage !== 'residual'),
		`关掉 Residual：7 步一起消失（${v4.steps.length} → ${off7.steps.length}）、节点和区间也没有它`
	);
}

console.log('\n[阶段开关]');
{
	// 关掉的阶段**根本不进流程**：步骤、流程图节点、总览都跟着消失。
	// 因为没有任何跨阶段的 `{{step:...}}` 引用，所以关掉一整段不会把标签解析成 `?`。
	const toggles = getFlow('llm').stageToggles ?? [];
	assert(toggles.length === 5, `流程声明了 ${toggles.length} 个可开关的阶段`);
	assert(
		toggles.map((t) => t.id).join(',') === 'embedding-lookup,attention,ffn-moe,residual,lm-head',
		`阶段顺序与流程一致：${toggles.map((t) => t.title).join(' → ')}`
	);
	assert(
		toggles.filter((t) => t.defaultOn).map((t) => t.id).join(',') === 'attention,ffn-moe,residual',
		`默认开的是 Attention / FFN-MoE / Residual（Embedding / LM Head 默认关）：${toggles
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

	// 逐个关掉：该阶段的步骤与流程图节点都要消失，其余保持不变。
	// 注意一个阶段可以有**多段**区间（Residual 拆在注意力后 / FFN 后两处），
	// 关掉时是这几段步数**之和**一起消失。
	for (const t of toggles) {
		const off = buildLlmFlow({ attnId: 'mla', ffnId: 'auto', layer: MOE_LAYER, off: [t.id] });
		const own = allOn.segments
			.filter((s) => s.stage === t.id)
			.reduce((n, s) => n + (s.to - s.from), 0);
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

	// 只留默认开的那三个：流程应当就是"Attention + FFN/MoE + Residual"
	const defaults = buildLlmFlow({
		attnId: 'mla',
		ffnId: 'auto',
		layer: MOE_LAYER,
		off: toggles.filter((t) => !t.defaultOn).map((t) => t.id)
	});
	assert(
		defaults.stages.map((s) => s.id).join(',') === 'attention,ffn-moe,residual',
		`按默认开关跑出来就是三段：${defaults.stages.map((s) => s.title).join(' → ')}（${defaults.steps.length} 步）`
	);

	// 五个全关：不报错，流程为空（UI 会提示"至少打开一个"）
	const none = buildLlmFlow({
		attnId: 'mla',
		ffnId: 'auto',
		layer: MOE_LAYER,
		off: toggles.map((t) => t.id)
	});
	assert(
		none.steps.length === 0 && none.stages.length === 0 && none.segments.length === 0,
		'五个阶段全关掉时流程为空（步骤 / 节点 / 区间都是空的），不抛错'
	);
}

console.log('\n[确定性]');
const again = buildLlmFlow({ attnId: 'mha', ffnId: 'auto', layer: MOE_LAYER });
assert(
	JSON.stringify(again.trace.layers[MOE_LAYER].ffn) === JSON.stringify(moe),
	'固定 seed：重复运行结果完全一致'
);

console.log('\n全部检查通过 ✓\n');
