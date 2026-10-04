/**
 * 流程（Flow）组装——**与具体算子无关**。
 *
 * 一条流程 = 一串**阶段**（stage）。每个阶段把自己那份中间量翻译成步骤，
 * 并声明它在流程图里怎么显示。LLM 是「Embedding → Attention → FFN/MoE → LM Head」。
 *
 * "步骤从哪来、流程图上有哪些节点"由数据描述，不写死在组件里——
 * 新增一条流程只需新写一个 `flows/xxx.ts` 并 `registerFlow()`。
 *
 * 阶段有两种作用范围（`scope`）：
 *   - `model`：整个模型只做一次（如 Embedding、LM Head）
 *   - `layer`：每个 Transformer 层各做一次（如 Attention、FFN/MoE）
 * 流程图据此说明"哪些节点是模型级的、哪些是正在检视的那一层"。
 */
import { matRefsOf } from './overview';
import type { Depth, ModelConfigLike, SlotId, Step } from './types';

/**
 * 阶段在流程图上的展示信息（不含"怎么产步骤"）。
 *
 * 单独拆出来，是因为层级阶段的展示信息（如 Attention 的副标题 = 当前插件名）
 * 由流程在运行时拼，步骤却由插件提供——两者来源不同，不能强绑成一个对象。
 */
export interface StageInfo {
	/** 稳定 id，同时作为步骤归属（见 `FlowSegment.stage`） */
	id: string;
	/** 属于哪个插槽（与 `SemanticsSpec.slot` 同一套词汇） */
	slot: SlotId;
	/** `model` = 整个模型一次；`layer` = 每层各一次 */
	scope: 'model' | 'layer';
	/** 流程图节点标题，如 `Embedding` */
	title: string;
	/** 流程图节点副标题（一句话），如当前插件的名字 */
	sub?: string;
	/** 一段说明，进页脚 / 文档 */
	desc?: string;
	/** 表示深度徽标 */
	depth: Depth;
}

/**
 * 阶段：展示信息 + 怎么把自己的中间量翻译成步骤。
 *
 * `TData` 是该阶段需要的中间量，由流程（如 `flows/llm.ts`）塞进来——
 * core 不认识它的内容，只负责按顺序拼装，所以**阶段可以自带任何数据**。
 */
export interface StageSpec<TData = unknown> extends StageInfo {
	steps(data: TData, cfg: ModelConfigLike): Step[];
}

/** 一个阶段产出的步骤 */
export interface StageOutput {
	stage: StageInfo;
	steps: Step[];
}

/** 步骤区间 → 阶段（供流程图高亮） */
export interface FlowSegment {
	stage: string;
	slot: SlotId;
	from: number;
	to: number;
}

export interface BuiltFlow {
	/** 流程图节点，按阶段顺序 */
	stages: StageInfo[];
	/** 全部步骤，已按阶段顺序拼好 */
	steps: Step[];
	/** 步骤区间到阶段的映射 */
	segments: FlowSegment[];
}

/**
 * 流程里**可以开关的阶段**。
 *
 * 与运行时产出的 `StageInfo` 分开：`StageInfo` 的副标题要等插件选好才知道
 * （Attention 的副标题就是当前插件名），而开关按钮在构建之前就得画出来。
 */
export interface StageToggle {
	/** 与 `StageInfo.id` 一致 */
	id: string;
	title: string;
	/** 初始状态：关掉的阶段不进流程（步骤与流程图节点一起消失） */
	defaultOn: boolean;
}

/** 一条流程（可插拔的顶层单位） */
export interface FlowSpec<TOpts = unknown, TResult extends BuiltFlow = BuiltFlow> {
	id: string;
	name: string;
	desc: string;
	/** 这条流程由哪些阶段组成、各自默认开不开（UI 的开关按钮按它渲染） */
	stageToggles: StageToggle[];
	/** 跑一遍参考模型，产出这条流程的全部阶段与步骤 */
	build(opts: TOpts): TResult;
}

/**
 * 把各阶段的步骤拼成一条流程。
 *
 * 区段（`segments`）在这里算，因为只有组装者知道每一步落在哪个阶段里；
 * 流程图高亮、页脚说明都基于它。
 */
export function buildFlow(outputs: StageOutput[]): BuiltFlow {
	const stages: StageInfo[] = [];
	const steps: Step[] = [];
	const segments: FlowSegment[] = [];

	for (const { stage, steps: own } of outputs) {
		// 同一个阶段 id 可以出现**多次**（如 Residual 拆在注意力后 / FFN 后两处）：
		// 流程图节点只留**一个**，位置取最后一次出现（Residual 的节点因此落在 FFN 之后、
		// 当"整层收尾"）；每段区间各记一条，步骤高亮 / 总览仍按真实位置。
		const dup = stages.findIndex((s) => s.id === stage.id);
		if (dup >= 0) stages.splice(dup, 1);
		stages.push(stage);
		segments.push({
			stage: stage.id,
			slot: stage.slot,
			from: steps.length,
			to: steps.length + own.length
		});
		steps.push(...own);
	}

	resolveStepRefs(steps);
	return { stages, steps, segments };
}

/**
 * 把标签里的 `{{step:步骤id}}` 占位符换成实际步骤号。
 *
 * 插件不该硬编码"第 N 步"这种序号——步骤一拆分/重排就会过期。所以插件写
 * `{{step:attn-av}}`，由这里在**整条流程拼好之后**统一解析——
 * 这样跨阶段的引用（如 Embedding 指向 Attention 的某一步）也能解析。
 */
export function resolveStepRefs(steps: Step[]): void {
	const no = new Map(steps.map((s, i) => [s.id, i + 1]));
	const fix = (s?: string) =>
		s?.replace(/\{\{step:([\w-]+)\}\}/g, (_m, id: string) => String(no.get(id) ?? '?'));

	for (const step of steps) {
		step.label = fix(step.label) ?? step.label;
		for (const t of step.tensors) {
			if (t.label) t.label = fix(t.label);
			// `shape` 的 `note`、`kvcache` 的 `note` 也是给人看的文案，同样要解析
			// （漏掉的话页面上会直接显示 `{{step:xxx}}`——`verify` 有一条断言盯着）
			if ((t.kind === 'shape' || t.kind === 'kvcache') && t.note) t.note = fix(t.note);
			// `preMask` 的**来源段**注解（`preMaskParts` 每段的 label）与**来源参考条**
			// （`topRef` 的 label）同样是给人看的文案
			// （如 V4 融合注意力里"压缩列 ← 第 N 步的 top-k 选择"）
			if (t.kind === 'transform') {
				if (t.preMaskParts) {
					for (const p of t.preMaskParts) p.label = fix(p.label) ?? p.label;
				}
				if (t.topRef?.view.label) t.topRef.view.label = fix(t.topRef.view.label) ?? t.topRef.view.label;
			}
			for (const ref of matRefsOf(t)) {
				if (ref.label) ref.label = fix(ref.label);
			}
		}
	}
}
