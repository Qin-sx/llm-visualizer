/**
 * 插件注册表。
 *
 * 新增一个算子 = 新增一个文件并在末尾 `register(xxx)`。
 * 新增一个**执行方式**（如 MLA 的吸收合并）= `registerImplementation()`。
 * 新增一条**流程**（如 diffusion / 多模态）= `registerFlow()`。
 */
import type { FlowSpec } from './flow';
import type {
	ImplementationSpec,
	LayerCtx,
	Mat,
	OverlaySpec,
	SemanticsSpec,
	SlotId,
	SlotTrace,
	Step
} from './types';

const semantics = new Map<string, SemanticsSpec<any>>();
const implementations = new Map<string, ImplementationSpec<any>>();
const overlays = new Map<string, OverlaySpec>();
const flows = new Map<string, FlowSpec<any, any>>();

export function register(spec: SemanticsSpec<any>): void {
	if (semantics.has(spec.id)) {
		console.warn(`[registry] 语义插件 ${spec.id} 被重复注册，后者覆盖前者`);
	}
	semantics.set(spec.id, spec);
}

export function getSemantics(id: string): SemanticsSpec<any> {
	const spec = semantics.get(id);
	if (!spec) throw new Error(`[registry] 找不到语义插件：${id}`);
	return spec;
}

export function listBySlot(slot: SlotId): SemanticsSpec<any>[] {
	return [...semantics.values()].filter((s) => s.slot === slot);
}

/** 全部已注册的语义插件（model 层据此分配权重） */
export function allSemantics(): SemanticsSpec<any>[] {
	return [...semantics.values()];
}

// ── 执行实现（"怎么算"）────────────────────────────────────
export function registerImplementation(spec: ImplementationSpec<any>): void {
	if (implementations.has(spec.id)) {
		console.warn(`[registry] 实现插件 ${spec.id} 被重复注册，后者覆盖前者`);
	}
	implementations.set(spec.id, spec);
}

export function getImplementation(id: string): ImplementationSpec<any> {
	const spec = implementations.get(id);
	if (!spec) throw new Error(`[registry] 找不到实现插件：${id}`);
	return spec;
}

/** 某个语义可以配哪些实现（UI 的"实现"下拉就用它） */
export function listImplementationsFor(semanticsId: string): ImplementationSpec<any>[] {
	return [...implementations.values()].filter((i) => i.supports.includes(semanticsId));
}

export function allImplementations(): ImplementationSpec<any>[] {
	return [...implementations.values()];
}

/**
 * 语义 × 实现 → 一个"当前生效的插件"。
 *
 * 调用方（`model/forward` 与流程）只跟这个对象打交道，不用关心
 * "这一步的 compute 到底来自语义还是实现"。
 */
export interface ResolvedPlugin<TTrace extends SlotTrace = SlotTrace> {
	semantics: SemanticsSpec<any>;
	implementation: ImplementationSpec<any> | null;
	/** 显示名，如 `多头潜在注意力 (MLA) · 矩阵吸收合并` */
	name: string;
	desc: string;
	compute(input: Mat, ctx: LayerCtx): TTrace;
	steps(trace: TTrace, ctx: LayerCtx): Step[];
	/** 权重包在 `weights.slots` 里的 key（实现自带权重就用实现的 id） */
	weightKey: string;
}

export function resolvePlugin(semanticsId: string, implId?: string | null): ResolvedPlugin {
	const sem = getSemantics(semanticsId);
	const impl = implId ? getImplementation(implId) : null;
	if (impl && !impl.supports.includes(semanticsId)) {
		throw new Error(
			`[registry] 实现 ${impl.id} 不支持语义 ${semanticsId}（它支持：${impl.supports.join(', ')}）`
		);
	}
	// 实现给了 compute/steps 就由它接管，否则退回语义插件的
	const owns = impl?.compute ? impl : null;
	return {
		semantics: sem,
		implementation: impl,
		name: impl ? `${sem.name} · ${impl.name}` : sem.name,
		desc: impl ? `${sem.desc}（${impl.desc}）` : sem.desc,
		compute: (owns?.compute ?? sem.compute) as ResolvedPlugin['compute'],
		steps: (owns?.steps ?? sem.steps) as ResolvedPlugin['steps'],
		weightKey: impl?.makeWeights ? impl.id : sem.id
	};
}

export function registerOverlay(spec: OverlaySpec): void {
	overlays.set(spec.id, spec);
}

export function listOverlays(): OverlaySpec[] {
	return [...overlays.values()];
}

// ── 流程（Flow）────────────────────────────────────────────
/**
 * 流程 = 一串阶段（Embedding → Attention → FFN/MoE → LM Head）。
 *
 * 注册进来的流程可以有多条（LLM / diffusion / 多模态），页面的流水线节点
 * 与步骤序列都从流程读，所以加一条流程不用碰 `core/` 与组件。
 */
export function registerFlow(spec: FlowSpec<any, any>): void {
	if (flows.has(spec.id)) {
		console.warn(`[registry] 流程 ${spec.id} 被重复注册，后者覆盖前者`);
	}
	flows.set(spec.id, spec);
}

export function getFlow(id: string): FlowSpec<any, any> {
	const spec = flows.get(id);
	if (!spec) throw new Error(`[registry] 找不到流程：${id}`);
	return spec;
}

export function listFlows(): FlowSpec<any, any>[] {
	return [...flows.values()];
}
