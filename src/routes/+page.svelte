<script lang="ts">
	import '$lib/slots'; // 副作用：注册全部算子插件
	import '$lib/flows'; // 副作用：注册全部流程
	import { untrack } from 'svelte';
	import { getFlow, getSemantics, listImplementationsFor } from '$lib/core/registry';
	import { getModel, layerKind, MODELS } from '$lib/model/config';
	import { buildLlmFlow, LLM_STAGE_TOGGLES, type LlmFlowResult } from '$lib/flows/llm';
	import { createPlayer, type Player } from '$lib/core/timeline';
	import { animateHandoffs, resetHandoffs } from '$lib/core/handoff';
	import { buildOverviewItems } from '$lib/core/overview';
	import Overview from '$lib/components/Overview.svelte';
	import Pipeline from '$lib/components/Pipeline.svelte';
	import StepPanel from '$lib/components/StepPanel.svelte';
	import PlayerBar from '$lib/components/Player.svelte';

	/**
	 * **代表模型**：注意力与 FFN 都按层定下来，所以 UI 上不是"自由组合插件"，
	 * 而是"选一个代表模型 + 选一层"。非法组合（如 MLA 配 V4）因此根本不存在。
	 */
	let modelId = $state(MODELS[0].id);
	const model = $derived(getModel(modelId));
	const layerList = $derived(Array.from({ length: model.cfg.num_layers }, (_, i) => i));
	/** 可开关的阶段（由流程声明，见 FlowSpec.stageToggles） */
	const stageToggles = getFlow('llm').stageToggles ?? LLM_STAGE_TOGGLES;

	/** 模型内可挑的注意力（R1 有 MHA / MLA）；空 = 逐层固定（V4） */
	let attnId = $state(MODELS[0].attnChoices[0] ?? '');
	let layer = $state(0);

	/**
	 * 换模型时把不适用的选择清掉：R1 选的 MLA 不能带进 V4，层号也要落在范围内。
	 */
	$effect(() => {
		const m = getModel(modelId);
		if (!m.attnChoices.includes(attnId)) attnId = m.attnChoices[0] ?? '';
		if (layer >= m.cfg.num_layers) layer = 0;
	});

	/**
	 * 关掉的阶段：默认按流程声明的 `defaultOn` 来（Attention / FFN-MoE 开，
	 * Embedding / LM Head 关）。关掉的阶段**根本不进流程**——步骤、流程图节点、
	 * 总览都跟着消失，而不是渲染出来再藏起来。
	 */
	let offStages = $state<string[]>(stageToggles.filter((t) => !t.defaultOn).map((t) => t.id));
	const toggleStage = (id: string) =>
		(offStages = offStages.includes(id) ? offStages.filter((x) => x !== id) : [...offStages, id]);

	/**
	 * 注意力的**执行方式**（"怎么算"）。
	 *
	 * 只有当前语义确实有实现可选时才显示这个下拉——所以选到 MLA 才会出现
	 * "矩阵吸收合并"。这是 `ImplementationSpec` 那一层在 UI 上的样子：
	 * 语义（算什么）与实现（怎么算）正交，组合不合法时根本不出现在列表里。
	 */
	const attnImpls = $derived(listImplementationsFor(attnId));
	let attnImplId = $state<string>('');
	// 换语义时清掉上一个语义的实现选择（否则会带着不支持的实现去 build）
	$effect(() => {
		const impls = listImplementationsFor(attnId);
		if (attnImplId && !impls.some((i) => i.id === attnImplId)) attnImplId = '';
	});

	// 初始化时**同步**算一次：这样预渲染出的 HTML 里就带着第一步的内容，
	// 不依赖客户端 JS 也能看到东西（也便于构建后直接验证渲染结果）。
	// 只取初始值；后续变化由下面的 $effect 负责。
	// svelte-ignore state_referenced_locally
	let demo = $state<LlmFlowResult>(
		buildLlmFlow({ modelId, attnId, attnImplId, ffnId: 'auto', layer, off: offStages })
	);
	let stepIndex = $state(0);
	/** 整条时间轴的进度（底部播放条用） */
	let progress = $state(0);
	/** 当前步内进度（页面动画用） */
	let stepProgress = $state(0);
	let playing = $state(false);
	let speed = $state(1);

	let player: Player | null = null;

	function attachPlayer(result: LlmFlowResult) {
		player?.destroy();
		player = createPlayer(
			result.steps,
			(p) => {
				stepIndex = p.stepIndex;
				progress = p.duration ? p.time / p.duration : 0;
				stepProgress = p.stepProgress;
			},
			(v) => {
				playing = v;
			}
		);
		// 变速偏好跨重建保留；用 untrack 避免把 speed 变成 effect 依赖
		player.setSpeed(untrack(() => speed));
		stepIndex = 0;
		progress = 0;
		stepProgress = 0;
		playing = false;
	}

	// 步骤切换后，把跨步骤复用的矩阵（如 `attn-q` 算出的 Q）飞向它在新步骤里的位置
	let lastHandoffStep = -1;

	// 换模型 / 换插件 / 换层 → 重算数据并重建时间轴
	$effect(() => {
		const result = buildLlmFlow({ modelId, attnId, attnImplId, ffnId: 'auto', layer, off: offStages });
		resetHandoffs();
		lastHandoffStep = -1; // 让下面的 effect 重新采集位置基线
		demo = result;
		attachPlayer(result);
	});

	$effect(() => {
		const idx = stepIndex;
		if (idx === lastHandoffStep) return;
		lastHandoffStep = idx;
		requestAnimationFrame(animateHandoffs);
	});

	// ── 流程总览（渲染在流水线下方） ──────────────
	const overviewItems = $derived(buildOverviewItems(demo.steps));

	const currentStepId = $derived(demo.steps[stepIndex]?.id ?? '');

	/** 点总览里的方块 → 跳到对应步骤（总览自己会画出这一步的来源线） */
	function jumpToStep(stepId: string) {
		const i = demo.steps.findIndex((s) => s.id === stepId);
		if (i >= 0) player?.seekStep(i);
	}

	const steps = $derived(demo.steps);
	const currentStep = $derived(steps[stepIndex] ?? null);
	/** 当前步骤落在哪个阶段（流水线高亮用） */
	const activeStage = $derived(
		demo.segments.find((s) => stepIndex >= s.from && stepIndex < s.to)?.stage ?? ''
	);

	/** 某一层用的注意力插件名（逐层固定时给 UI 显示用） */
	const attnNameOf = (l: number) => getSemantics(demo.model.attnByLayer[l] ?? attnId).name;

	function handleSpeed(v: number) {
		speed = v;
		player?.setSpeed(v);
	}
</script>

<main>
	<header>
		<div class="brand">
			<h1>LLM<span>Visualizer</span></h1>
			<p>可插拔地拆解大模型的注意力与 MoE 内部机制</p>
		</div>
		<div class="badge">教学缩放版 · 随机权重</div>
	</header>

	<section class="controls">
		<label>
			<span class="ctl-label">代表模型</span>
			<select bind:value={modelId}>
				{#each MODELS as m (m.id)}
					<option value={m.id}>{m.name}</option>
				{/each}
			</select>
		</label>

		{#if model.attnChoices.length > 1}
			<label>
				<span class="ctl-label">Attention 插件</span>
				<select bind:value={attnId}>
					{#each model.attnChoices as id (id)}
						<option value={id}>{getSemantics(id).name}</option>
					{/each}
				</select>
			</label>
		{/if}

		{#if attnImpls.length > 0}
			<label>
				<span class="ctl-label">Attention 实现</span>
				<select bind:value={attnImplId}>
					<option value="">朴素（逐头上投影）</option>
					{#each attnImpls as im (im.id)}
						<option value={im.id}>{im.name}</option>
					{/each}
				</select>
			</label>
		{/if}

		<div class="layers">
			<span class="ctl-label">检视层</span>
			<div class="layer-row">
				{#each layerList as l (l)}
					<button
						class="layer-chip"
						class:active={layer === l}
						class:moe={layerKind(model.cfg, l) === 'moe'}
						title={`第 ${l + 1} 层：${attnNameOf(l)}`}
						onclick={() => (layer = l)}
					>
						<span class="num">{l + 1}</span>
						<span class="tag">{model.layerLabels[l]}</span>
					</button>
				{/each}
			</div>
			<!-- 注意力逐层固定的模型（V4）没有"选插件"的下拉，标签本身就是"这一层用什么" -->
			<span class="legend">
				标签 = 这一层的{model.attnChoices.length > 1 ? 'FFN 类型' : '注意力类型'}
			</span>
		</div>

		<div class="layers">
			<span class="ctl-label">展示阶段（关掉的整段从流程里去掉）</span>
			<div class="layer-row">
				{#each stageToggles as t (t.id)}
					<button
						class="stage-chip"
						class:active={!offStages.includes(t.id)}
						title={offStages.includes(t.id) ? `打开 ${t.title}` : `关掉 ${t.title}`}
						onclick={() => toggleStage(t.id)}
					>
						{t.title}
					</button>
				{/each}
			</div>
		</div>
	</section>

	<Pipeline
		stages={demo.stages}
		{activeStage}
		{layer}
		layerCount={demo.model.cfg.num_layers}
	/>

	<!-- 流程总览放在流水线**下面**：先看清"整条链有哪几个阶段、当前在哪一段"，
	     再看展开的算子细节 -->
	<Overview items={overviewItems} activeStepId={currentStepId} onpick={jumpToStep} />

	<section class="stage">
		{#if currentStep}
			<StepPanel step={currentStep} progress={stepProgress} />
		{:else if steps.length === 0}
			<p class="loading">
				四个阶段都关掉了。至少打开一个（见上方「展示阶段」），才有动画可看。
			</p>
		{:else}
			<p class="loading">加载中…</p>
		{/if}
	</section>

	<aside class="desc">
		<div class="desc-title">{demo.model.name} · {demo.attn.name} + {demo.ffnSpec.name}</div>
		<p>{demo.model.desc}</p>
		<p>{demo.attn.desc}</p>
		<p>{demo.ffnSpec.desc}</p>
		<div class="caveat">
			维度按 {demo.model.name} 的结构等比缩小；权重为固定 seed
			的随机值——本页讲的是<b>机制</b>，不是知识。
		</div>
		<p class="lic"><a href="/licenses/">第三方许可 →</a></p>
	</aside>

	{#if steps.length > 0}
		<footer>
			<PlayerBar
				{playing}
				{progress}
				{stepIndex}
				stepCount={steps.length}
				{speed}
				onToggle={() => player?.toggle()}
				onPrev={() => player?.prev()}
				onNext={() => player?.next()}
				onRestart={() => player?.restart()}
				onSeek={(r) => player?.seekRatio(r)}
				onSpeed={handleSpeed}
			/>
		</footer>
	{/if}
</main>

<style>
	main {
		max-width: 1180px;
		margin: 0 auto;
		padding: 1.5rem 1.25rem 6rem;
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}
	header {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 1rem;
	}
	h1 {
		margin: 0;
		font-size: 1.6rem;
		font-weight: 800;
		letter-spacing: -0.02em;
		color: #1e293b;
	}
	h1 span {
		color: #4f46e5;
		margin-left: 0.25rem;
	}
	.brand p {
		margin: 0.2rem 0 0;
		font-size: 0.8rem;
		color: #64748b;
	}
	.badge {
		font-size: 0.68rem;
		padding: 0.25rem 0.6rem;
		border-radius: 999px;
		background: #eef2ff;
		color: #4338ca;
		border: 1px solid #c7d2fe;
		white-space: nowrap;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 1.25rem;
		align-items: flex-end;
		padding: 0.75rem 0.9rem;
		background: #ffffff;
		border: 1px solid #e2e8f0;
		border-radius: 0.6rem;
	}
	.controls label {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
	}
	.ctl-label {
		font-size: 0.68rem;
		color: #64748b;
	}
	select {
		font-size: 0.8rem;
		padding: 0.3rem 0.5rem;
		border-radius: 0.375rem;
		border: 1px solid #e2e8f0;
		background: #f8fafc;
		color: #1e293b;
		min-width: 14rem;
	}
	.layers {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
	}
	.layer-row {
		display: flex;
		gap: 0.25rem;
	}
	.layer-chip {
		display: flex;
		flex-direction: column;
		align-items: center;
		line-height: 1.1;
		min-width: 2.4rem;
		padding: 0.25rem 0.4rem;
		border-radius: 0.375rem;
		border: 1px solid #e2e8f0;
		background: #ffffff;
		color: #64748b;
		font-size: 0.75rem;
		cursor: pointer;
	}
	.layer-chip .tag {
		font-size: 0.56rem;
		letter-spacing: 0.02em;
		opacity: 0.8;
	}
	.layer-chip.moe {
		background: #fdf4ff;
		border-color: #f5d0fe;
		color: #a21caf;
	}
	.layer-chip.active {
		outline: 2px solid #4f46e5;
		outline-offset: 1px;
		font-weight: 700;
	}
	/* 阶段开关：开着的用靛蓝实底，关掉的灰掉并加删除线——一眼看出"这段不在流程里" */
	.stage-chip {
		padding: 0.28rem 0.6rem;
		border-radius: 0.375rem;
		border: 1px solid #e2e8f0;
		background: #f8fafc;
		color: #94a3b8;
		font-size: 0.72rem;
		cursor: pointer;
		text-decoration: line-through;
	}
	.stage-chip.active {
		background: #eef2ff;
		border-color: #a5b4fc;
		color: #3730a3;
		font-weight: 600;
		text-decoration: none;
	}
	.legend {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		font-size: 0.62rem;
		color: #94a3b8;
	}
	.stage {
		min-height: 24rem;
		padding: 1.1rem;
		background: #ffffff;
		border: 1px solid #e2e8f0;
		border-radius: 0.75rem;
	}
	.loading {
		color: #94a3b8;
		font-size: 0.85rem;
	}
	.desc {
		padding: 0.8rem 0.95rem;
		background: #f8fafc;
		border: 1px dashed #e2e8f0;
		border-radius: 0.6rem;
	}
	.desc-title {
		font-size: 0.8rem;
		font-weight: 600;
		color: #334155;
		margin-bottom: 0.35rem;
	}
	.desc p {
		margin: 0.25rem 0;
		font-size: 0.78rem;
		color: #64748b;
		line-height: 1.5;
	}
	.caveat {
		margin-top: 0.5rem;
		padding-top: 0.5rem;
		border-top: 1px solid #e2e8f0;
		font-size: 0.7rem;
		color: #94a3b8;
	}
	.lic {
		margin-top: 0.5rem !important;
	}
	.lic a {
		color: #4338ca;
		text-decoration: none;
	}
	.lic a:hover {
		text-decoration: underline;
	}
	footer {
		position: sticky;
		bottom: 1rem;
	}
</style>
