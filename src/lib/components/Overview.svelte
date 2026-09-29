<script lang="ts">
	/**
	 * 流程总览（挂在主流水线**下面**：先看清整条链有哪几个阶段，再看展开的算子细节）。
	 *
	 * 两类小组横向串成一条连贯的流（小组之间用竖直虚线分隔）：
	 *   - 矩阵乘组：左操作数在左下、右操作数在上、结果在右下
	 *   - 单目运算组：输入在左、输出在右（如 `S_h → P_h`）
	 *
	 * 每个张量画成一个**方框**（名字 + 尺寸）。配色按张量角色（`MatRef.tone`，
	 * 不填则"操作数灰 / 产物靛蓝"），`MatRef.split` 还能画成"几段拼起来的"条。
	 *
	 * 同一个张量在多个小组里出现时，用一条 SVG 曲线把相邻两次出现连起来，
	 * 于是"上一步的 Q 就是这一步的 Q"这种前后关系一眼可见；
	 * 其中**终点落在当前步方框上的那条**（= 这一步的输入从哪来）额外加箭头。
	 */
	import { onMount } from 'svelte';
	import type { OverviewItem } from '$lib/core/overview';
	import { itemKeys } from '$lib/core/overview';
	import MiniMatmul from './MiniMatmul.svelte';
	import MiniUnary from './MiniUnary.svelte';

	let { items, activeStepId, onpick }: {
		items: OverviewItem[];
		activeStepId: string;
		onpick?: (stepId: string) => void;
	} = $props();

	let flowEl: HTMLDivElement | undefined = $state();
	/**
	 * 量出来的全部复用连线（每个复用张量的相邻两次出现各一条）。
	 *
	 * 只把**终点落在当前步方框上**的那些画出来，也就是"这一步的输入是从哪来的"，
	 * 于是任何时刻最多两条线（如算分数那一步：`Q → Q_h` 与 `K → K_hᵀ`）。
	 */
	let paths = $state<{ key: string; d: string; toStep: string }[]>([]);
	const shown = $derived(paths.filter((p) => p.toStep === activeStepId));

	/** 一个方框锚点（坐标相对 `.flow` 容器） */
	type OvBox = {
		key: string;
		stepId: string;
		left: number;
		right: number;
		top: number;
		bottom: number;
	};
	const midY = (b: OvBox) => (b.top + b.bottom) / 2;

	/** 判碰撞时把方框往里缩一点：**贴着边走不算压住**，免得把本来好看的曲线赶去绕远路 */
	const INSET = 3;
	/** 走廊（横向）的最小高度——比这窄的缝隙不拿来走线 */
	const MIN_BAND = 12;
	/** 同一条走廊上多条曲线错开的间距 */
	const LANE = 6;
	/** 绕行折线的圆角半径 */
	const CORNER = 6;

	const f = (v: number) => v.toFixed(1);

	/** 把一条三次贝塞尔采样成点（判"这条线会不会从方框上压过去"用） */
	function sample(x1: number, y1: number, mx: number, x2: number, y2: number) {
		const pts: { x: number; y: number }[] = [];
		const N = 24;
		for (let i = 0; i <= N; i++) {
			const t = i / N;
			const u = 1 - t;
			pts.push({
				x: u * u * u * x1 + 3 * u * u * t * mx + 3 * u * t * t * mx + t * t * t * x2,
				y: u * u * u * y1 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t * t * t * y2
			});
		}
		return pts;
	}

	/**
	 * 横向"走廊"：所有方框 y 区间的补集（够宽的）。
	 *
	 * 走廊里没有任何方框，所以线沿着走廊横着走一定不会压到东西——**横段不需要再判碰撞**，
	 * 只要管好两端的竖段。同排的两个方框之间必然隔着别的方框（操作数在左、产物在右，
	 * 复用连线只能横穿产物），所以横段必须抬到走廊里走，不能贴着方框中线走。
	 */
	function corridors(all: OvBox[], flowH: number) {
		const iv = all.map((b) => [b.top, b.bottom] as [number, number]).sort((p, q) => p[0] - q[0]);
		const out: { y: number; h: number }[] = [];
		let cur = 0;
		for (const [t, bt] of iv) {
			if (t - cur >= MIN_BAND) out.push({ y: (cur + t) / 2, h: t - cur });
			cur = Math.max(cur, bt);
		}
		if (flowH - cur >= MIN_BAND) out.push({ y: (cur + flowH) / 2, h: flowH - cur });
		return out.length ? out : [{ y: flowH / 2, h: flowH }];
	}

	/** 竖着从 y1 走到 y2（x 固定）会不会压到方框 */
	function vertBlocked(x: number, y1: number, y2: number, all: OvBox[], skip: OvBox) {
		const lo = Math.min(y1, y2);
		const hi = Math.max(y1, y2);
		return all.some(
			(o) =>
				o !== skip &&
				x > o.left + INSET &&
				x < o.right - INSET &&
				lo < o.bottom - INSET &&
				hi > o.top + INSET
		);
	}

	/** S 曲线（默认走法）会不会压到别的方框 */
	function blocked(a: OvBox, b: OvBox, all: OvBox[], x1: number, y1: number, mx: number, x2: number, y2: number) {
		return sample(x1, y1, mx, x2, y2).some((p) =>
			all.some(
				(o) =>
					o !== a &&
					o !== b &&
					p.x > o.left + INSET &&
					p.x < o.right - INSET &&
					p.y > o.top + INSET &&
					p.y < o.bottom - INSET
			)
		);
	}

	/** 折线 + 圆角（绕行路线用；直角硬拐会显得像框线） */
	function rounded(pts: { x: number; y: number }[]) {
		const p = pts.filter((q, i) => i === 0 || Math.hypot(q.x - pts[i - 1].x, q.y - pts[i - 1].y) > 0.5);
		if (p.length < 2) return '';
		let d = `M ${f(p[0].x)} ${f(p[0].y)}`;
		for (let i = 1; i < p.length - 1; i++) {
			const prev = p[i - 1];
			const cur = p[i];
			const next = p[i + 1];
			const d1 = Math.hypot(cur.x - prev.x, cur.y - prev.y);
			const d2 = Math.hypot(next.x - cur.x, next.y - cur.y);
			const r = Math.min(CORNER, d1 / 2, d2 / 2);
			d += ` L ${f(cur.x + ((prev.x - cur.x) / d1) * r)} ${f(cur.y + ((prev.y - cur.y) / d1) * r)}`;
			d += ` Q ${f(cur.x)} ${f(cur.y)} ${f(cur.x + ((next.x - cur.x) / d2) * r)} ${f(cur.y + ((next.y - cur.y) / d2) * r)}`;
		}
		d += ` L ${f(p[p.length - 1].x)} ${f(p[p.length - 1].y)}`;
		return d;
	}

	/**
	 * 绕行走法（"直角 + 圆角"，不用贝塞尔弧）：
	 *
	 *   起点侧边 → 竖向走到走廊 → 沿走廊横穿 → 竖向走到终点侧边
	 *
	 * 横段一定在走廊里（不可能压方框），所以只需**逐个试竖段的落点**：
	 * 起点的右侧 / 左侧、终点的左侧 / 右侧，哪个竖着走不会压到方框就用哪个。
	 * 弧线做不到这一点——同排相邻组之间只隔 3px，弧线还没来得及拐弯就蹭到旁边的方框了；
	 * 从方框**上边**直接往上走也会穿过正上方的权重方框（`K` 上面就是 `W_K`），
	 * 所以落点必须试出来而不是算出来。
	 *
	 * 走廊也按"离两点中点近"的顺序试，都不行就返回 `null`（调用方退回 S 曲线）。
	 */
	function detour(
		a: OvBox,
		b: OvBox,
		all: OvBox[],
		bands: { y: number; h: number }[],
		lanes: Map<number, number>
	) {
		for (const band of bands) {
			const lane = lanes.get(band.y) ?? 0;
			const off = Math.min(LANE, Math.max(0, (band.h - 6) / 2));
			const cy = band.y + ((lane % 3) - 1) * off;
			// 从"朝向对方"的那条边出去/进来（换行后的连线是往左走的，别绕一圈）
			const rightward = b.left >= a.left;
			const exits = rightward ? [a.right + 2, a.left - 4] : [a.left - 4, a.right + 2];
			const entries = rightward ? [b.left - 4, b.right + 4] : [b.right + 4, b.left - 4];
			for (const ex of exits) {
				if (vertBlocked(ex, midY(a), cy, all, a)) continue;
				for (const en of entries) {
					if (vertBlocked(en, cy, midY(b), all, b)) continue;
					lanes.set(band.y, lane + 1);
					// 终点让开 2px 给箭头；从右边进来时就停在右边
					const end = rightward ? b.left - 2 : b.right + 2;
					return rounded([
						{ x: ex, y: midY(a) },
						{ x: ex, y: cy },
						{ x: en, y: cy },
						{ x: en, y: midY(b) },
						{ x: end, y: midY(b) }
					]);
				}
			}
		}
		return null;
	}

	/**
	 * 量出每个方框锚点的位置，把同一张量的**相邻两次出现**连成曲线。
	 *
	 * 锚点是 `MiniBox` 上的 `data-ov-node`（值 = `refKey`，跨步骤复用的张量用 `handoffId`
	 * 归一到同一个 key，如 MHA 的 `K` 与 `K_hᵀ` 都是 `k`）。**这条属性丢了曲线就静默消失**，
	 * 所以这里额外做一次对账：数据上明明有复用（某个 key 出现 ≥2 次）却一条锚点都没量到，
	 * 说明锚点被删了，直接报出来而不是安静地不画。
	 */
	function redraw() {
		if (!flowEl || typeof window === 'undefined') return;
		const base = flowEl.getBoundingClientRect();
		const all: OvBox[] = [];

		flowEl.querySelectorAll<HTMLElement>('[data-ov-node]').forEach((el) => {
			const key = el.dataset.ovNode;
			if (!key) return;
			const r = el.getBoundingClientRect();
			if (r.width === 0) return;
			all.push({
				key,
				stepId: el.dataset.ovStep ?? '',
				left: r.left - base.left,
				right: r.right - base.left,
				top: r.top - base.top,
				bottom: r.bottom - base.top
			});
		});

		const byKey = new Map<string, OvBox[]>();
		for (const b of all) {
			const list = byKey.get(b.key);
			if (list) list.push(b);
			else byKey.set(b.key, [b]);
		}

		/**
		 * 每个步骤在流程里的先后次序。
		 *
		 * 连线必须按**步骤顺序**连，不能按 x 坐标——总览会换行（`flex-wrap`），
		 * 换行后"后面的步骤"反而在左边（MoE 的 `r`：路由步在第 1 行右端、
		 * 选专家步在第 2 行左端）。按 x 排会把箭头指向**还没算出来**的那个方框，
		 * "来源"就反了。
		 */
		const order = new Map(items.map((it, i) => [it.stepId, i]));

		const bands = corridors(all, base.height);
		/** 每条走廊已经排了几条线——同一走廊里的线要错开，否则会叠在一起 */
		const lanes = new Map<number, number>();
		const out: { key: string; d: string; toStep: string }[] = [];

		for (const [key, list] of byKey) {
			if (list.length < 2) continue;
			list.sort(
				(a, b) =>
					(order.get(a.stepId) ?? 0) - (order.get(b.stepId) ?? 0) || a.left - b.left
			);
			for (let i = 1; i < list.length; i++) {
				const a = list[i - 1];
				const b = list[i];
				const x1 = a.right;
				const y1 = midY(a);
				// 终点往回让 2px，给箭头留位置（否则箭头会贴在方框边上）
				const x2 = b.left - 2;
				const y2 = midY(b);
				const mx = (x1 + x2) / 2;
				// 不压方框就直接连 S 曲线；压上了才绕走廊，走廊按"离两点中点近"排序
				const straight = `M ${f(x1)} ${f(y1)} C ${f(mx)} ${f(y1)} ${f(mx)} ${f(y2)} ${f(x2)} ${f(y2)}`;
				const mid = (y1 + y2) / 2;
				const byNear = [...bands].sort((p, q) => Math.abs(p.y - mid) - Math.abs(q.y - mid));
				const d = blocked(a, b, all, x1, y1, mx, x2, y2)
					? (detour(a, b, all, byNear, lanes) ?? straight)
					: straight;
				out.push({ key, toStep: b.stepId, d });
			}
		}
		paths = out;

		if (!out.length && expectedCurves > 0) {
			console.warn(
				`[流程总览] 数据上有 ${expectedCurves} 条复用连线，却一条都没画出来——` +
					'方框的 `data-ov-node` 锚点可能被删了（见 MiniBox.svelte）。'
			);
		}
	}

	/** 数据侧应该画出的曲线条数（= 每个复用 key 的「出现次数 − 1」之和） */
	const expectedCurves = $derived.by(() => {
		const n = new Map<string, number>();
		for (const it of items) {
			for (const k of itemKeys(it)) n.set(k, (n.get(k) ?? 0) + 1);
		}
		return [...n.values()].filter((c) => c > 1).reduce((s, c) => s + c - 1, 0);
	});

	onMount(() => {
		redraw();
		const ro = new ResizeObserver(() => redraw());
		if (flowEl) ro.observe(flowEl);
		return () => ro.disconnect();
	});

	// 小组数据变化（切插件/换层）后重画连线
	$effect(() => {
		items;
		requestAnimationFrame(redraw);
	});
</script>

<div class="overview">
	<div class="bar">
		<span class="title">流程总览</span>
		<span class="hint">
			矩阵乘组：左操作数在左下、右操作数在上；单目运算组：输入在左、输出在右 ·
			<b>当前这一步的输入从哪来，用带箭头的线画出</b> · 点方框跳到该步
		</span>
	</div>

	<div class="flow" bind:this={flowEl}>
		<svg class="links" aria-hidden="true">
			<defs>
				<marker
					id="ov-arrow"
					viewBox="0 0 8 8"
					refX="7"
					refY="4"
					markerWidth="5"
					markerHeight="5"
					orient="auto-start-reverse"
				>
					<path d="M0,0 L8,4 L0,8 z" fill="#6366f1"></path>
				</marker>
			</defs>
			{#each shown as p, i (p.key + ':' + i)}
				<path d={p.d} />
			{/each}
		</svg>

		{#each items as it, i (it.id)}
			{#if i > 0}<span class="arrow">→</span>{/if}
			{#if it.kind === 'matmul'}
				<MiniMatmul
					block={it}
					active={it.stepId === activeStepId}
					last={i === items.length - 1}
					{onpick}
				/>
			{:else}
				<MiniUnary
					item={it}
					active={it.stepId === activeStepId}
					last={i === items.length - 1}
					{onpick}
				/>
			{/if}
		{/each}
	</div>
</div>

<style>
	.overview {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
		padding: 0.6rem 0.8rem;
		background: #ffffff;
		border: 1px solid #e2e8f0;
		border-radius: 0.6rem;
	}
	.bar {
		display: flex;
		align-items: baseline;
		gap: 0.6rem;
		flex-wrap: wrap;
	}
	.title {
		font-size: 0.72rem;
		font-weight: 600;
		color: #334155;
	}
	.hint {
		font-size: 0.62rem;
		color: #94a3b8;
	}
	.hint b {
		color: #6366f1;
		font-weight: 600;
	}
	.flow {
		position: relative;
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: 0.2rem;
	}
	/* 连线画在节点下面 */
	.links {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
		z-index: 0;
	}
	/*
	 * 只画"当前步的来源"这一种线（见 `shown`）：靛蓝实线 + 箭头。
	 * 箭头是必须的——线是横向走的，光看线分不出前后，加箭头才知道"从哪来"。
	 * 例：点 `attn-score` 那一步的 `K_hᵀ`，`Q → Q_h` 与 `K → K_hᵀ` 两条带箭头指向本步；
	 * 点 `attn-k` 那一步的 `K`，箭头指向本组左下角的 `X`（K 的输入来自 `attn-q`）。
	 */
	.links path {
		fill: none;
		stroke: #6366f1;
		stroke-width: 2.2;
		marker-end: url(#ov-arrow);
	}
	.arrow {
		position: relative;
		z-index: 1;
		color: #cbd5e1;
		font-size: 0.9rem;
		font-weight: 700;
		padding-bottom: 1.6rem;
	}
</style>
