<script lang="ts">
	import type { Step, TensorView } from '$lib/core/types';
	import {
		animationUnits,
		fitCellSize,
		isAnimatable,
		progressOf,
		stepDurationMs
	} from '$lib/core/steps';
	import { untrack } from 'svelte';
	import Dataflow from './Dataflow.svelte';
	import Formula from './Formula.svelte';
	import Tensor from './Tensor.svelte';
	import Emph from './Emph.svelte';

	let {
		step,
		progress = 0,
		onTakeOver,
		onHandBack
	}: {
		step: Step;
		progress?: number;
		/** 本页 ▶ / ↺ 开始接管时调用（宿主据此**暂停**底部总播放器，否则它下一帧就把控制权抢回去） */
		onTakeOver?: () => void;
		/** 点"交还给底部总播放器"时调用，参数是"现在显示到本页的哪个比例"（宿主据此把总播放器挪过来） */
		onHandBack?: (stepRatio: number) => void;
	} = $props();

	/** 动画单元：并行行（`parallel`）里的多个视图合成一个单元一起播，其余各自一个 */
	const units = $derived(animationUnits(step));
	const animCount = $derived(units.length);

	// ── 页级播放控制：**一个**按钮驱动本页所有动画 ──────────
	const SPEEDS = [0.5, 1, 2];
	/** 1× 速度下播完本页所需时间：由本页动画工作量算出来（与底部总播放器同一套） */
	const BASE_MS = $derived(stepDurationMs(step));

	let localMode = $state(false);
	let localProgress = $state(0);
	let playing = $state(false);
	let speed = $state(1);
	let raf = 0;
	let lastTs = 0;

	/** 有效进度：本页接管后用本地进度，否则跟随底部播放条 */
	const eff = $derived(localMode ? localProgress : progress);

	// 底部播放条一旦开始推进，本页就交还控制权——否则页级"接管"状态会一直压着全局进度
	let lastSeen: number | undefined;
	$effect(() => {
		const p = progress;
		if (lastSeen === undefined) {
			lastSeen = p; // 首次只记录基线
			return;
		}
		if (p === lastSeen) return;
		lastSeen = p;
		untrack(() => {
			if (localMode) {
				localMode = false;
				playing = false;
				// **把本地进度对齐到"现在显示的位置"**：交还之后画面跟着全局走，
				// 本地进度如果还停在上次中断的地方，下次点 ▶ 就会从那个旧位置接着播——
				// 画面上会突然跳到很多格之后。
				localProgress = p;
			}
		});
	});

	function tick(ts: number) {
		if (!playing) return;
		// 钳制单帧推进量：页级播放按**墙钟** dt 推进（`localProgress += dt·speed/BASE_MS`），
		// 长任务（页面首次重排 / GC）后 dt 会一下子补上几百毫秒；钳到 50ms 让动画一格一格走。
		const dt = lastTs ? Math.min(0.05, (ts - lastTs) / 1000) : 0;
		lastTs = ts;
		localProgress = Math.min(1, localProgress + (dt * 1000 * speed) / BASE_MS);
		if (localProgress >= 1) {
			playing = false;
			raf = 0;
			return;
		}
		raf = requestAnimationFrame(tick);
	}

	/**
	 * 本页接管播放。
	 *
	 * 两件事都必须做，否则点 ▶ 会**跳一下**：
	 *   ① `onTakeOver()` → 让底部总播放器**停下来**。不停的话它下一帧就会推进全局进度，
	 *      下面那个 `$effect` 立刻把控制权抢回去 → 画面先跳到本地进度、再跳回全局进度。
	 *   ② `localProgress = eff` → **从"现在显示的位置"接着播**。不这么写的话，
	 *      本地进度还停在"上次中断的地方"（或初值 0），一点 ▶ 就跳到那儿去。
	 */
	function takeOver(from: number) {
		onTakeOver?.();
		localMode = true;
		localProgress = from;
	}

	function togglePlay() {
		if (playing) {
			playing = false;
			return;
		}
		const from = eff >= 1 ? 0 : eff; // 已经播完 → 从头再来
		takeOver(from);
		playing = true;
		lastTs = 0;
		raf = requestAnimationFrame(tick);
	}

	function reset() {
		takeOver(0);
		playing = false;
	}

	function followGlobal() {
		// 先把总播放器挪到"现在显示的位置"，再交还——否则交还的一瞬间画面会跳回去
		onHandBack?.(localProgress);
		localMode = false;
		playing = false;
	}

	$effect(() => {
		return () => {
			playing = false;
			if (raf) cancelAnimationFrame(raf);
		};
	});

	/** 本页有多个可动画视图时，进度切成等份依次播放；当前在第几段 */
	const segIndex = $derived(
		animCount > 0 ? Math.min(animCount - 1, Math.floor(eff * animCount)) : 0
	);

	/**
	 * 本页进度 `eff`（0..1）下某个视图的实际进度。
	 * 分段与"并行视图按工作量反比缩放"的规则都在 `core/steps.ts`，与 verify 共用一份。
	 */
	function subProgress(t: TensorView): number {
		if (!isAnimatable(t)) return 0;
		return progressOf(t, units, eff);
	}

	/**
	 * 扫描拼接驱动：找标了 `scan` 的视图（如窗口槽位打分），按它的 reveal 进度算出
	 * "正在填出来的那一行"，查 `scan.seq[r]` 得到来源**行块**（目标视图 + from/to），给目标视图
	 * 设 `scanRows`——于是源矩阵上出现一个随窗口 reveal **移动的、和拷贝行数一致高**的扫描框
	 * （一次拷 `ratio` 行 → 三个矩阵的框都是 `ratio` 行高）。
	 */
	const scanRows = $derived.by(() => {
		const map = new Map<string, [number, number]>();
		for (const t of step.tensors) {
			if (!t.scan?.seq) continue;
			const rows = t.shape[0] ?? 0;
			if (!rows) continue;
			const p = subProgress(t);
			const r = Math.min(rows - 1, Math.floor(p * rows));
			const entry = t.scan.seq[r];
			if (entry) map.set(entry.target, [entry.from, entry.to]);
		}
		return map;
	});

	/**
	 * 布局：先按 `row` 切成横向的"行"，行内再按 `group` 切成纵向堆叠的"列"。
	 * 没有 `row` 的视图各自独占一行。
	 */
	const layout = $derived.by(() => {
		const rows: {
			row?: number;
			parallel: boolean;
			vcenter: boolean;
			noHsep: boolean;
			cols: { group?: string; items: TensorView[] }[];
		}[] = [];
		for (const t of step.tensors) {
			let r = rows[rows.length - 1];
			if (t.row === undefined || !r || r.row !== t.row) {
				r = { row: t.row, parallel: !!t.parallel, vcenter: !!t.vcenter, noHsep: !!t.noHsep, cols: [] };
				rows.push(r);
			}
			if (t.vcenter) r.vcenter = true;
			if (t.noHsep) r.noHsep = true;
			let c = r.cols[r.cols.length - 1];
			if (t.group === undefined || !c || c.group !== t.group) {
				c = { group: t.group, items: [] };
				r.cols.push(c);
			}
			c.items.push(t);
		}
		return rows;
	});

	/** 这一列是否**自带行首箭头**（`hideInput` 的变换：输入由左边矩阵画，行首是它自己的引头）——
	 *  这时 `.hrow` 不再重复画 `→`（否则 `v̄ → RMSNorm ──▶ v̄~` 会冒出两个连续箭头）。 */
	function hasLeadArrow(items: TensorView[]): boolean {
		return items.some((t) => t.kind === 'transform' && !!t.hideInput);
	}

	/**
	 * 并排各列的**垂直对齐**（按"每列最后一块矩阵的顶边"对齐）。
	 *
	 * 一行里最常见的是"矩阵乘（2×2：B 在上、A/C 在下）+ 右边接一条链"：
	 * `X`、`[v | s]` 在下半行，而链上的矩阵在自己那一列的**顶部**——
	 * 于是链比 `X` / `[v | s]` 高出一整行。
	 *
	 * 纯 CSS 的顶对齐 / 底对齐都救不了：底对齐会被两列**各自下方的框**带偏
	 * （矩阵乘有"算式框 + 图注"，链有"明细框"，高度不同）。
	 * 这里就量出各列锚点，把锚点高的那列往下推 `max − anchor` 像素。
	 * 矩阵块结构一致（表头定高 + 网格 + 两行注解定高），所以"最后一块的顶边"对齐
	 * 就等于整块矩阵顶边对齐。
	 *
	 * 列里**没有矩阵**时（如 `v̄` 那条 `row`）锚点为空，就拿这一列的**列顶**去对：
	 * 内容顶部贴到邻居"最后一块矩阵"的顶边。**不能退回底对齐**——邻居下方还有
	 * "图注 + 算式框"，底对齐会把这一列一路推到页面很下面，和矩阵、箭头完全脱开。
	 *
	 * 同一轮里还给 `fillWidth` 的行算**格子边长**（见下），两件事都是"量 → 改 → 重量"，
	 * 就合成一个循环。
	 */
	let tensorsEl: HTMLElement | undefined = $state();
	/** 第 `ri` 行、第 `ci` 列需要往下推的像素 */
	let rowOffsets: number[][] = $state([]);
	/**
	 * 第 `ri` 行量出来的格子边长（只给标了 `fillWidth` 的行算）；`undefined` = 不接管。
	 *
	 * 一行**只有一个**边长：`X`、`[v | s]`、`ape`… 大小一致才读得出"它们是同一条链上的"。
	 */
	let rowCell: (number | undefined)[] = $state([]);

	/** 这一行要不要"放大到占满可用宽度"（行里任一视图标了 `fillWidth`） */
	function rowFills(r: (typeof layout)[number]): boolean {
		return r.cols.some((c) => c.items.some((t) => t.fillWidth));
	}

	/** 量一个元素的**内容宽度**：临时把宽度设成 `max-content`，量完立刻改回（同一 tick 内，不会闪）。 */
	function contentWidth(el: HTMLElement): number {
		const prev = el.style.width;
		el.style.width = 'max-content';
		const w = el.getBoundingClientRect().width;
		el.style.width = prev;
		return w;
	}

	$effect(() => {
		void layout; // 布局变了（换步骤 / 换行）就重新量
		const host = tensorsEl;
		if (!host) return;
		// `.tensors` 的直接子元素就是"行"，顺序与 `layout` 一一对应
		const rows = [...host.children] as HTMLElement[];
		if (!rows.length) return;

		const measure = () => {
			// **播放期间不量**：对齐（fillWidth 格子 / 注解定高 / 并排偏移）在挂载时已经量好，
			// 动画每帧只改矩阵**下方**的明细框（`.calc` 的 terms），不影响矩阵对齐。
			// 而每帧的内容变化会触发 ResizeObserver → 量 → 写样式 → 又触发——gate 页那种
			// 大页面上一次全量 getBoundingClientRect 强制重排能卡 1~2 秒。
			// `eff` 用 `untrack` 读：它每帧都在变，不能当依赖（否则这个 effect 每帧重跑）；
			// 只在被调用（挂载 / RO 回调 / 暂停）的那一刻读一次当前值。页级播放（`playing`）
			// 和底部播放（`eff` 推进中）都挡。暂停 / 播完时 `playing` 变 false、`eff` 回到
			// 0 或 1，这个 effect 会重跑、量一遍，不会漏掉布局变化。
			if (playing) return;
			if (untrack(() => eff) > 0 && untrack(() => eff) < 1) return;
			// ── ① fillWidth 的行：格子边长放大到"内容刚好占满可用宽度" ──
			// `applied` 从渲染出来的格子上读（`MatrixGrid` 每个格子的 `width` 就是边长），
			// 这样不用管这一行是矩阵乘还是变换、各自用的什么兜底值。
			const cells = rows.map((row, ri) => {
				if (!rowFills(layout[ri])) return undefined;
				const cell = row.querySelector<SVGElement>('svg.block rect.cell');
				const applied = cell ? Number(cell.getAttribute('width')) : NaN;
				if (!(applied > 0)) return undefined;
				// 行已经横向满了（内容 ≥ 容器）：fit 不可能再放大（满了就是最大），
				// 也不该缩小（矩阵已经到可读尺寸）——别再走昂贵的 `max-content` 测量
				// （300+ 个 SVG 格子的 max-content 重排一次能卡几百毫秒）。
				if (row.scrollWidth >= row.clientWidth - 1) return applied;
				// 留 4px 余量：正好顶满会把 `.hrow` 的横向滚动条挤出来，一出现就又得缩，来回抖
				return fitCellSize(applied, contentWidth(row), row.clientWidth - 4);
			});
			if (cells.some((c, i) => c !== rowCell[i])) rowCell = cells;

			// ── ② 矩阵**上方**那几行（名字 / 真实尺寸 / 来源）统一高度 ──
			// 注解放在矩阵上面、紧挨着名字；行数不同就会把矩阵顶边顶歪 → 取**同一网格里同一行**的最大高度给这些块用。
			//
			// 按"行"分组而不是按整个网格：矩阵乘那种 2×2 里第一行（`W_gate`）和第二行
			// （`X` / `[v | s]`）的注解高度**可以不同**，第一行矩阵上方就不会多出一段空隙。
			// 同一行里的矩阵顶边照样齐（块之间互相看不见，纯 CSS 做不到，只能量）。
			const groups = new Map<string, HTMLElement[]>();
			const gridNo = new Map<Element, number>();
			for (const el of host.querySelectorAll<HTMLElement>('.head, .real, .src')) {
				const block = el.parentElement;
				if (!block?.matches('.mblock, .side, .part')) continue;
				/*
				 * 子网格（`ConcatView` 的 `.parts`、`SumView` 的 `.terms`）本身只是**外层网格的一个格子**：
				 * 它里面的块和外层的兄弟块在**同一视觉行**上，必须和外层一起取高度。
				 *
				 * 不然就是"子网格里的块窄 → 注解折行多 → 它的矩阵被顶下去"：
				 * 所以一路向上走到**最外层的那个布局网格**再分组。
				 */
				let grid = block.closest('.matrices, .row, .parts');
				while (grid) {
					const outer = grid.parentElement?.closest('.matrices, .row, .parts');
					if (!outer || !outer.contains(grid)) break;
					grid = outer;
				}
				let where = 'standalone';
				if (grid) {
					if (!gridNo.has(grid)) gridNo.set(grid, gridNo.size);
					// 块在最外层网格里所属的**格子**（可能是子网格本身，如 `.parts`）。
					// "第几行"按**顶边**判定：`grid-row-start` 在这里算出来也是 `auto`。
					let item: HTMLElement | null = block;
					while (item && item.parentElement !== grid) item = item.parentElement;
					where = `${gridNo.get(grid)}#${
						item ? Math.round(item.getBoundingClientRect().top) : 'x'
					}`;
				}
				const cls = el.classList.contains('head')
					? 'head'
					: el.classList.contains('real')
						? 'real'
						: 'src';
				const key = `${where}#${cls}`;
				const list = groups.get(key);
				if (list) list.push(el);
				else groups.set(key, [el]);
			}
			for (const els of groups.values()) {
				// 先回到自然高度再量（已经设过的话，量到的就是上一轮的固定值）
				for (const el of els) el.style.height = '';
				const h = Math.max(...els.map((el) => el.getBoundingClientRect().height));
				for (const el of els) el.style.height = `${h}px`;
			}

			// ── ③ 并排各列的垂直对齐（按"每列最后一块矩阵的顶边"对齐）──
			// 放在 ② **之后**：注解定高会改块的内部高度、矩阵顶边跟着动，
			// 先对再改就等于拿旧位置对了一遍。
			const next = rows.map((row, ri) => {
				const cols = [...row.querySelectorAll<HTMLElement>(':scope > .hcol')];
				if (cols.length < 2) return [];
				// 已经加上的偏移要从量到的位置里扣掉，否则"量 → 改 → 再量"会自己滚自己
				const applied = cols.map((c) => parseFloat(c.style.marginTop) || 0);
				const tops = cols.map((col, i) => col.getBoundingClientRect().top - applied[i]);
				const blocks = cols.map((col, i) =>
					[...col.querySelectorAll<SVGElement>('svg.block')].map((b) => {
						const r = b.getBoundingClientRect();
						return { top: r.top - applied[i], bottom: r.bottom - applied[i] };
					})
				);
				// `vcenter`：这一行改成**垂直居中**——每列内容（第一块顶 到 最后一块底）的中心
				// 对齐到"内容跨度最大那一列"的中心。用于"右边一块要夹在左边两块中间"的场合
				// （如 `-sape` 页：窗口槽位打分垂直居中于 s_overlap 链和 s 链之间）。
				if (layout[ri]?.vcenter) {
					let maxSpan = 0;
					let tTop = 0;
					for (const bs of blocks) {
						if (!bs.length) continue;
						const span = bs[bs.length - 1].bottom - bs[0].top;
						if (span > maxSpan) {
							maxSpan = span;
							tTop = bs[0].top;
						}
					}
					if (!maxSpan) return cols.map(() => 0);
					const targetCenter = tTop + maxSpan / 2;
					return blocks.map((bs, i) => {
						if (!bs.length) return +((targetCenter - tops[i]).toFixed(2));
						const span = bs[bs.length - 1].bottom - bs[0].top;
						return +(targetCenter - (bs[0].top + span / 2)).toFixed(2);
					});
				}
				const anchors = blocks.map((bs) => (bs.length ? bs[bs.length - 1].top : null));
				const real = anchors.filter((a): a is number => a !== null);
				if (!real.length) return cols.map(() => 0);
				const max = Math.max(...real);
				// 保留小数：取整会让两列差出 1px（量到的是小数，矩阵网格边就是小数像素）
				//
				// 列里**没有矩阵**时（如 `v̄` 那条 `row`）拿它的**列顶**去对锚点：
				// 内容顶部贴到邻居"最后一块矩阵"的顶边，箭头指的就是它。
				return anchors.map((a, i) => +((a === null ? max - tops[i] : max - a)).toFixed(2));
			});
			const same =
				next.length === rowOffsets.length &&
				next.every(
					(a, i) =>
						a.length === rowOffsets[i]?.length && a.every((v, j) => v === rowOffsets[i][j])
				);
			if (!same) rowOffsets = next;

			// ── ④ 横向箭头（`→`）的垂直定位 ──────────────────────────
			// harrow 默认 `align-self: center` 居中于整行，行被最高列（如 2×2 矩阵乘）撑起时
			// 会悬空；这里量相邻两列**最后一块矩阵**中心的均值，把箭头挪过去。
			rows.forEach((row, ri) => {
				const arrows = [...row.querySelectorAll<HTMLElement>(':scope > .harrow')];
				if (!arrows.length) return;
				const cols = [...row.querySelectorAll<HTMLElement>(':scope > .hcol')];
				if (cols.length < 2) return;
				// 用 ③ 刚算出的列偏移（`next`）预测矩阵落点（`rowOffsets` 还未回流）
				const expect = next[ri] ?? [];
				const blocks = cols.map((col, i) => {
					const applied = parseFloat(col.style.marginTop) || 0;
					const shift = expect[i] ?? applied;
					return [...col.querySelectorAll<SVGElement>('svg.block')].map((b) => {
						const r = b.getBoundingClientRect();
						return { top: r.top - applied + shift, bottom: r.bottom - applied + shift };
					});
				});
				arrows.forEach((ar, ai) => {
					const a = blocks[ai];
					const b = blocks[ai + 1];
					if (!a?.length || !b?.length) return;
					const centerA = (a[a.length - 1].top + a[a.length - 1].bottom) / 2;
					const centerB = (b[b.length - 1].top + b[b.length - 1].bottom) / 2;
					const target = (centerA + centerB) / 2;
					// 基于"当前没加 margin 的位置"算偏移，重复量不会自己滚自己
					const rawTop = ar.getBoundingClientRect().top - (parseFloat(ar.style.marginTop) || 0);
					const h = ar.getBoundingClientRect().height;
					const mt = target - (rawTop + h / 2);
					if (Math.abs(mt - (parseFloat(ar.style.marginTop) || 0)) > 0.05)
						ar.style.marginTop = `${mt.toFixed(2)}px`;
				});
			});
		};

		// 挂载时**不立即量**：同步 measure 会把强制重排压进"点击翻页"的长任务，挪到下一帧再量，
		// 对齐在 fade-in 期间补上（fillWidth 的 fit 本来就是默认格子，不改变首次显示）。
		const raf = requestAnimationFrame(measure);
		// 再量一帧：② 改的是块内部的高度，可能不改变"行"本身的外框尺寸
		// （`.hrow` 宽度固定），那样 `ResizeObserver` 不会回调，得自己补一次
		/*
		 * `ResizeObserver` 回调**每帧最多折算成一次 `measure()`**：`measure()` 会写样式、自己又观察
		 * 同一行，"回调 → 改布局 → 又回调"可能抖动雪崩（分像素取整）；折到 rAF 上每帧最多一次。
		 */
		/*
		 * 字体（KaTeX 异步加载）就绪后补量一次：rAF 那下可能用回退字体量出错的对齐；
		 * 首次进入应用本身也在忙，挂载后量到的可能是过渡值——等 `document.fonts.ready`
		 * / 浏览器空闲（`requestIdleCallback`，没有 rIC 用延迟兜底）再量一次兜住。
		 */
		let alive = true;
		let idleId: number | undefined;
		let settleTimer: number | undefined;
		const realign = () => {
			if (alive) requestAnimationFrame(measure);
		};
		if (typeof document !== 'undefined') {
			if (document.fonts?.ready) document.fonts.ready.then(realign);
			if (typeof requestIdleCallback !== 'undefined') {
				idleId = requestIdleCallback(realign, { timeout: 1200 });
			} else {
				settleTimer = window.setTimeout(realign, 600);
			}
		}
		let pending = 0;
		const schedule = () => {
			if (pending) return;
			pending = requestAnimationFrame(() => {
				pending = 0;
				measure();
			});
		};
		const ro = new ResizeObserver(schedule);
		for (const row of rows) ro.observe(row);
		return () => {
			alive = false;
			cancelAnimationFrame(raf);
			if (pending) cancelAnimationFrame(pending);
			if (idleId !== undefined && typeof cancelIdleCallback !== 'undefined') cancelIdleCallback(idleId);
			if (settleTimer !== undefined) clearTimeout(settleTimer);
			ro.disconnect();
		};
	});
</script>

{#key step.id}
	<div class="step-panel">
		<div class="head">
			<span class="kind">{step.kind}</span>
			<span class="text-sm font-medium text-slate-800"><Emph text={step.label} /></span>
		</div>

		{#if step.formula}
			<div class="formula-row">
				<Formula tex={step.formula} />
			</div>
		{/if}

		{#if animCount > 0}
			<div class="ctrl-bar">
				<button class="play" onclick={togglePlay}>{playing ? '⏸ 暂停' : '▶ 播放'}</button>
				<button class="mini" onclick={reset} title="重置">↺</button>

				<span class="spd">速度</span>
				{#each SPEEDS as s (s)}
					<button class="chip" class:active={speed === s} onclick={() => (speed = s)}>{s}×</button>
				{/each}

				<div class="track"><div class="fill" style={`width:${eff * 100}%`}></div></div>

				{#if animCount > 1}
					<span class="segs">第 {segIndex + 1} / {animCount} 段</span>
				{/if}
				{#if localMode}
					<button class="mini" onclick={followGlobal} title="交还给底部总播放器">↩</button>
				{/if}
			</div>
		{/if}

		<div class="tensors" bind:this={tensorsEl}>
			{#each layout as r, ri (ri)}
				{#if r.cols.length > 1}
					<!-- 横向并排：流水线（→）或并行分支（浅色虚线隔开） -->
					<div class="hrow" class:parallel={r.parallel}>
						{#each r.cols as c, ci (ci)}
							{#if ci > 0}
								{#if r.parallel && !r.noHsep}
								<!--
									并行分支之间画一条**浅色虚线**，而不是 `→`：这两块的运算是**各算各的**
									（如分组输出投影的 `o^(0)` / `o^(1)`、MoE 里两个专家 E0 / E2），
									只是并排放在一起，没有先后。标了 `noHsep` 的并行行（相关计算，如
									`-sape` 的 s_overlap / s 同时加 ape 但汇向同一个窗口槽位打分）改画箭头。
								-->
									<span class="hsep" aria-hidden="true"></span>
								{:else if hasLeadArrow(r.cols[ci].items)}
								<!--
									这一列是 `hideInput` 的变换（输入由左边矩阵画）：它行首**自己带引头**
									（如 `RMSNorm ──▶`），外面再画一个 `→` 就成了两个连续箭头。
								-->
									<span aria-hidden="true"></span>
								{:else}
									<span class="harrow">→</span>
								{/if}
							{/if}
							<div class="hcol" style:margin-top={`${rowOffsets[ri]?.[ci] ?? 0}px`}>
								{#each c.items as t (t.name)}
									<Tensor view={t} progress={subProgress(t)} compact cellSize={rowCell[ri]} scanRows={scanRows.get(t.name)} />
								{/each}
							</div>
						{/each}
					</div>
				{:else}
					{#each r.cols[0].items as t (t.name)}
						<Tensor view={t} progress={subProgress(t)} cellSize={rowCell[ri]} scanRows={scanRows.get(t.name)} />
					{/each}
				{/if}
			{/each}
		</div>

		<!-- 参考数据流图：静态地图，放在本页最下面（不参与动画计时） -->
		{#if step.diagram}
			<Dataflow spec={step.diagram} />
		{/if}
	</div>
{/key}

<style>
	.step-panel {
		animation: fade-in 0.35s ease-out;
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}
	.head {
		display: flex;
		align-items: baseline;
		gap: 0.5rem;
	}
	.kind {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.6rem;
		letter-spacing: 0.06em;
		padding: 0.1rem 0.35rem;
		border-radius: 0.25rem;
		background: #eef2ff;
		color: #4338ca;
		border: 1px solid #c7d2fe;
	}
	.formula-row {
		background: #f8fafc;
		border: 1px solid #e2e8f0;
		border-radius: 0.375rem;
		padding: 0.5rem 0.75rem;
		overflow-x: auto;
	}

	/* 页级播放控制：一个按钮驱动本页所有动画 */
	.ctrl-bar {
		display: flex;
		align-items: center;
		gap: 0.35rem;
		flex-wrap: wrap;
		padding: 0.4rem 0.5rem;
		background: #ffffff;
		border: 1px solid #e2e8f0;
		border-radius: 0.5rem;
	}
	.play {
		padding: 0.28rem 0.7rem;
		border-radius: 0.375rem;
		border: 1px solid #4f46e5;
		background: #4f46e5;
		color: #fff;
		font-size: 0.75rem;
		cursor: pointer;
		white-space: nowrap;
	}
	.play:hover {
		background: #4338ca;
	}
	.mini {
		padding: 0.22rem 0.45rem;
		border-radius: 0.3rem;
		border: 1px solid #e2e8f0;
		background: #f8fafc;
		color: #64748b;
		font-size: 0.72rem;
		cursor: pointer;
	}
	.mini:hover {
		background: #f1f5f9;
	}
	.spd {
		font-size: 0.68rem;
		color: #94a3b8;
		margin-left: 0.25rem;
	}
	.chip {
		padding: 0.18rem 0.42rem;
		border-radius: 999px;
		border: 1px solid #e2e8f0;
		background: #fff;
		color: #64748b;
		font-size: 0.68rem;
		cursor: pointer;
	}
	.chip.active {
		background: #eef2ff;
		border-color: #c7d2fe;
		color: #4338ca;
	}
	.track {
		flex: 1 1 6rem;
		min-width: 4rem;
		height: 0.3rem;
		border-radius: 999px;
		background: #e2e8f0;
		overflow: hidden;
	}
	.fill {
		height: 100%;
		background: #6366f1;
		transition: width 0.08s linear;
	}
	.segs {
		font-size: 0.66rem;
		color: #94a3b8;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		white-space: nowrap;
	}

	.tensors {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
		gap: 1.4rem 1rem;
		align-items: start;
	}
	/* 矩阵乘 / 掩码等视图要占满整行——矩阵必须按 2×2 排布，不能被挤到换行 */
	.tensors :global(.tensor.wide) {
		grid-column: 1 / -1;
	}

	/*
	 * 横向流水线 / 并行分支。
	 *
	 * `align-items: flex-start`：**顶对齐**，然后由脚本量出每列该往下推多少
	 * （见上面 `rowOffsets` 的说明）——两列的"最后一块矩阵顶边"对齐。
	 *
	 * 不能用 `flex-end`（底对齐）：两列**下方各有自己的框**（矩阵乘是"算式框 + 图注"、
	 * 链是"明细框"），高度不同，底对齐会把矩阵错开 33px。
	 */
	.hrow {
		grid-column: 1 / -1;
		display: flex;
		align-items: flex-start;
		gap: 0.55rem;
		overflow-x: auto;
		padding-bottom: 0.4rem;
	}
	/* 并行分支之间不画箭头，用更大的间距表达"同时进行" */
	.hrow.parallel {
		gap: 0.8rem;
	}
	/*
	 * 并行分支之间的分隔虚线（代替 `→`）。
	 *
	 * `flex: 0 0 0` → 宽度 0，只有 `border-left` 那 1px；`align-self: stretch` 让它**撑满整行高度**
	 * （容器是 `align-items: flex-start`，别的块顶对齐，只有这条线拉伸）。
	 * 颜色取浅灰（`#cbd5e1`）——是"这里断开"的提示，不该抢矩阵的注意力。
	 */
	.hsep {
		flex: 0 0 0;
		align-self: stretch;
		border-left: 1px dashed #cbd5e1;
	}
	.hcol {
		display: flex;
		flex-direction: column;
		gap: 0.8rem;
		/*
		 * **可以被压缩**。
		 *
		 * 一行的宽度来源是每块的**注解行**（`.real` / `.src`，长句子 max-content 能到 300~500px），
		 * 不是矩阵——矩阵是固定像素的，本来就小。不让压缩的话，并排两块的注解宽度直接相加，
		 * 一行就顶出去。
		 *
		 * 改成可压缩 + `min-width: min-content` 之后：**装得下就一点不变**（不触发收缩），
		 * 装不下才把注解折行（矩阵是固定像素的，仍是它的 `min-content` 下界，不会被挤小、
		 * 更不会被缩到 `MIN_TEXT_CELL` 以下）。MoE 那几个并排行本来就不溢出，所以完全不受影响。
		 */
		flex-shrink: 1;
		min-width: min-content;
	}
	/* 最后一列把**剩下的宽度吃掉**（同各视图内部 grid 的"最后一列 1fr"，原则 ⑤）：
	   右侧没有别的块时，这一列的注解（如 `-indexer-logits` 的 top-k 选择说明）可以一路延伸到行尾，
	   而不是被 `min-content` 卡在"最大单词宽度"上折成好几行 */
	.hcol:last-child {
		flex-grow: 1;
	}
	.harrow {
		align-self: center;
		color: #cbd5e1;
		font-size: 1.1rem;
		font-weight: 700;
		flex-shrink: 0;
	}

	@keyframes fade-in {
		from {
			opacity: 0;
			transform: translateY(4px);
		}
		to {
			opacity: 1;
			transform: none;
		}
	}
</style>
