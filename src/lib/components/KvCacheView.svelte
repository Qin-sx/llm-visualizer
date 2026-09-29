<script lang="ts">
	/**
	 * KV Cache 视图：把"缓存里到底躺着什么"摊开，并与另一种方案比体积。
	 *
	 *   每个 token 一行 —— 缓存是随着 token 一个一个填起来的
	 *   ┌─ MLA：只存潜向量 + 位置编码 ─────────┐        ┌─ MHA 要缓存 K 和 V ──────┐
	 *   │ c_KV [6 × 2]      k_pe' [6 × 2]     │        │ K [6 × 8]     V [6 × 8]  │
	 *   │ ▢▢ / ▢▢ / ...     ▢▢ / ▢▢ / ...     │   vs   │ ▢▢▢▢▢▢▢▢ / ...           │
	 *   │ 每 token 4 个数 · 真实 576           │        │ 每 token 16 个数 · 真实 32768│
	 *   └─────────────────────────────────────┘        └──────────────────────────┘
	 *
	 * 两边的格子**一样大**，所以"面积差"就是"缓存差"——这正是 MLA 的立身之本。
	 */
	import type { KvCacheView } from '$lib/core/types';
	import MatrixGrid from './MatrixGrid.svelte';

	let { view, progress = 0 }: { view: KvCacheView; progress?: number } = $props();

	const CELL = 22;
	/** 缓存里有几个 token（= 行数） */
	const tokens = $derived(view.blocks[0]?.data?.length ?? 0);
	/** 已填了几行 */
	const doneRows = $derived(Math.round(progress * tokens));
	/** 对比方案每 token 共几个数 */
	const comparePerToken = $derived(view.compare.parts.reduce((s, p) => s + p.perToken, 0));
	/** 对比侧已点亮的格子数（跟着一起长） */
	const compareDone = $derived(Math.round(progress * tokens * comparePerToken));

	const ratio = (a: number, b: number) => (b / a >= 10 ? Math.round(b / a) : (b / a).toFixed(1));
</script>

<div class="kv">
	<div class="head">
		<span class="title">KV Cache</span>
		<span class="sub">每个 token 追加一行，{tokens} 个 token 就是 {tokens} 行</span>
	</div>

	<div class="cols">
		<!-- 本方案 -->
		<div class="col mine">
			<div class="col-title">
				{view.mineTitle ?? 'MLA：只存潜向量 + 位置编码'}
				<span class="badge ok">省</span>
			</div>
			<div class="blocks">
				{#each view.blocks as b (b.name)}
					<div class="block">
						<div class="blk-head">
							<span class="nm">{b.name}</span>
							<!-- 行数按这一块自己的数据来：滑窗块与压缩块的行数可以不同 -->
							<span class="sz">[{b.data?.length ?? tokens} × {b.size}]</span>
						</div>
						<MatrixGrid data={b.data ?? []} cellSize={CELL} reveal={progress} />
						{#if b.label}<div class="blk-label">{b.label}</div>{/if}
					</div>
				{/each}
			</div>
			<div class="total">
				每 token <b>{view.perToken}</b> 个数
				<span class="real">真实 {view.realPerToken}</span>
			</div>
		</div>

		<div class="vs">
			<div class="vs-arrow">vs</div>
			<div class="vs-ratio">×{ratio(view.perToken, comparePerToken)}</div>
		</div>

		<!-- 对比方案：格子一样大，面积差就是缓存差 -->
		<div class="col theirs">
			<div class="col-title">
				{view.compare.label}
				<span class="badge bad">大</span>
			</div>
			<div class="blocks">
				{#each view.compare.parts as p (p.name)}
					<div class="block">
						<div class="blk-head">
							<span class="nm">{p.name}</span>
							<span class="sz">[{tokens} × {p.perToken}]</span>
						</div>
						<div class="cells" style={`--cols:${p.perToken};--cell:${CELL}px`}>
							{#each Array.from({ length: tokens * p.perToken }) as _, j (j)}
								<i class:on={j < compareDone}></i>
							{/each}
						</div>
						{#if p.sub}<div class="blk-label">{p.sub}</div>{/if}
					</div>
				{/each}
			</div>
			<div class="total">
				每 token <b>{comparePerToken}</b> 个数
				<span class="real">真实 {view.compare.realPerToken}</span>
			</div>
		</div>
	</div>

	<div class="bars">
		<div class="bar-row">
			<span class="bar-label">{view.mineLabel ?? 'MLA'}</span>
			<div class="bar-track">
				<div
					class="bar fill-mine"
					style={`width:${(view.perToken / comparePerToken) * 100}%`}
				></div>
			</div>
			<span class="bar-num">{view.perToken}</span>
		</div>
		<div class="bar-row">
			<span class="bar-label">MHA</span>
			<div class="bar-track"><div class="bar fill-theirs" style="width:100%"></div></div>
			<span class="bar-num">{comparePerToken}</span>
		</div>
		<div class="concl">
			真实模型里是 <b>{view.realPerToken}</b> vs <b>{view.compare.realPerToken}</b> →
			<b>{ratio(view.realPerToken, view.compare.realPerToken)}× 更小</b>
			<span class="hint"
				>（demo 的 d_model 只有 8，放不下真实那种量级差，所以这里只有 {ratio(
					view.perToken,
					comparePerToken
				)}×）</span
			>
		</div>
	</div>

	{#if view.note}
		<div class="note">{view.note}</div>
	{/if}
</div>

<style>
	.kv {
		display: flex;
		flex-direction: column;
		gap: 0.8rem;
	}
	.head {
		display: flex;
		align-items: baseline;
		gap: 0.5rem;
	}
	.title {
		font-size: 0.85rem;
		font-weight: 700;
		color: #1e293b;
	}
	.sub {
		font-size: 0.7rem;
		color: #94a3b8;
	}
	.cols {
		display: flex;
		align-items: stretch;
		gap: 0.9rem;
	}
	.col {
		flex: 1;
		min-width: 0;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		padding: 0.6rem 0.7rem;
		border-radius: 0.5rem;
	}
	.col.mine {
		background: #eef2ff;
		border: 1px solid #c7d2fe;
	}
	.col.theirs {
		background: #f8fafc;
		border: 1px dashed #e2e8f0;
	}
	.col-title {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		font-size: 0.74rem;
		font-weight: 600;
		color: #334155;
	}
	.badge {
		font-size: 0.6rem;
		padding: 0.02rem 0.3rem;
		border-radius: 0.2rem;
	}
	.badge.ok {
		background: #dcfce7;
		color: #15803d;
	}
	.badge.bad {
		background: #fee2e2;
		color: #b91c1c;
	}
	.blocks {
		display: flex;
		gap: 0.9rem;
		flex-wrap: wrap;
		align-items: flex-start;
	}
	.block {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
	}
	.blk-head {
		display: flex;
		align-items: baseline;
		gap: 0.3rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.nm {
		font-size: 0.76rem;
		font-weight: 700;
		color: #3730a3;
	}
	.theirs .nm {
		color: #64748b;
	}
	.sz {
		font-size: 0.62rem;
		color: #4f46e5;
	}
	.theirs .sz {
		color: #94a3b8;
	}
	.blk-label {
		font-size: 0.6rem;
		color: #64748b;
		max-width: 16rem;
		line-height: 1.4;
	}
	.theirs .blk-label {
		color: #94a3b8;
	}
	/* 对比方案的 K/V 用**同样大小**的格子：面积差就是缓存差 */
	.cells {
		display: grid;
		grid-template-columns: repeat(var(--cols), var(--cell));
		gap: 0.05rem;
	}
	.cells i {
		width: var(--cell);
		height: var(--cell);
		background: #fff;
		border: 0.6px solid #e2e8f0;
		opacity: 0.25;
		transition: opacity 0.15s;
	}
	.cells i.on {
		background: #e2e8f0;
		opacity: 1;
	}
	.total {
		font-size: 0.7rem;
		color: #475569;
	}
	.total b {
		color: #3730a3;
		font-size: 0.82rem;
	}
	.theirs .total b {
		color: #b91c1c;
	}
	.real {
		margin-left: 0.4rem;
		font-size: 0.62rem;
		color: #94a3b8;
	}
	.vs {
		align-self: center;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.1rem;
	}
	.vs-arrow {
		font-size: 0.72rem;
		color: #94a3b8;
		font-weight: 700;
	}
	.vs-ratio {
		font-size: 0.72rem;
		font-weight: 700;
		color: #15803d;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.bars {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		padding: 0.55rem 0.7rem;
		border-radius: 0.5rem;
		background: #fbfdff;
		border: 1px dashed #cbd5e1;
	}
	.bar-row {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	.bar-label {
		width: 2.4rem;
		font-size: 0.66rem;
		color: #64748b;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.bar-track {
		flex: 1;
		height: 0.7rem;
		border-radius: 0.15rem;
		background: #f1f5f9;
		overflow: hidden;
	}
	.bar {
		height: 100%;
		border-radius: 0.15rem;
		transition: width 0.2s;
	}
	.fill-mine {
		background: #6366f1;
	}
	.fill-theirs {
		background: #cbd5e1;
	}
	.bar-num {
		width: 3.4rem;
		text-align: right;
		font-size: 0.66rem;
		color: #475569;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.concl {
		font-size: 0.72rem;
		color: #334155;
		margin-top: 0.15rem;
	}
	.concl b {
		color: #15803d;
	}
	.hint {
		color: #94a3b8;
		font-size: 0.64rem;
	}
	.note {
		font-size: 0.7rem;
		line-height: 1.55;
		color: #475569;
		padding: 0.5rem 0.65rem;
		border-radius: 0.4rem;
		background: #f8fafc;
		border-left: 3px solid #a5b4fc;
	}
</style>
