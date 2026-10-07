<script lang="ts">
	import { base } from '$app/paths';
	// 纯静态页：列出随产物分发的第三方软件及其许可。
	// 存在的原因：构建会压缩 JS，把依赖里自带的版权声明（尤其是 GSAP 的）
	// 一并去掉，而 GSAP 的许可条款明确要求「不得移除 proprietary notices」。
	type Dep = { name: string; version: string; license: string; url?: string; note?: string };

	/** 会打进构建产物、随页面一起分发的运行时依赖 */
	const runtime: Dep[] = [
		{ name: 'svelte', version: '5.57.1', license: 'MIT', url: 'https://github.com/sveltejs/svelte' },
		{
			name: '@sveltejs/kit',
			version: '2.70.3',
			license: 'MIT',
			url: 'https://github.com/sveltejs/kit'
		},
		{ name: 'katex', version: '0.16.47', license: 'MIT', url: 'https://github.com/KaTeX/KaTeX' },
		{
			name: 'gsap',
			version: '3.15.0',
			license: "Standard 'no charge' license",
			url: 'https://gsap.com/standard-license/',
			note: '商用免费，但须保留其版权声明、不得用于制作竞品'
		}
	];

	/** 只在构建期使用、不进入产物的开发依赖（按 MIT / ISC / Apache-2.0 授权） */
	const buildOnly = ['vite', 'tailwindcss', 'typescript', 'postcss', 'autoprefixer', 'svelte-check'];
</script>

<svelte:head>
	<title>第三方许可 — LLM Visualizer</title>
	<meta name="robots" content="noindex" />
</svelte:head>

<main>
	<header>
		<h1>第三方许可</h1>
		<p>LLM Visualizer 基于以下开源软件构建，在此致谢。</p>
	</header>

	<section>
		<h2>随产物分发的运行时依赖</h2>
		<table>
			<thead>
				<tr><th>软件</th><th>版本</th><th>许可</th></tr>
			</thead>
			<tbody>
				{#each runtime as d (d.name)}
					<tr>
						<td>
							<a href={d.url} target="_blank" rel="noreferrer">{d.name}</a>
							{#if d.note}<span class="note">{d.note}</span>{/if}
						</td>
						<td class="mono">{d.version}</td>
						<td>{d.license}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</section>

	<section class="callout">
		<h2>关于 GSAP</h2>
		<p>
			GSAP 采用 <a href="https://gsap.com/standard-license/" target="_blank" rel="noreferrer"
				>Standard “no charge” License</a
			>。该许可<strong>已覆盖商业用途，无需额外付费或申请</strong>，但要求：
		</p>
		<ul>
			<li>不得移除或篡改 GSAP 的版权声明与品牌标识；</li>
			<li>不得反向工程 GSAP 以制作与之竞争的产品。</li>
		</ul>
		<p class="muted">
			本页列出运行时依赖，正是为了满足第一条——构建压缩会丢掉依赖里自带的版权声明，故在此集中保留。
		</p>
	</section>

	<section>
		<h2>仅构建期使用（不随产物分发）</h2>
		<p class="muted">
			以下工具只在构建/校验时运行，不会进入最终页面，故无需随产物附带许可声明：
			<span class="mono">{buildOnly.join(' · ')}</span> 等，均以 MIT / ISC / Apache-2.0 授权。
		</p>
	</section>

	<p class="back"><a href={base + '/'}>← 返回 LLM Visualizer</a></p>
</main>

<style>
	main {
		max-width: 820px;
		margin: 0 auto;
		padding: 2.5rem 1.5rem 4rem;
	}
	header {
		margin-bottom: 2rem;
	}
	h1 {
		font-size: 1.6rem;
		margin: 0 0 0.4rem;
	}
	h2 {
		font-size: 0.95rem;
		color: #334155;
		margin: 0 0 0.75rem;
	}
	header p,
	.muted {
		color: #64748b;
		font-size: 0.85rem;
		margin: 0;
	}
	section {
		margin-bottom: 2rem;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		background: #ffffff;
		border: 1px solid #e2e8f0;
		border-radius: 0.6rem;
		overflow: hidden;
		font-size: 0.85rem;
	}
	th,
	td {
		text-align: left;
		padding: 0.55rem 0.8rem;
		border-bottom: 1px solid #f1f5f9;
	}
	th {
		font-size: 0.72rem;
		color: #64748b;
		background: #f8fafc;
	}
	tr:last-child td {
		border-bottom: none;
	}
	a {
		color: #4338ca;
		text-decoration: none;
	}
	a:hover {
		text-decoration: underline;
	}
	.mono {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.8rem;
		color: #475569;
	}
	.note {
		display: block;
		color: #94a3b8;
		font-size: 0.72rem;
		margin-top: 0.15rem;
	}
	.callout {
		background: #ffffff;
		border: 1px solid #e2e8f0;
		border-left: 3px solid #4f46e5;
		border-radius: 0.6rem;
		padding: 1rem 1.2rem;
	}
	.callout p {
		font-size: 0.85rem;
		margin: 0 0 0.6rem;
	}
	.callout ul {
		margin: 0 0 0.6rem;
		padding-left: 1.1rem;
		font-size: 0.85rem;
		color: #334155;
	}
	.callout li {
		margin-bottom: 0.25rem;
	}
	.back {
		font-size: 0.85rem;
	}
</style>
