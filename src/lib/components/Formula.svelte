<script lang="ts">
	import katex from 'katex';
	import 'katex/dist/katex.min.css';

	let { tex }: { tex: string } = $props();

	const html = $derived(
		katex.renderToString(tex, { throwOnError: false, displayMode: false, output: 'html' })
	);
</script>

<span class="formula">{@html html}</span>

<style>
	.formula {
		display: inline-block;
		color: #4338ca;
	}
	.formula :global(.katex) {
		font-size: 1em;
	}
</style>
