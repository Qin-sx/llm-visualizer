<script lang="ts">
	/**
	 * 把文案里的 `**…**` 渲染成粗体。
	 * 不用 `{@html}`：按 `**` 切开交替渲染，天然没有注入面。
	 */
	let { text }: { text: string } = $props();

	const parts = $derived(text.split('**').map((t, i) => ({ text: t, strong: i % 2 === 1 })));
</script>

{#each parts as p, i (i)}{#if p.strong}<strong>{p.text}</strong>{:else}{p.text}{/if}{/each}
