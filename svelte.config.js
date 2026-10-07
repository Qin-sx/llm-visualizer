import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	kit: {
		// GitHub Pages 项目页部署在 /llm-visualizer/ 子路径下，CI 工作流用 BASE_PATH 注入前缀；
		// 本地 dev/build 不设 BASE_PATH → 空 base，./dev.sh 的 localhost:9000 工作流不受影响。
		paths: { base: process.env.BASE_PATH || '' },
		adapter: adapter({ pages: 'build', assets: 'build', fallback: null, precompress: false }),
		prerender: { entries: ['/'] },
		alias: { $lib: './src/lib' }
	}
};

export default config;
