/**
 * 随机权重初始化。
 *
 * 关键设计：权重由**插件自己声明**（`makeWeights`），model 层只负责遍历注册表并按层分配。
 *
 * 权重全部随机（固定 seed），因为本项目讲的是机制而非知识。
 *
 * **每个插件一条独立的随机流**（`seed ^ hash(插件 id)`）：
 * 按注册顺序共用一条流的话，**新增一个插件就会把它之后所有插件的权重全部挪位**，
 * 已有页面的数字、文档里记的值全都要重测。现在加插件只影响自己。
 */
import { mulberry32, randMat, type Mat } from '$lib/core/mat';
import { allImplementations, allSemantics } from '$lib/core/registry';
import { CONFIG } from './config';

export interface Weights {
	/** [vocab_size, d_model] —— Embedding 阶段的词表矩阵 */
	embed: Mat;
	/** [d_model, vocab_size] —— LM Head 的词表投影矩阵 */
	lmHead: Mat;
	/** 插件 id → 每层的权重包（即 `ctx.w`） */
	slots: Record<string, unknown[]>;
}

/** 由字符串派生一个稳定整数（FNV-1a），用于给每个插件分出独立的随机流 */
function hashSeed(s: string): number {
	let h = 0x811c9dc5;
	for (let i = 0; i < s.length; i++) {
		h ^= s.charCodeAt(i);
		h = Math.imul(h, 0x01000193);
	}
	return h >>> 0;
}

/** 某条命名随机流的种子 */
export function streamSeed(seed: number, name: string): number {
	return (seed ^ hashSeed(name)) >>> 0;
}

export function initWeights(seed: number = CONFIG.seed): Weights {
	const slots: Record<string, unknown[]> = {};
	// 语义插件与**自带权重的实现插件**各分一条流（实现一般不需要自己的权重，
	// 如 MLA 的吸收合并用的就是语义插件那批权重现场合并出来的矩阵）
	const owners = [
		...allSemantics().filter((s) => s.makeWeights),
		...allImplementations().filter((i) => i.makeWeights)
	];
	for (const spec of owners) {
		const rnd = mulberry32(streamSeed(seed, spec.id));
		slots[spec.id] = Array.from({ length: CONFIG.num_layers }, (_, l) =>
			spec.makeWeights!(rnd, CONFIG, l)
		);
	}

	return {
		embed: randMat(CONFIG.vocab_size, CONFIG.d_model, mulberry32(streamSeed(seed, 'embed'))),
		lmHead: randMat(CONFIG.d_model, CONFIG.vocab_size, mulberry32(streamSeed(seed, 'lm-head'))),
		slots
	};
}
