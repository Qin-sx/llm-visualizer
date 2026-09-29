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
 *
 * 只初始化**这个模型、这一层真正用得到的插件**：V4 的注意力逐层不同
 * （SWA / CSA / HCA），给纯滑窗层初始化一份 CSA 权重既没意义、也算不出来——
 * CSA 的 `ape` 形状取决于该层的压缩比。
 */
import { mulberry32, randMat, type Mat } from '$lib/core/mat';
import { allImplementations, allSemantics } from '$lib/core/registry';
import { attnIdsFor, ffnIdsOf, pluginsOf, R1_MODEL, type ModelSpec } from './config';

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

export function initWeights(spec: ModelSpec = R1_MODEL, seed?: number): Weights {
	const cfg = spec.cfg;
	const s = seed ?? cfg.seed;
	const used = new Set(pluginsOf(spec));
	const ffnIds = ffnIdsOf(spec);
	const slots: Record<string, unknown[]> = {};
	// 语义插件与**自带权重的实现插件**各分一条流（实现一般不需要自己的权重，
	// 如 MLA 的吸收合并用的就是语义插件那批权重现场合并出来的矩阵）
	const owners = [
		...allSemantics().filter((x) => used.has(x.id) && x.makeWeights),
		...allImplementations().filter((x) => used.has(x.id) && x.makeWeights)
	];
	for (const owner of owners) {
		// 注意：一条流跨层往下走（`rnd` 只建一次），顺序是"插件在外、层在内"。
		// 换顺序会让所有已有插件的权重挪位，页面上的数字全要重测。
		const rnd = mulberry32(streamSeed(s, owner.id));
		const pack: unknown[] = [];
		for (let l = 0; l < cfg.num_layers; l++) {
			const here = ffnIds.includes(owner.id) ? [owner.id] : attnIdsFor(spec, l);
			if (here.includes(owner.id)) pack[l] = owner.makeWeights!(rnd, cfg, l);
		}
		slots[owner.id] = pack;
	}

	return {
		embed: randMat(cfg.vocab_size, cfg.d_model, mulberry32(streamSeed(s, 'embed'))),
		lmHead: randMat(cfg.d_model, cfg.vocab_size, mulberry32(streamSeed(s, 'lm-head'))),
		slots
	};
}
