/**
 * attention 插槽的插件注册。
 *
 * 新增一个注意力变体（GQA / DSA / …）：
 *   1. 在本目录新建 xxx.ts，实现 SemanticsSpec 并 export
 *   2. 在下面 import 并 register
 * 新增一个**执行方式**（如 MLA 的矩阵吸收合并）：实现 ImplementationSpec
 * 并 `registerImplementation()` —— 与语义正交。
 */
import { register, registerImplementation } from '$lib/core/registry';
import { mha } from './mha';
import { mla } from './mla';
import { mlaAbsorb } from './absorb';
import { swa } from './swa';
import { csa } from './csa';
import { hca } from './hca';

register(mha);
register(mla);
register(swa);
register(csa);
register(hca);
registerImplementation(mlaAbsorb);

export { mha, mla, mlaAbsorb, swa, csa, hca };
