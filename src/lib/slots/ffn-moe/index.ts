/**
 * ffn-moe 插槽的插件注册。
 *
 * 新增一个 MoE 变体（如 fine-grained MoE、其他路由策略）：
 *   在本目录新建 xxx.ts，实现 SemanticsSpec 并在这里 register 即可。
 */
import { register } from '$lib/core/registry';
import { denseFfn } from './dense-ffn';
import { deepseekMoe } from './deepseek-moe';

register(denseFfn);
register(deepseekMoe);

export { denseFfn, deepseekMoe };
