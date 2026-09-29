/**
 * 流程注册：引入即注册（与 `slots/index.ts` 同一套路）。
 *
 * 新增一条流程（diffusion / 多模态 / …）：
 *   1. 在本目录新建 xxx.ts，写一个 `buildXxxFlow(opts): BuiltFlow` 并 `registerFlow()`
 *   2. 在下面加一行 import
 * 流水线节点与步骤序列都从流程读。
 */
import './llm';
