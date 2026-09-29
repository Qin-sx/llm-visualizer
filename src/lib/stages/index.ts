/**
 * 阶段（Stage）注册与汇总。
 *
 * 阶段与算子插件的区别（见 `core/flow.ts`）：
 *   - 算子插件（`slots/`）：**每层各做一次**，由 `SemanticsSpec` 描述
 *   - 阶段（`stages/`）：流程里的一个环节，可以是模型级的（Embedding / LM Head）
 *
 * 已实现：embedding-lookup、lm-head（next-token）
 * 后续 diffusion / 多模态的模型级阶段也放这里（如 `vae-decode.ts`、`patch-embed.ts`）。
 */
export { embeddingLookup, type EmbeddingData } from './embedding';
export { nextTokenHead, type HeadData } from './lm-head';
