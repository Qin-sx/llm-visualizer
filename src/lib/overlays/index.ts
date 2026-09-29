/**
 * Overlay（并行 / 量化）——**尚未实现**，只留接口。
 *
 * 设计意图：
 * Overlay 与所有插槽**正交**，且**可以同时开多个**——它不改数据，
 * 只往已有步骤上追加视觉标注（切分线、通信箭头、量化 block 网格）。
 *
 * 新增方式：在本目录下加 tp.ts / ep.ts / pp.ts / dp.ts / fp8.ts，
 * 每个文件实现 OverlaySpec 并调用 `registerOverlay()`。
 */
export {};
