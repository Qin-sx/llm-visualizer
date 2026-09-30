/**
 * 「真实尺寸」标注——本项目的第一条表示策略（见 `slots/attention/mha.ts` 的头注释）：
 * **每个矩阵都要标出真实模型里的尺寸与缩小倍数**。
 *
 * 所以这份判定只有一处：所有带 `MatRef` 的视图（`matmul` / `transform` / `concat` /
 * `sum` / `lookup`）都调它，新增视图时也别再自己写一遍——两边算法一漂移，
 * 就会出现"这一页标了真实尺寸、那一页没标"。
 *
 * 注意 `mask` / `matrix` / `row` / `bars` / `tiles` 这几种视图**不带 `MatRef`**：
 * 它们画的是不物化的中间量（如融合注意力的分数矩阵），真实尺寸取决于序列长度，
 * 不是一个固定维度——那种情况写在 `label` 里，而不是硬塞一个 `realShape`。
 */
import type { MatRef } from '$lib/core/types';

/** 缩放倍数：≥10 取整，否则一位小数 */
function ratio(f: number): string {
	return f >= 10 ? String(Math.round(f)) : f.toFixed(1);
}

/**
 * `真实 [7168 × 7168] · 行缩小 896× · 列缩小 896×`；没有 `realShape` 时返回 `null`。
 *
 * `ref.realParts` 有值时在后面补一段**分段尺寸**：`[v | s]` 这种"一块矩阵里其实是两样东西"的，
 * 光看合计看不出"值占多少列、打分占多少列"。分段尺寸由声明者显式给出，不在这里反推（理由见 `MatRef.realParts`）。
 */
export function realLabel(ref: MatRef): string | null {
	const rs = ref.realShape;
	if (!rs || rs.length < 2) return null;
	const parts: string[] = [];
	if (rs[0] / ref.shape[0] >= 1.5) parts.push(`行缩小 ${ratio(rs[0] / ref.shape[0])}×`);
	if (rs[1] / ref.shape[1] >= 1.5) parts.push(`列缩小 ${ratio(rs[1] / ref.shape[1])}×`);
	const seg = ref.realParts?.length
		? `（${ref.realParts.map((p) => `${p.label} ${p.cols} 列`).join(' + ')}）`
		: '';
	return `真实 [${rs.join(' × ')}] · ${parts.length ? parts.join(' · ') : '与真实同量级'}${seg}`;
}
