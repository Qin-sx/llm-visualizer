/**
 * 矩阵格子的配色，供 MatrixGrid / MaskView 共用。
 *
 * 正值 → 靛蓝，负值 → 玫红，白色为零；按最大绝对值归一化。
 */
import type { Mat } from '$lib/core/types';

export function maxAbsOf(data: (number | null)[][] | Mat): number {
	let m = 0;
	for (const row of data)
		for (const v of row) {
			if (v === null || !Number.isFinite(v)) continue;
			const a = Math.abs(v);
			if (a > m) m = a;
		}
	return m || 1;
}

export function cellColor(v: number | null, maxAbs: number): string {
	if (v === null) return '#ffffff'; // 尚未算出
	if (!Number.isFinite(v)) return '#f1f5f9'; // −∞
	const t = Math.max(-1, Math.min(1, v / maxAbs));
	const target = t >= 0 ? [79, 70, 229] : [225, 29, 72];
	const a = Math.abs(t) * 0.88;
	const r = Math.round(255 + (target[0] - 255) * a);
	const g = Math.round(255 + (target[1] - 255) * a);
	const b = Math.round(255 + (target[2] - 255) * a);
	return `rgb(${r},${g},${b})`;
}

export function cellTextColor(v: number | null, maxAbs: number): string {
	if (v === null || !Number.isFinite(v)) return '#cbd5e1';
	return Math.abs(v / maxAbs) > 0.55 ? '#ffffff' : '#334155';
}

export function fmtCell(v: number | null): string {
	if (v === null) return '';
	if (!Number.isFinite(v)) return '−∞';
	return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2);
}
