/**
 * 核心类型定义。
 *
 * 设计原则：本文件（以及整个 core/）**不允许出现任何具体算子名**。
 * 一旦需要写 `if (semantics === 'mha')`，说明抽象漏了。
 */

// ── 基础张量类型 ──────────────────────────────────────────
export type Vec = number[];
/** 行优先的二维矩阵 */
export type Mat = number[][];

// ── 插槽与表示深度 ────────────────────────────────────────
export type SlotId = 'embedding' | 'attention' | 'kvcache' | 'ffn-moe' | 'residual' | 'lm-head';

/**
 * L0 叙述级：一个节点 + 公式 + 一段文字
 * L1 结构级：张量形状 + 数据流
 * L2 算子级：逐步骤、分块、逐专家
 */
export type Depth = 'L0' | 'L1' | 'L2';

// ── 步骤原语 ──────────────────────────────────────────────
export type PrimitiveKind =
	| 'NARRATE' // 纯叙述，不做计算
	| 'PROJECT' // 线性投影
	| 'MATMUL' // 矩阵乘
	| 'ACT' // 逐元素激活（SiLU 等）
	| 'SOFTMAX'
	| 'MASK' // 掩码（因果 / padding）
	| 'RESCALE' // 按指数差修正（FlashAttention 用）
	| 'ROPE' // 旋转位置编码
	| 'ROUTE' // MoE 路由打分
	| 'SELECT' // top-k 选择
	| 'CONCAT'
	| 'ADD'
	| 'LOAD'
	| 'STORE';

// ── 张量视图：渲染层认识这几种就够 ────────────────────────
export interface TensorViewBase {
	name: string;
	label?: string;
	shape: number[];
	/**
	 * 格子边长提示（px）。不写就按视图类型兜底（矩阵乘 / 变换各有各的公式）。
	 *
	 * 用于把某一步的矩阵放大一些——比 `fillWidth`（占满整行）温和，是个定值。
	 * **同一行的视图必须给同一个值**，否则并排的矩阵大小不一致、顶边对不齐
	 * （如 `-compress-pool` 的 `w⊙v` / `v̄` / 归一化链三块要一起给）。
	 */
	cellSize?: number;
	/** 布局：同一 `row` 的视图**横向并排**（如 Q·Kᵀ → mask → P·V 流水线）；
	 * 不填则独占一行。
	 */
	row?: number;
	/**
	 * 布局：同一 `row` 内、同一 `group` 的视图**纵向堆叠**成一列
	 * （如 softmax 的输入行 / 输出行 / 权重矩阵）。
	 */
	group?: string;
	/**
	 * 布局：同一 `row` 的多个视图是**并行分支**（如一个 token 同时过 top-k 个专家），
	 * 而非先后流水线。区别有二：
	 *   - 不画中间的 `→` 箭头；
	 *   - 页级播放时它们**共用一个动画段**（一起播），而不是按等份依次播。
	 */
	parallel?: boolean;
	/**
	 * 布局：这一行并排的各列改成**垂直居中**对齐（默认按"每列最后一块矩阵的顶边"对齐）。
	 *
	 * 用于"左右两块要夹在中间"的场合（如 `-sape` 页：左边 s_overlap / s 两条链上下分布，
	 * 右边的窗口槽位打分要**垂直居中**在它们之间）。标在任一视图上对该行生效。
	 */
	vcenter?: boolean;
	/**
	 * 并行行（`parallel`）里默认在列间画**浅色虚线**（表示"各算各的、完全无关"，如 MoE 的两个专家）。
	 * 标了 `noHsep` 的行改成画 `→` 箭头——用于"并行但**相关**"的计算（如 `-sape` 页的
	 * s_overlap / s 同时加 ape，但都汇向同一个窗口槽位打分，不是无关分支）。
	 */
	noHsep?: boolean;
	/**
	 * **扫描拼接**：这个视图在 reveal 时，把"正在填出来的那一段行"的来源行块高亮到别的视图上。
	 *
	 * 用于 `-sape` 页：窗口槽位打分逐行 reveal（拼接），每一行的值来自 s_overlap+ape 或 s+ape 的
	 * 某一段行（按 `ratio` 行一组拷贝）。`seq[r]` 是窗口槽位打分第 r 行所属来源**行块**
	 * （`target` = 源视图的 name、`from`/`to` = 源矩阵要高亮的行块闭区间；`null` = 无来源，如空槽）。
	 * `StepPanel` 按这个表把 `scanRows` 传给目标视图，于是源矩阵上出现一个随窗口 reveal
	 * **移动的、和拷贝行数一致高的扫描框**（一次拷 `ratio` 行 → 三个矩阵的框都是 `ratio` 行高）。
	 */
	scan?: { seq: ({ target: string; from: number; to: number } | null)[] };
	/**
	 * 布局：同一 `row` 的多个视图**共用一个动画段，而且进度完全同步**（照样画 `→` 箭头）。
	 *
	 * 和 `parallel` 的区别：`parallel` 是"并行分支同时开算"，两边工作量不同时按工作量
	 * **反比**缩放进度（小的先算完、然后空转等大的）；`sync` 是"两边的揭示必须**一一对上**"——
	 * 如组内加权平均那一步：`w ⊙ v` 的**第 c 列**算完，右边的 `v̄` 才出现第 c 个数。
	 */
	sync?: boolean;
	/**
	 * 占满整行（跨过 `step.tensors` 那个自动填充网格的所有列）。
	 *
	 * 不 wide 的视图只拿到一格（约 220px）：一句长注解挤在那一格里会折成好几行，
	 * 旁边要是还有别的视图，nowrap 的文本会横穿过去。注解长的视图（`shape` 的 `note`、`row` 的 `label`）把它打开。
	 */
	wide?: boolean;
	/**
	 * 这一行**放大到占满可用宽度**（目前只对 `matmul` / `transform` 生效）。
	 *
	 * 打开之后 `StepPanel` 会量出这一行的可用宽度，反推出一个"内容刚好占满"的格子边长，
	 * 传给这一行的所有块；同时这一页的注解改成**占满各自矩阵的宽度**（见 `app.css`）。
	 *
	 * 刻意做成**逐视图开关**而不是全局规则：放大是"这一页正好有横向空间"的排版决定，
	 * 别的页（格子密、块多）放大会溢出或者把页面拉得很长。用户只要求在
	 * c128 的 gate 那一页这么做（"我只需要把 c128 的第 6 步的矩阵放大一些，占满页面宽度"）。
	 */
	fillWidth?: boolean;
	/**
	 * 在顶部"流程总览"里作为一个节点出现；值为显示名（如 `'v̄'`）。
	 *
	 * 矩阵乘 / 变换 / 拼接 / 求和 / 查表自带具名操作数，总览从 `a`/`b`/`out` 就能推出节点，
	 * **不用填**。只有 `matrix` / `row` / `bars` / `ewise` 这几类"没有具名操作数"的视图需要它——
	 * 总览把一步里**连续几个**声明了 `overview` 的视图按先后串成 `input → output`
	 * （如组内加权平均的 `w ⊙ v → v̄`）。
	 *
	 * 一步里一个都不声明，这一步在总览里就会**整段消失**（不报错，只是不画）——
	 * 所以新增这类视图时记得顺手补上（`verify` 有断言）。
	 */
	overview?: string;
}

/**
 * 只显示形状，不显示数据。
 *
 * 大张量（如 [8×128] 的 hidden state）铺成热力图既占地方又读不出信息，
 * 直接标尺寸更清楚。**能用形状表达的就不要画图。**
 */
export interface ShapeView extends TensorViewBase {
	kind: 'shape';
	note?: string;
	/** 强调显示（如最终输出） */
	accent?: boolean;
}

/** 一行数值，每个元素带数字——用于展示向量 / 单行 */
export interface RowView extends TensorViewBase {
	kind: 'row';
	data: Vec;
	labels?: string[];
	/** 颜色归一化上限；不传则用该行的最大绝对值 */
	scaleMax?: number;
	/**
	 * 是否按进度**逐个**揭示（第 i 个数在第 i 步算完时才出现）。
	 *
	 * 用在"这一行是由左边那块逐列算出来的"场合——如组内加权平均：`w ⊙ v` 的第 c 列
	 * 算完，`v̄` 的第 c 个数才出现（配合 `sync` + `revealOrder: 'col'`）。
	 */
	animated?: boolean;
	/**
	 * 每多少个"揭示单位"才出一个数——就是**左边那块矩阵的行数**。
	 * 见 `ValueRow` 的 `revealGroup`。
	 */
	revealGroup?: number;
}

/** 小矩阵热力图（仅用于 ≤16×16 这类真正需要看整体形态的矩阵，如注意力矩阵） */
export interface MatrixView extends TensorViewBase {
	kind: 'matrix';
	data: Mat;
	/**
	 * 矩阵**显示名**（画在矩阵上方，和 `transform` / `matmul` 的表头同款）。
	 *
	 * `matrix` 视图的 `name` 是内部 id（如 `hca-pool-result`），不是给人看的名字；
	 * 单独一个字段才能显示成 `v̄` 这种。放在同行、旁边都是有表头的矩阵时用
	 * （如压缩器的结果矩阵 `v̄`）。
	 */
	title?: string;
	/** 高亮的单元格 */
	highlight?: [number, number];
	/** 是否逐格揭示（用于 softmax 这类逐元素计算） */
	animated?: boolean;
	/**
	 * 逐格揭示的**顺序**：默认 `'row'`（行优先，逐行扫过去）；
	 * `'col'` 是**列优先**——一整列算完再走下一列。
	 *
	 * 用在"结果要按列归约"的场合：组内加权平均的 `w ⊙ v` 用 `'col'`，
	 * 每列算完右边的 `v̄` 就出现对应的那个数（正在累加的那一列会整列加淡底）。
	 */
	revealOrder?: 'row' | 'col';
	/**
	 * 逐列揭示时**每多少个"揭示单位"算完一整列**（= 上游那个矩阵的行数）。
	 *
	 * 给"按列归约的结果"用：结果矩阵要**整列一起出现**（这一列加完了才轮到下一列），
	 * 才能和上游"列算完"的时刻严格对上。换算式与 `ValueRow.revealGroup` 同一套。
	 */
	revealGroup?: number;
	/**
	 * **逐组揭示**：把行按 `groupRows` 行一组切开，一组一组地算（组内仍按列走）。
	 *
	 * 用在"整段序列分组压成若干条"的场合：8 个 token、一组 4 个 → **先算前 4 个得到第 1 条、
	 * 再算后 4 个得到第 2 条**。
	 *
	 * 写在**结果**上时，换算式是"上游每算完一组的一列，我就出一格"
	 * （`⌊round(reveal × 行 × 列 × groupRows) / groupRows⌋`，见 `MatrixGrid`）——
	 * 这里 `groupRows` 与上游那个视图写的是**同一个数字**（"一组几行"）。
	 */
	groupRows?: number;
	/**
	 * 框出这一**段行** `[起始, 结束]`（闭区间）。
	 *
	 * 用在"整块矩阵都画出来，但只有其中几行参与这一步"的场合——
	 * 如压缩器里"8 个 token 全画出来，框出的这 4 行才是这一条压缩条目看的窗口"。
	 */
	highlightRows?: [number, number];
	/** 框出这几列（和 `highlightRows` 配合：如 `[v | s]` 里"打分那几列"） */
	highlightCols?: number[];
}

export interface TilesView extends TensorViewBase {
	kind: 'tiles';
	data: Mat[];
	tileLabels?: string[];
}

export interface BarsView extends TensorViewBase {
	kind: 'bars';
	data: Vec;
	labels?: string[];
	/**
	 * 是否按进度逐个揭示。
	 *
	 * 分布类的结果（如 softmax 之后的概率）是"算完才知道"的，一上来就画满
	 * 会剧透前面那步矩阵乘的过程，所以支持跟着进度长出来。
	 */
	animated?: boolean;
}

/** 矩阵乘法中的一个操作数：可只带形状（大矩阵），也可带数据（小矩阵） */
export interface MatRef {	name: string;
	label?: string;
	shape: number[];
	/** 有数据才能展示具体数值 */
	data?: Mat;
	/**
	 * 真实模型中的形状（如 DeepSeek R1 的 [7168 × 7168]）。
	 * 用于在矩阵旁标注"真实 size 是多少、此处缩小了几倍"。
	 */
	realShape?: number[];
	/**
	 * 跨步骤衔接用的稳定标识。
	 *
	 * 同一个 id 出现在不同步骤里时，切换步骤会有一个"矩阵飞过去"的动画，
	 * 让读者看到"上一步算出的 Q 就是这一步用的 Q"。见 core/handoff.ts。
	 */
	handoffId?: string;
	/**
	 * 这个张量**是什么角色**——顶部流程总览据此上色（不填则按"操作数灰 / 产物靛蓝"）。
	 *
	 * 放在 `MatRef` 上而不是某个视图里：这是张量自己的属性，
	 * 任何视图（总览、数据流图…）都该用同一套配色。
	 */
	tone?: 'input' | 'latent' | 'rope' | 'head' | 'result' | 'weight' | 'cache';
	/**
	 * 这个张量是**几段拼起来的**（如 MLA 每个头的 `Q_h = [nope 段 | rope 段]`）。
	 *
	 * 总览里会把方框画成一条分段条，一眼看出两段的来源与比例。
	 */
	split?: { label: string; ratio: number; tone?: MatRef['tone'] }[];
	/**
	 * "真实尺寸"那行要不要**分段标注**——如 `[v | s]` 里"值占多少列、打分占多少列"。
	 *
	 * 为什么单独一个字段、不复用 `split`：`split` 的 label 带的是**缩放后**的列数（`'nope 3'`），
	 * 而且它的比例**不一定等于真实比例**（MLA 的 nope/rope 缩放比是 3:2，真实是 448:64），
	 * 从 `split` 反推真实列数会算出 307.2 这种假数字。所以要标注真实分段尺寸时显式给出。
	 */
	realParts?: { label: string; cols: number }[];
	/**
	 * **分组框**：把某一段行或列用浅色虚线框圈起来，用来区分"同一个矩阵里的两类东西"。
	 *
	 * 最典型的用法是 V4 混合注意力的 K / V：前 `S` 列（行）是**滑窗里的精确 token**、
	 * 接着 `nE` 列（行）是**压缩条目**——同一块矩阵、两种来源，用不同浅色区分。
	 * 颜色与文字的对应写在块的注解里（和 `highlightCols` 那套"框 + 注解"的用法一致）。
	 */
	bands?: MatrixBand[];
}

/** 矩阵里的一段分组框（见 `MatRef.bands`）：`from` / `to` 是**闭区间**的索引 */
export interface MatrixBand {
	/** 沿哪个方向分段 */
	axis: 'row' | 'col';
	from: number;
	to: number;
	/** 框的颜色（浅色；同一类东西在不同矩阵里用同一个颜色） */
	tone: string;
}

/**
 * 矩阵乘**过程**视图——这是本项目最核心的表示。
 *
 * 不只给结果，而是取一个输出元素，把
 * `C[i][j] = Σ_p A[i][p]·B[p][j]`
 * 的每一步（取行、取列、逐项相乘、求和）用真实数字摊开。
 */
export interface MatmulView extends TensorViewBase {
	kind: 'matmul';
	a: MatRef;
	b: MatRef;
	out: MatRef;
	/** 默认聚焦的输出元素 [i, j] */
	focus?: [number, number];
	/**
	 * 框出**结果块**的这几列。
	 *
	 * 用在"结果是一块拼起来的东西、下一步只用其中几列"的场合——
	 * 如 `[v | s]` 里"打分那几列"（下一步要拿它加 ape 做 softmax）。
	 */
	highlightCols?: number[];
	/** 额外标量（如 1/√d_h），会显示在式子里 */
	scaleNote?: string;
}

/**
 * 逐元素二元运算视图：`out[i][j] = a[i][j] ⊙ b[i][j]`（两个操作数**同形状**）。
 *
 * 排布与 `MatmulView` 完全一样（`b` 右上、`a` 左下、`out` 右下），读法也照旧
 * （横向 `a` → `out`、纵向 `b` → `out`）；区别只有一个：**没有 Σ**——
 * 每个输出格只由两个**同位置**的输入格决定，所以高亮的是同一个格子，
 * 算式框里也只有一项（不是 `Σ_p`）。
 *
 * 用在组内加权平均那一步：`w ⊙ v`——两个操作数都画出来
 * （`v` 取自 `[v | s]` 的值那几列、`w` 就是上一步 softmax 的结果），
 * 而不是只画一个"已经乘过权重"的结果矩阵。
 */
export interface EwiseView extends TensorViewBase {
	kind: 'ewise';
	a: MatRef;
	b: MatRef;
	out: MatRef;
	/** 运算符符号，默认 `⊙` */
	op?: string;
	/** 框出**结果块**的这几行（闭区间）——如"这条压缩条目看的窗口那 4 行" */
	highlightRows?: [number, number];
	/** 框出**结果块**的这几列 */
	highlightCols?: number[];
	/**
	 * 逐格揭示的顺序：默认行优先；`'col'` 是**列优先**（一整列算完再走下一列）。
	 * 用在"结果要按列归约"的场合——列算完，右边的 `v̄` 才出对应的那个数。
	 */
	revealOrder?: 'row' | 'col';
	/**
	 * **逐组揭示**：行按 `groupRows` 行一组切开，一组一组地算（组内仍按列）。
	 *
	 * "整段序列压成若干条"时用：8 个 token、一组 4 个 → 先算前 4 个（第 1 条）、
	 * 再算后 4 个（第 2 条）。框会跟着动画走：**框出的就是正在算的那一组**
	 * （所以不用再手写 `highlightRows`）。
	 */
	groupRows?: number;
}

/**
 * 掩码的一个"来源段"：掩码动画按段**依次**处理（每段逐行处理自己的列区间），
 * 每段有自己的来源注解（如"滑窗规则"、"indexer 的 top-k"、"sink 恒保留"）。
 *
 * 用于"一张矩阵的掩码来自几个不同规则"的场合（如 V4 融合注意力的
 * `S（+ sink 列）`：token 列 ← 滑窗、压缩列 ← indexer 的 top-k、sink 列 ← 恒保留）。
 */
export interface MaskPart {
	/** 这一段的列区间（闭区间，含 `to`） */
	from: number;
	to: number;
	/** 来源注解：这一段掩码是谁决定的 */
	label: string;
}

/**
 * 掩码过程视图：动画演示 j > i 的位置被逐个置为 ∅（数学上是 −∞）。
 *
 * 未处理 → 原值淡显；处理且保留 → 原值；处理且屏蔽 → ∅（≈ −∞）。
 */
export interface MaskView extends TensorViewBase {
	kind: 'mask';
	/**
	 * 这张矩阵显示的是谁（如 `S_h`）。会渲染成与 `transform` / `sum` 视图
	 * 同款的表头（名字 + 尺寸），这样并排时矩阵顶边能对齐。
	 * 不填则退回视图自身的 `name`。
	 */
	matrixName?: string;
	/** 原始分数（全为有限值） */
	scores: Mat;
	/** 掩码：1 = 保留，0 = 屏蔽 */
	mask: Mat;
	/**
	 * 可选：掩码按**来源段**依次处理（每段逐行处理自己的列区间、各有注解），
	 * 而不是整块一起逐行。缺省 = 整块一次逐行（现状）。
	 */
	parts?: MaskPart[];
}

/**
 * 拼接过程视图：把多个块（如各头的输出）并排摆开，再逐列揭示拼接结果。
 *
 * 动画由 progress 驱动：进度走过某个块的列区间时，那部分列出现在结果里，
 * 同时该块高亮——于是能看清"结果的哪几列来自哪个头"。
 */
export interface ConcatView extends TensorViewBase {
	kind: 'concat';
	/** 待拼接的各块，按拼接顺序排列 */
	parts: MatRef[];
	/** 拼接结果 */
	result: MatRef;
	/**
	 * 可选：拼接结果再作为**左操作数**做一次矩阵乘（如 `Concat · W_O → O`）。
	 * `result` 只画一次，同时充当这次矩阵乘的 A——否则 `Concat` 会在页面上出现两次。
	 */
	then?: {
		op: string;
		/** 矩阵乘的右操作数 */
		b: MatRef;
		/** 矩阵乘的结果 */
		result: MatRef;
	};
}

/**
 * 变换过程视图：把"由上一个矩阵算出当前矩阵"做成动画，可串成一条链。
 *
 *   1) 只有一段：                [输入] ──op──▶ [输出]
 *   2) 两段逐行（`then`）：       [输入] ──op1──▶ [中间] ──op2──▶ [结果]
 *   3) 第二段是矩阵乘（`then.b`）：
 *                                          ┌──────────┐
 *                                          │ [B]      │   ← B 画在右上
 *      [输入] ──op1──▶ [输出 = A] ─────────┤ [A]  [C] │   ← 输出同时是这次矩阵乘的左操作数
 *                                          └──────────┘
 *   4) 再加一段逐行变换（`tail`）：在第 3 种的 C 右边再接 `op3` 与 `[D]`——
 *      `[输入] ──op1──▶ [A] ──op2(B)──▶ [C] ──op3──▶ [D]`，C 因此只画一次。
 *
 * 逐行处理——左侧输入矩阵的当前行高亮，右侧输出矩阵逐行揭示，
 * 下方给出当前行的前后数值对比。用于 softmax / top-k / SiLU 这类逐行独立计算的算子。
 */
export interface TransformView extends TensorViewBase {
	kind: 'transform';
	/** 输入矩阵（上一步的产物） */
	input: MatRef;
	/**
	 * 输入块**不画**——它的内容由**同行左边**的另一块矩阵显示（如压缩器的 `-compress-pool`
	 * 把 `-compress-kv` 并进来时，`v̄` 由 pool 的结果矩阵画，RMSNorm 的输入块就不重复画）。
	 * 变换的节点 / 时长等逻辑照常使用 `input`（只是不渲染那一块）。
	 */
	hideInput?: boolean;
	/** 输出矩阵 */
	output: MatRef;
	/** 运算名，如 'softmax' */
	op: string;
	/**
	 * 在**输入与输出**矩阵上框出这几列（整列加底色 + 外框）。
	 *
	 * 用在"只有末尾几维被这个算子碰到"的场合——如输出做逆 RoPE 时，
	 * 把 `O_h` 与 `O_h′` 里那几维都框出来，读者才知道转的是哪一段。
	 * 两边用同一批列号：这一段的**位置**前后不变，变的只是值。
	 */
	highlightCols?: number[];
	/**
	 * 可选：输入块先做**逐格掩码**（如因果掩码 j > i → ∅），再做逐行变换。
	 * 有了它，掩码和后面的变换共用同一个矩阵块——`S_h` 只画一次。
	 */
	preMask?: Mat;
	/**
	 * 可选：`preMask` 按**来源段**依次掩码（每段逐行处理自己的列区间、各有来源注解），
	 * 而不是整块一起逐行。如 V4 融合注意力的 `S（+ sink 列）`：
	 * token 列 ← 滑窗规则、压缩列 ← indexer 的 top-k、sink 列 ← 恒保留。
	 * 缺省 = 整块一次逐行掩码（现状）。没有 `preMask` 时忽略。
	 */
	preMaskParts?: MaskPart[];
	/**
	 * 可选：在**输入块（矩阵）上方**叠一个"来源参考条"——一个静态的小掩码矩阵
	 * （如 `[S × nE]` 的 top-k 选择矩阵），列左边界对齐输入矩阵的 `colFrom` 列。
	 *
	 * 用于"这张矩阵的某一列段掩码是从哪个中间产物来的"（如 V4 融合注意力的压缩列
	 * ← indexer 的 top-k 选择）。只显示不复播（来源动画在上一步已演过），`view`
	 * 以完整进度渲染；不参与动画计时、也不参与并排对齐测量。
	 */
	topRef?: {
		/** 参考条视图（`MaskView`：scores + mask + label），progress 固定为 1 */
		view: MaskView;
		/** 参考条左边界对齐输入矩阵的哪一列（输入矩阵列坐标） */
		colFrom: number;
	};
	/**
	 * 可选第二段，接着 `output` 往下做：
	 *   - 不给 `b`：再做一次**逐行变换**（如 softmax → top-k），`output` 只画一次；
	 *   - 给了 `b`：这一段是**矩阵乘** `output · b → result`，`b` 画在 `output` 上方，
	 *     `output` 同时充当左操作数（所以 `P_h` 只画一次）。
	 */
	then?: {
		op: string;
		result: MatRef;
		/** 矩阵乘的右操作数（给了它，这一段就是矩阵乘） */
		b?: MatRef;
	};
	/**
	 * 可选第三段，接着 `then.result` 再往下做**一次逐行变换**（如输出做逆 RoPE）。
	 *
	 * 加它就是为了"同一个矩阵只画一次、产物紧挨着产物"：
	 *
	 *   `S ──softmax──▶ P_h ──×V──▶ O_h ──逆 RoPE──▶ O_h′`
	 *
	 * 一条链走完，`O_h′` 就贴在 `O_h` 右边。不这么写的话，`O_h` 得在下一行再当一次输入
	 * 画出来——同一个矩阵在一页里出现两次、还挂着两个名字。
	 *
	 * 只支持**逐行变换**，不支持矩阵乘：链尾再挂一个矩阵乘会多出两块，整行放不下。
	 */
	tail?: {
		op: string;
		result: MatRef;
		/** 在 `then.result` 与 `tail.result` 上**同时**框出这几列（位置不变，只有值变） */
		highlightCols?: number[];
	};
}

/**
 * 加权求和 / 逐元素相加视图。
 *
 *   [项1 × w1] ┐
 *   [项2 × w2] ┼──Σ──▶ [结果] ──┐
 *              ┘        [追加项] ┼──+──▶ [最终结果]
 *                               ┘
 *
 * 加法**不是矩阵乘**，所以单列一种视图。
 * 用于 MoE 的"各专家输出按门控权重求和，再叠加共享专家"。
 */
export interface SumView extends TensorViewBase {
	kind: 'sum';
	/** 第一段被相加的各项（同形） */
	terms: MatRef[];
	/** 逐项逐行的系数：`weights[j][t]` 是第 j 项在第 t 行的权重；缺省视为全 1（纯相加） */
	weights?: number[][];
	/**
	 * 每项右侧**竖挂一列**（与 `terms` 一一对应）——如 mHC 的 pre 门控列：
	 * `r_k | pre[:,k]`（读门矩阵放它作用的流旁边，逐行揭示和该项同步）。
	 * 列宽通常 `[S, 1]`；不参与求和，只是把"这一项乘的系数"画出来。
	 */
	termSide?: MatRef[];
	/** 第一段的结果（如 `routed`） */
	result: MatRef;
	/**
	 * 可选第二段：在 `result` 之上**继续加**这些项（如叠加共享专家）。
	 *
	 * 有了它，`result` 只画一次——第二段接着它的结果往下加，而不是把同一个矩阵
	 * 再画一遍当输入（那样 `routed` 会在页面上出现两次）。
	 */
	then?: {
		terms: MatRef[];
		result: MatRef;
	};
}

/**
 * 查表过程视图：按 key 从表里把对应的行**取出来**。
 *
 * embedding 的本质是查表，不是矩阵乘，所以单列一种视图：
 *
 *   [W_E]  ──查表──▶  [x_0]
 *   整张表画出来，      取出来的行摞成结果矩阵，
 *   正在取的那行高亮     逐行揭示
 *
 * 左边能看到"表一共多大、只用了其中哪几行"，右边能看到"取出来的到底是什么"。
 */
export interface LookupView extends TensorViewBase {
	kind: 'lookup';
	/** 被查的表，整张画出来 */
	table: MatRef;
	/** 查表用的 key（如 token id），与 `rows` 一一对应 */
	keys: number[];
	/** 取出来的行（每行一个向量） */
	rows: Vec[];
	/** 查表结果：`rows` 摞成的矩阵 */
	result: MatRef;
	/** key 的名称，如 `token id` */
	keyName?: string;
}

/**
 * KV Cache 视图：把"缓存里到底躺着什么"摊开，并与另一种方案比体积。
 *
 * 这是 MLA 存在的**全部理由**，而它既不是矩阵乘也不是变换——是"缓存里有哪些数"
 * 的静态结构 + 一个体积对比。所以单列一种视图（同 `sum` / `lookup` 的取舍）。
 *
 * 画法：**每个 token 一行**（`blocks[].data` 是 `[token 数 × 每 token 几个数]`），
 * 逐行揭示——这样能看出"缓存是随着 token 一个一个填起来的"，而不只是看到第一行。
 */
export interface KvCacheView extends TensorViewBase {
	kind: 'kvcache';
	/** 左侧（本方案）的标题；不填按 MLA 的说法 */
	mineTitle?: string;
	/** 左侧体积条上的短标签（如 `MLA` / `V4`）；不填按 MLA */
	mineLabel?: string;
	/** 本方案缓存的各块，按缓存里的顺序排开 */
	blocks: {
		name: string;
		label?: string;
		/** 每个 token 几个数 */
		size: number;
		/** 真实模型里每个 token 几个数 */
		realSize: number;
		/** 整段缓存：`[token 数 × size]`，一行一个 token */
		data?: Mat;
	}[];
	/** 对比方案（如 MHA 缓存的 K / V），只比体积，不画真实数值 */
	compare: {
		label: string;
		parts: {
			name: string;
			/** 每个 token 几个数 */
			perToken: number;
			/** 小字说明，如 `2 头 × 4 维` */
			sub?: string;
		}[];
		/** 真实模型里每 token 几个数 */
		realPerToken: number;
	};
	/** 本方案每 token 元素数（= blocks 的 size 之和） */
	perToken: number;
	realPerToken: number;
	/** 一句话结论，如 absorb 技巧 */
	note?: string;
}

export type TensorView =
	| ShapeView
	| RowView
	| MatrixView
	| TilesView
	| BarsView
	| MatmulView
	| EwiseView
	| MaskView
	| ConcatView
	| TransformView
	| SumView
	| LookupView
	| KvCacheView;

// ── 声明式动画 ────────────────────────────────────────────
/** 语义引用——由渲染层解析成真实 DOM 节点，绝不用 DOM 选择器 */
export type ElementRef =
	| { tensor: string; index?: number[] }
	| { node: string }
	| { group: string };

export interface AnimDirective {
	target: ElementRef;
	effect: 'fade' | 'pulse' | 'draw' | 'flow' | 'scale' | 'count';
	duration: number;
	stagger?: number;
	from?: Record<string, number | string>;
	to?: Record<string, number | string>;
}

// ── 步骤的参考数据流图 ────────────────────────────────────
/**
 * 数据流图上的一个节点。
 *
 * 位置由**声明者**给定（`col` / `row`）。
 */
export interface DataflowNode {
	id: string;
	/** 节点上的名字，如 `c_KV` */
	label: string;
	/** 第二行小字，如 `[6 × 2] 所有头共享` */
	sub?: string;
	col: number;
	row: number;
	/**
	 * 配色分类。
	 *
	 * `absorbed` / `gone` 是给"同一个算子的不同实现"做**差异标注**用的：
	 * 吸收合并版就是在朴素版那张图上，把折出来的环节标成 `absorbed`、
	 * 把不再需要的矩阵标成 `gone`——两张图布局完全一样，一眼能看出差在哪。
	 */
	tone?: 'input' | 'latent' | 'rope' | 'head' | 'result' | 'cache' | 'weight' | 'absorbed' | 'gone';
	/** 淡显（如"吸收合并后不再走这条路"），配合 `gone` 用 */
	faded?: boolean;
	/**
	 * 把一个节点画成**分段的条**，用来表达"这个张量是几段拼起来的"。
	 *
	 * MLA 的 Q/K 就是典型：每个头 = `[nope 段 | rope 段]`，位置编码只占尾部一小截。
	 * 只画成一个方块、把"拼起来"写在标注里，读者看不出两段的来源和比例——
	 * 所以节点本身要能显示分段（`ratio` 是各段占比）。
	 */
	split?: { label: string; ratio: number; tone?: DataflowNode['tone'] }[];
}

/** 数据流图上的一条边（正交走线：横 → 竖 → 横） */
export interface DataflowEdge {
	from: string;
	to: string;
	/** 边上的运算名，如 `W_DQ` / `RoPE` */
	label?: string;
	/** 虚线：不是前向计算，而是"decode 时从这里取缓存"这类 */
	dashed?: boolean;
	/** 边的颜色，用来标出"这一路是位置编码段"这类角色（默认灰） */
	tone?: DataflowNode['tone'];
	/** 淡显：如"吸收合并后不再走这条路" */
	faded?: boolean;
	/**
	 * 竖段离目标左边界的距离（px，默认 22）。
	 *
	 * 两条边从同一个节点出发、又拐向相邻的目标时，竖段会重叠成一条线——
	 * 给其中一条换个偏移量就能错开。
	 */
	routeOffset?: number;
	/** 从目标的哪一边进入。默认 `left`；`bottom` 用于"从下面喂进来"（如缓存 → 注意力） */
	enter?: 'left' | 'bottom';
}

/**
 * 一步的参考数据流图。
 *
 * 用于"这一步在整个算子里处于什么位置"很容易迷路的地方——MLA 尤其需要：
 * 它的 Q/K/V 各自经过压缩 / 归一化 / 升维 / 旋转，还有一份潜向量进缓存，
 * 光看单页的矩阵很难拼出全貌。
 *
 * 图是**静态的**（不参与动画计时），只把本步正在算的节点高亮出来，
 * 这样它是一张随时可查的地图，而不是又要等一遍的动画。
 */
export interface DataflowSpec {
	nodes: DataflowNode[];
	edges: DataflowEdge[];
	/** 本步正在算的节点（高亮） */
	active?: string[];
	/** 怎么看这张图（图例之外的一句话） */
	hint?: string;
}

// ── 步骤 ──────────────────────────────────────────────────
export interface Step {
	id: string;
	kind: PrimitiveKind;
	/** 解说文案 */
	label: string;
	/** KaTeX 公式 */
	formula?: string;
	/**
	 * 时间轴上的停留时长**不在这里写**——它由本步的动画工作量算出来
	 * （`core/steps.ts` 的 `stepDurationMs`），这样每一页的"每格耗时"都一致，
	 * 不会出现"这页的矩阵乘很快、那页很慢"。插件只管写清楚有哪些视图。
	 */
	tensors: TensorView[];
	anim?: AnimDirective[];
	/** 可选：渲染在张量视图**下方**的参考数据流图（见 `DataflowSpec`） */
	diagram?: DataflowSpec;
}

// ── 插件执行上下文 ────────────────────────────────────────
export interface ModelConfigLike {
	seq_len: number;
	d_model: number;
	num_layers: number;
	num_heads: number;
	head_dim: number;
	first_k_dense: number;
	dense_intermediate: number;
	moe_intermediate: number;
	num_routed_experts: number;
	num_shared_experts: number;
	top_k: number;
	vocab_size: number;
	/** 是否 decoder-only（因果掩码）。注意力类插件应据此决定是否屏蔽未来位置 */
	causal?: boolean;
	// 低秩注意力（MLA）需要的维度，可选
	/** Q 的低秩维度 */
	q_lora_rank?: number;
	/** KV 共享的潜向量维度 */
	kv_lora_rank?: number;
	/** 每头不含位置编码的 Q/K 维度 */
	qk_nope_head_dim?: number;
	/** 每头位置编码占的维度 */
	qk_rope_head_dim?: number;
	/** 每头 V 的维度 */
	v_head_dim?: number;

	// 混合注意力（DeepSeek V4）需要的维度，可选
	/** 滑动窗口大小：最近这么多个 token 用**精确** KV（真实 128） */
	swa_window?: number;
	/**
	 * 逐层的压缩比，长度 = `num_layers`：`0` 纯 SWA、`4` CSA、`128` HCA。
	 * 真实 V4-Flash 是 43 层，demo 用官方 4 层 parity harness 的结构。
	 */
	compress_ratios?: number[];
	/** RoPE 的底数（真实 10000） */
	rope_theta?: number;
	/**
	 * 压缩层用的 RoPE 底数（真实 40000）。
	 *
	 * 纯滑窗层用主 RoPE，而 CSA / HCA 层整层换成压缩版 YaRN RoPE——**Q、滑窗 KV、
	 * 压缩 KV 三处都用它**，所以底数必须跟着层走，不能全局写死一个。
	 */
	compress_rope_theta?: number;
	/** indexer 的头数与每头维度（只有 CSA 层有 indexer） */
	index_n_heads?: number;
	index_head_dim?: number;
	/** indexer 给压缩条目打分后取 top-k 参与注意力 */
	index_topk?: number;
	/** 输出投影的分组低秩（真实 `o_groups=8` / `o_lora_rank=1024`） */
	o_groups?: number;
	o_lora_rank?: number;

	// 混合残差流（DeepSeek V4 的 mHC）需要的字段，可选
	/** 并行残差流条数（真实 4）；不填 = 普通残差（R1） */
	hc_mult?: number;
	/** Sinkhorn 归一化迭代次数（真实 20） */
	hc_sinkhorn_iters?: number;
	/** 门控下限 / Sinkhorn 除法的 eps（真实 1e-6） */
	hc_eps?: number;

	// hash 路由（DeepSeek V4 前几层不学路由）需要的字段，可选
	/** 前这么多个层用 hash 路由（真实 V4-Flash = 3）；不填 = 全学（R1） */
	n_hash_layers?: number;
}

export interface LayerCtx {
	layer: number;
	cfg: ModelConfigLike;
	/** 该插槽在本层的权重包，由 model 层提供；插件自行断言其类型 */
	w: unknown;
	/** 本层输入的 token id 序列（hash 路由查表用；其他插件可忽略） */
	tokenIds?: number[];
}

// ── 插件协议 ──────────────────────────────────────────────
/** 所有插槽 trace 的公共约束：必须暴露 out，供流水线串起来 */
export interface SlotTrace {
	out: Mat;
}

/**
 * 语义插件：定义一个算子的**数学含义**——"算什么"。
 *
 * 注意与"实现"的区别：MLA 是语义，FlashAttention 是实现。
 */
export interface SemanticsSpec<TTrace extends SlotTrace = SlotTrace> {
	id: string;
	slot: SlotId;
	/** 显示名 */
	name: string;
	/** 一句话说明 */
	desc: string;
	/** KaTeX 公式 */
	formula?: string;
	defaultDepth: Depth;
	maxDepth: Depth;
	/** 执行：返回全部中间量 */
	compute(input: Mat, ctx: LayerCtx): TTrace;
	/** 把中间量翻译成声明式步骤（含动画指令） */
	steps(trace: TTrace, ctx: LayerCtx): Step[];
	/**
	 * 该插件需要什么权重。返回值会作为 `ctx.w` 传给 compute / steps。
	 * model 层在初始化时对每个已注册插件调用一次。
	 * 不需要自己权重的插件（如纯逐元素算子）可省略。
	 */
	makeWeights?(rnd: () => number, cfg: ModelConfigLike, layer: number): unknown;
}

/**
 * 执行实现插件：定义一个算子的**执行方式**——"怎么算"。
 *
 * 与语义插件**正交**：MLA 是语义（算什么），「矩阵吸收合并」是执行方式（怎么算）。
 * 同一个语义可以配多个实现，只要实现声明 `supports` 它；组合不合法时 UI 不提供该选项。
 *
 * 实现**可以完全接管**这个插槽：
 *   - 给了 `compute` / `steps` → 用它的（如吸收合并，步骤序列完全不同）；
 *   - 不给 → 退回语义插件的（大多数"实现"只是换个 kernel，数据一模一样）。
 */
export interface ImplementationSpec<TTrace = SlotTrace> {
	id: string;
	slot: SlotId;
	/** 显示名，如 `矩阵吸收合并` */
	name: string;
	/** 一句话说明（UI 上拼在语义描述后面） */
	desc: string;
	/** 声明支持哪些语义 id */
	supports: string[];
	/** 自己算：给了就接管这个插槽的计算 */
	compute?(input: Mat, ctx: LayerCtx): TTrace;
	/** 自己出步骤：给了就接管这个插槽的步骤 */
	steps?(trace: TTrace, ctx: LayerCtx): Step[];
	/**
	 * 自己需要的权重；不给就用语义插件的。
	 *
	 * 注意吸收合并**不需要新权重**——它用的是同一批权重算出来的合并矩阵，
	 * 所以它没实现这个钩子，而是在 `compute` 里现场合并
	 * （真实推理只在加载权重时合并一次）。
	 */
	makeWeights?(rnd: () => number, cfg: ModelConfigLike, layer: number): unknown;
}

/** KV Cache：v0 只留接口，注册表为空 */
export interface KvCacheSpec {
	id: string;
	name: string;
	desc: string;
	compute(trace: unknown, ctx: LayerCtx): unknown;
	steps(trace: unknown, ctx: LayerCtx): Step[];
}

/** Overlay（并行 / 量化）：v0 只留接口，注册表为空 */
export interface OverlaySpec {
	id: string;
	name: string;
	desc: string;
	/** 叠加式：不改数据，只往已有步骤上追加视觉标注 */
	decorate(steps: Step[], ctx: LayerCtx): Step[];
}
