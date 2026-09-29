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
export type SlotId = 'embedding' | 'attention' | 'kvcache' | 'ffn-moe' | 'lm-head';

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
	 * 布局：同一 `row` 的视图**横向并排**（如 Q·Kᵀ → mask → P·V 流水线）；
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
	 * 占满整行（跨过 `step.tensors` 那个自动填充网格的所有列）。
	 *
	 * 不 wide 的视图只拿到一格（约 220px）：一句长注解挤在那一格里会折成好几行，
	 * 旁边要是还有别的视图，nowrap 的文本会横穿过去。注解长的视图（`shape` 的 `note`、`row` 的 `label`）把它打开。
	 */
	wide?: boolean;
	/**
	 * 在顶部"流程总览"里作为一个节点出现；值为显示名（如 `'P_h'`）。
	 * 矩阵乘的操作数/结果会自动进总览，这个字段用于补充**单目运算**的产物
	 * （如 softmax 之后的权重矩阵），从而让 `S_h → P_h` 也出现在流程里。
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
}

/** 小矩阵热力图（仅用于 ≤16×16 这类真正需要看整体形态的矩阵，如注意力矩阵） */
export interface MatrixView extends TensorViewBase {
	kind: 'matrix';
	data: Mat;
	/** 高亮的单元格 */
	highlight?: [number, number];
	/** 是否按行优先逐格揭示（用于 softmax 这类逐元素计算） */
	animated?: boolean;
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
export interface MatRef {
	name: string;
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
	/** 额外标量（如 1/√d_h），会显示在式子里 */
	scaleNote?: string;
}

/**
 * 掩码过程视图：动画演示 j > i 的位置被逐个置为 −∞。
 *
 * 未处理 → 原值淡显；处理且保留 → 原值；处理且屏蔽 → −∞。
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
	 * 可选：输入块先做**逐格掩码**（如因果掩码 j > i → −∞），再做逐行变换。
	 * 有了它，掩码和后面的变换共用同一个矩阵块——`S_h` 只画一次。
	 */
	preMask?: Mat;
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
	 * 画出来——同一个矩阵在一页里出现两次、还挂着两个名字
	 * （用户指出："我是指 O_h 旁边有一个 O_h′，放在同一行"）。
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
}

export interface LayerCtx {
	layer: number;
	cfg: ModelConfigLike;
	/** 该插槽在本层的权重包，由 model 层提供；插件自行断言其类型 */
	w: unknown;
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
