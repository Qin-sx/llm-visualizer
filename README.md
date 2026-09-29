# LLM Visualizer

可插拔地拆解大模型的**注意力**与 **MoE** 内部机制 —— 按 DeepSeek R1 的结构等比缩小，
用逐步骤动画把每一步的矩阵运算演给你看。

> 讲的是**机制**，不是知识：维度按 R1 结构等比缩小，权重是固定 seed 的随机值。

## 快速开始

宿主机不需要装 Node，全程在容器里跑：

```bash
./dev.sh up        # 起服务并打开浏览器 → http://localhost:9000
./dev.sh down      # 停止
./dev.sh logs      # 实时看日志
```

| 命令 | 作用 |
| --- | --- |
| `./dev.sh check` | 类型检查（svelte-check） |
| `./dev.sh verify` | 端到端校验：在 Node 里跑参考模型与插件，断言数学正确性 |
| `./dev.sh build` | 产出静态站点到 `./build` |
| `./dev.sh restart` | 重启（页面报 500 时先试这个） |

端口默认 **9000**，可用 `PORT=9100 ./dev.sh up` 覆盖。不想用脚本就直接
`docker compose up -d dev` / `docker compose run --rm check`。

## 界面能看什么

- **Attention 插件**：多头注意力（MHA）/ MLA（DeepSeek R1 真实用的那个）
- **Attention 实现**：选到 MLA 时多一个「矩阵吸收合并」执行方式，输出与朴素版逐元素相同
- **FFN / MoE 插件**：dense FFN / DeepSeekMoE（路由 → 选专家 → 专家 → 共享专家叠加）
- **展示阶段开关**：Embedding / Attention / FFN-MoE / LM Head 四段可开关，关掉的**整段从流程里去掉**
- **检视层**：4 层任选，前 2 层 dense、后 2 层 MoE
- **流程总览**：整条链的张量方框图，点方框跳到该步；当前步的输入从哪来用带箭头的线画出
- **每步的数据流图**：分层 DAG，高亮本步正在算的节点
- **KV Cache 视图**：缓存里到底躺着什么，并与 MHA 的缓存体积对比（小 56.9×）
- **播放器**：播放 / 暂停 / 单步 / 拖动 / 变速

## 目录结构

```
src/lib/
  core/        与算子无关的地基：类型、矩阵运算、注册表、时间轴、步骤时长规则
  slots/       算子插件（算什么）：attention/{mha,mla,absorb}、ffn-moe/{dense-ffn,deepseek-moe}
  stages/      阶段（流程里的一个环节）：embedding、lm-head
  flows/       流程：llm（Embedding → Attention → FFN/MoE → LM Head）
  model/       参考模型：缩放配置、随机权重、前向
  components/  渲染层：12 种张量视图（矩阵乘 / 掩码 / 拼接 / 变换 / 加权求和 / 查表 / 缓存
               以及 shape / row / matrix / tiles / bars），不认识视图来自哪个算子
  overlays/    并行 / 量化标注（只留接口，尚未实现）
src/routes/    页面：主页面 + licenses（第三方许可清单）
scripts/verify.ts   端到端校验
```

## 四层可插拔

| 层 | 管什么 | 加东西要动哪 |
| --- | --- | --- |
| **Flow** | 一条流程 = 一串阶段 | 新写 `flows/xxx.ts` + `registerFlow()` |
| **Stage** | 流程里的一个环节 | 新写 `stages/xxx.ts`，在流程里列进去 |
| **Semantics** | 算什么（如 MLA） | 新写 `slots/<插槽>/xxx.ts` + `register()` |
| **Implementation** | 怎么算（如吸收合并） | 同目录 + `registerImplementation()` |

四层互不干扰：`core/` 里没有"按算子分支"的逻辑，加插件、换实现、加流程都不需要改 core 与组件。

## 校验

```bash
./dev.sh verify
```

在 Node 里直接跑一遍参考模型与插件，断言：形状与数值（softmax 归一化、因果掩码、RoPE 保范数）、
矩阵乘的形状真的能乘、吸收合并与朴素版**逐元素相同**、可插拔性、播放器能翻到底、确定性。

## 第三方许可

`/licenses/` 页列出随构建产物分发的第三方软件及其许可。注意 `gsap` 用的是
GreenSock 标准许可（非 MIT），其条款要求保留版权声明。
