# 进度

README 讲这个项目是什么，这里记录做到哪一步、实测结果怎样、删掉了什么。
下面的数字都是 2026-10-12 清理时重新跑出来的；没能复现的旧数字已经删掉，不再引用。

## 功能状态

### 已实现（有测试覆盖）

| 功能 | 位置 | 测试 |
|---|---|---|
| SQLite 持久化，重启不丢数据；旧库增量迁移 | `memory.py` | `test_memory.py`, `test_migration.py` |
| 概念多重继承、环检测、属性继承（离得最近的优先） | `ontology.py` | `test_ontology.py` |
| `is_a` / 属性 / 关系查询，前向规则（带例外），缺数据返回 `unknown`，冲突检查 | `reasoner.py` | `test_reasoner.py` |
| 结构修改提案 → 回归门 → 版本快照 → 回滚；`split_category` / `merge_categories` / `reparent` 等 | `schema_evolver.py` | `test_schema_evolution.py` |
| 知识缺口检测，观察的新颖性检查，三句式正则 | `novelty.py` | `test_learning_loop.py` |
| 研究问题生成与打分，去重，冷却期，尝试次数上限 | `curiosity.py` | `test_curiosity.py`, `test_exploration_audit.py` |
| 本地语料检索：停用词、词边界、内容哈希、相关性分级 | `research.py` | `test_evidence_independence.py` |
| ≥2 个独立来源才写库，单一来源只记 `candidate` Claim，冲突时双方都保留 | `learner.py` | `test_exploration_audit.py`, `test_evidence_independence.py` |
| 生命周期状态机，晋升条件，事件日志 | `lifecycle.py` | `test_lifecycle.py` |
| `classify`（match / ambiguous / novel / insufficient） | `learner.py` | `test_eval_concept_learning.py` |
| `consolidate`：按稳定的属性分歧拆分类别，必须过回归门 | `learner.py` | `test_transfer_experiment.py` |
| 带步数 / 时间预算的探索循环 | `loop.py` | `test_learning_loop.py` |
| 字符级 CNN / BiGRU，随机初始化，组隔离与泄漏检查 | `neural/model.py`, `trainer.py`, `dataset.py` | `test_neural.py` |
| 评测（准确率、P/R/F1、混淆矩阵、拒识率、关键词基线） | `neural/evaluator.py` | `test_neural.py` |
| 适配器：网络输出只能变成 `candidate` Claim | `neural/adapter.py` | `test_neural.py` |
| 未知样本隔离队列、人工审核、增量训练 + 回放、回归门禁、active 指针与回滚 | `neural/queue.py`, `gate.py`, `cycle.py` | `test_neural_3b.py`, `test_neural_transfer3b.py` |
| OOD 五档判定（阈值只用 val 定），候选类别发现 + 留一验证 | `neural/ood.py`, `discovery.py` | `test_neural_3c.py`, `test_neural_discovery_xfer.py` |
| 变量绑定多步规则推理（`?x` 合一、有界深度搜索、证明链、冲突报告，不写库） | `reasoner.py` | `test_chain_m1.py` |
| 规则生命周期（active/candidate/validated/disabled/rejected，变更记事件）与关系行真实状态传播 | `memory.py`, `models.py`, `reasoner.py` | `test_rule_lifecycle_m2.py` |
| 可组合中文语义解析（主体/谓词/对象/否定/能力模态，7 式切片） | `semparse.py`, `novelty.py` | `test_semparse_m3a.py` |

全量测试：`pytest -q` → 装了 torch 时 107 passed（98 旧 + 9 新）；没装 torch 时 87 passed、20 skipped。

### 只有接口

- `research.WebSearchProvider`：调用就抛 `NotImplementedError`，要用的话得有联网和来源审核方案。

### 计划中（还没有代码）

- 回归门改成按种子数据和历史快照做通用检查，不再写死 `cup`。
- `curiosity.IMPORTANT_CONCEPTS` 改成可配置，或者根据数据算出来。
- 扩充神经网络训练数据。现在的规模下 CNN 比不过关键词基线，继续调模型意义不大。

## 实测结果

### 神经分类器（`data/neural/`，CPU，seed 7，300 epoch）

复现命令：`neural-train --model cnn --tag cnn-v1 --seed 7 --epochs 300`，然后 `neural-eval` / `neural-compare`。

| 模型 | val L1 | test L1 | test L2 | 关键词基线 test L1 | 未知拒识率 (8 条) |
|---|---|---|---|---|---|
| char-CNN | 0.574 | 0.420 | 0.261 | 0.435 | 0.75 |
| char-BiGRU | — | 0.377 | 0.217 | 0.435 | 0.00 |

- test 集有 69 条带标签样本，5 个一级类别，随机猜的准确率是 0.2。
- 两个模型都没超过手写关键词基线。CNN 结果重跑一致；GRU 比旧 README 里写的（0.391 / 0.275）低，
  说明 GRU 在不同环境下结果不稳定。
- OOD（`neural-ood-build`）：拟合出的温度 T=2.3 把 NLL 降下来了，但 ECE 反而从 0.170 升到 0.194，
  所以判定时用的是原始概率（T=1）。

### 知识库评测与对照实验（测试里打印的结果，`pytest -s`）

- `test_eval_concept_learning.py`：一共 10 个用例。分类准确率 4/4，误写入 0 次，
  证据不足时正确弃权 3/3，未知概念能识别出来。
- `test_transfer_experiment.py`：5 个留出样例。A（不学习）、B（只学事实）、C（事实 + 结构拆分）
  都是 5/5，只有 C 多拿到 1 个子类别加分。
- `test_neural_transfer3b.py`：3 个新样本 + 12 个旧样本。带回放的增量训练旧样本保持率 0.917，
  不带回放的是 0.75。
- `test_neural_discovery_xfer.py`：只有 1 个留出词（铁锅）。

以上每组只有 3–12 个样本，只能说明代码路径跑得通，不能用来证明"学习带来增益"。

## 已移除的部分（2026-10-12 清理）

下面这些模块都删了（约 1.2 万行代码 + 34 个测试文件），可以在 git 历史 `6732730` 及之前的提交里找到：

| 移除内容 | 原来的说法 | 实际情况 |
|---|---|---|
| `interact.py`（chat REPL） | 交互教学 | 1300 多行的中文命令分发，把下面这些模块串在一起 |
| `semantics.py`, `phraselearn.py`, `semlearn.py`, `constructions.py`, `verbalize.py`, `dialog.py` | 组合式中文理解、可学习语义、双向构式、多轮对话 | 正则槽位模板，加上从示例里归纳的前缀 / 后缀字符串改写规则 |
| `cognitive_cycle.py`, `metacog.py`, `executor.py`, `cogtask.py`, `cog_planners.py`, `operations.py`, `explainer.py`, `answers.py`, `verify.py` | 统一认知循环、元认知、独立验证 | 按正则把输入分到"数学 / 逻辑 / 概念"三条路径，再按失败原因字符串做 if-else 分支 |
| `analogy.py` | 结构类比迁移 | 比较执行轨迹的目标类型、操作名序列和槽位键有多少重合 |
| `chainreason.py` | 多步证明链 | 规则里可以带变量（`?x`），做有深度上限的证明搜索，不写库。实现本身是成立的，但只有已删除的执行器调用它，种子规则也都是具体规则，用不上变量。如果以后要给 `reasoner.py` 加变量规则，可以从这里移植回来 |
| `rulelearn.py`, `learnloop.py` | 可学习认知技能 | 从示例拟合仿射、取模、单属性合取和决策表规则，再在受限 DSL 里执行。用例都是数值或玩具状态，和概念分类无关 |
| `world.py` | 开放式学习闭环微世界 | 一张 3 状态 × 2 动作的写死转移表 |
| `mathengine.py` | 自研符号引擎 | 四则运算、一元一次方程、多项式求导，和项目主题无关 |
| `web/` | 本地认知工作台 | FastAPI + 原生前端，套在上面这些模块外面 |
| `neural/intent.py`, `data/semantics/` | 意图网 | 只服务于已删除的正则解析器，测试准确率 0.745，还不如规则 |
| `neural/continual.py` | 3B 预留 | 已经被 `queue.py` / `gate.py` 取代，没有任何代码调用它 |

清理时顺手修的问题：

- `learner.classify` 里有一段重复的导入和赋值，`consolidate` 末尾有一段永远执行不到的代码。
- 好几个 neural CLI 命令写死了 `ROOT/models`，没读 `COGNITIVE_AI_MODELS` 环境变量。
- `neural-ood` 要用的 OOD 文件以前没有任何命令能生成，现在加了 `neural-ood-build`。
- `test_migration.py` 原来依赖作者本机的 Windows 临时文件，在其他机器上永远跳过。
  现在改成在测试里自己构造一个旧版数据库。
- `test_neural_3c.py` 里有 5 个测试需要 torch，但没加跳过标记，没装 torch 时会直接失败。
- 改掉了两处与代码不符的注释：OOD 实际用的是原始概率，不是"校准后概率"；
  评测基线是手写的关键词表，不是"符号解析器"。
