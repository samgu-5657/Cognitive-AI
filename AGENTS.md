# Cognitive AI — 长期开发规则 (AGENTS.md)
# 适用于 AI 编程代理与人类开发者的持续开发约束

## 1. 根本禁令（不可违反）

1. 不加载预训练大语言模型（GPT / Claude / Qwen 等）作为认知核心。
2. 不调用在线 LLM API 代替分类、推理或自主探索。
3. 不把"写更好的提示词"当作本项目的实现方式。
4. 不把随机输出、硬编码示例伪装成理解结果或"模型学到的知识"。
5. 不把规则引擎 / 知识库更新误称为神经网络 / 权重更新。
6. 区分三种学习并如实记录：
   - A. 情景记忆更新（新经历/观察入库）；
   - B. 认知结构更新（概念/类别/属性/关系/规则修改，带版本）；
   - C. 神经网络参数更新（未来独立模块，必须从随机初始化开始）。
7. 禁止自主学习过程中自行修改并执行自身的 Python 源代码。
   自我改进只允许发生在受管理的知识、规则和（未来）模型参数中。
8. 不允许自主探索无限运行：`loop.py` / CLI `explore` 必须有 steps / 时间预算。
9. 不允许未经验证的重大结构修改直接覆盖整个认知体系：
   高影响修改必须走提案 (SchemaChangeProposal) → 针对性验证 → 应用/拒绝/回滚。
10. 不得为让功能"看上去可用"而伪造搜索结果或证据。

## 2. 架构边界

- `models.py`: 纯数据结构，无 IO、无推理。
- `memory.py`: 唯一持久化入口 (SQLite)。其他模块不得自行读写 DB 文件。
- `ontology.py`: 概念 CRUD、多重继承、循环检测、属性继承查询。
- `reasoner.py`: 可解释规则推理，返回结构化结果 + 证据追溯。未知必须标记 unknown。
- `novelty.py`: 新颖性/缺口检测。诚实标记中文理解能力的边界。
- `schema_evolver.py`: 唯一允许修改概念结构的模块，负责版本快照与回滚。
- `curiosity.py`: 研究问题生成与评分。评分公式与权重集中管理、可配置。
- `research.py`: `ResearchProvider` 接口 + `LocalCorpusProvider`（停用词剔除、
  词边界匹配、内容指纹 `content_hash`、相关性分级 direct/indirect/unrelated）。
  外部来源只做数据处理。关键词匹配不是语义理解（见 README 局限）。
- `policy.py`: 研究/探索策略常量唯一来源（佐证数、可信度阈值、冷却期、
  尝试上限、每步新增问题上限、停用词、近重复阈值、晋升门槛、分类阈值）。
  逻辑模块不得各自定义阈值。
- `lifecycle.py`: 知识生命周期（candidate→hypothesis→supported→verified，
  另有 contested/rejected/superseded）与晋升门检查 + 事件记录。
  自主管线最高只能写 hypothesis；supported/verified 必须显式 promote。
- `learner.py` / `loop.py`: 学习循环与预算控制。`Learner` 支持
  `enable_structural=False`（纯事实系统 B）；`consolidate()` 为验证式结构学习。
- `cli.py`: 薄命令行封装，不放核心逻辑。

## 3. 数据与持久化

- 默认数据库：`data/cognitive.db` (SQLite, 标准库)。
- 种子概念：`data/concepts.json`；本地资料：`data/corpus/`。
- 重启后知识、经历、修改记录、当前版本号必须仍然存在（用测试保证）。
- 每次认知结构修改必须记录：版本前后、原因、相关经历、证据、影响范围。

## 4. 知识状态

- 新知识默认不可靠：`verified` / `hypothesis` / `contested` / `unknown` 四态。
- 单一未验证来源不得修改核心概念体系。
- 冲突证据必须保留双方，不得静默覆盖。

## 4b. 研究→知识的证据门槛（审计后强制规则，Phase 2 扩展）

- `research_question` 返回固定绑定
  (`question_id`/`target_id`/`key`/`evidence_sources`/`action`/
  `applied_change_id`)，探索日志逐字记录，日志↔修改必须可互查。
- `inconclusive` 永不写入语义知识、永不产生结构版本。
- 证据命中必须提及目标概念（id 或名称）；问题文本与 `target_id`
  不一致时直接记 `inconclusive`，不写库。
- 知识库写入需要至少 `MIN_CORROBORATING_SOURCES=2` 个**独立来源**
  （按 `content_hash` 去重， excerpt 相似度≥阈值视为同一来源）
  给出**相同**的清洗后原子值；写入一律为 `hypothesis`。
- 单一来源的线索记为 `candidate` Claim，不写属性、不升版本。
- 抽取值只接受短原子值，拒绝句子残片。
- 已知属性遇到 corroborated 对立值时：旧行不动，记 contested Claim +
  contested 对手行 + lifecycle 事件（`record_conflict`）。
- 生命周期：candidate→hypothesis（自动上限）→supported（≥3 源+显式
  promote）→verified（≥3 源+具名确认+回归通过）；两文件一致永不自动
  verified。详见 `lifecycle.describe()` / `cli lifecycle --describe`。
- 无效尝试记 `attempts` + 冷却期，超限后保留不再强行研究。
- 同一 `(text, target)` 问题永不重复建行；每步新增问题有上限；
  `new_questions` 只统计真正新增的行。
- 结构学习（split/merge/reparent）只走 `consolidate()` + 回归门；
  纯事实系统设 `enable_structural=False` 即拒结构修改。

## 5. 运行方式 (Windows PowerShell)

```powershell
py -m pip install -e ".[test]"
py -m cognitive_ai.cli doctor
py -m cognitive_ai.cli concepts
py -m cognitive_ai.cli inspect cup
py -m cognitive_ai.cli gaps
py -m cognitive_ai.cli explore --steps 5
py -m cognitive_ai.cli history
pytest -q
```

## 6. 测试要求

- 每个新模块必须有明确职责和对应测试文件。
- 测试检验实际行为（分类、继承、冲突、回滚、预算停止、重启不丢失），不只检验函数存在。
- 修改核心分类体系前必须能运行回归子集；失败则拒绝/回滚。

## 7. 功能状态划分（提交时更新）

- 已实现：在 README "功能状态"中标记为 Implemented。
- 仅接口：标记为 Interface-only，并说明缺什么。
- 计划中：标记为 Planned，不创建空逻辑模块充数。

## 8. 神经网络 (未来)

- 未来引入 NN 时：从随机初始化开始，单独评估学习与持续更新能力，
  不得复用预训练权重冒充持续学习成果，需新增独立测试。
