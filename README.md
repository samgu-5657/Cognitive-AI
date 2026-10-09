# Cognitive AI — 持续学习的人工认知系统 (Phase 1 原型)

受人类认知过程启发、**不使用预训练 LLM** 的本地认知系统。
内部思考使用概念/记忆/规则等可检查的数据结构，中文只是输入输出形式之一。

> 诚实边界：第一阶段没有文本理解模型，不能可靠理解任意中文句子。
> 提供结构化观察接口 + 有限中文规则解析（`X是Y` / `X有Y` / `X可以Z`）。

## 功能状态

- Implemented: 数据模型、概念体系（多重继承+循环检测）、SQLite 持久化
  （含 Phase 2 向后兼容迁移：`lifecycle_events` 表 + evidences 指纹列）、
  规则推理（继承/例外/冲突/unknown）、新颖性检测、结构演化（版本+回滚）、
  好奇心评分+问题队列、本地语料研究（停用词/词边界/内容指纹/相关性分级）、
  知识生命周期（candidate→hypothesis→supported→verified + contested/
  rejected/superseded，自主管线最高 hypothesis）、分类器 `classify`
 （match/ambiguous/novel）、验证式结构学习 `consolidate()`、
  10 案例评测集 + A/B/C 对照实验、无限运行防护、CLI、全部测试。
- Interface-only: `WebSearchProvider`（显式抛 `NotImplementedError`，不伪造结果）。
- Planned: 神经网络模块（随机初始化，独立评估）、公开网页搜索、更丰富的中文解析。

## 快速开始 (Windows PowerShell)

> 本机 `python` 指向 Microsoft Store 存根（执行失败），请使用 `py` 启动器。
> 控制台默认编码 (GBK) 可能把中文显示为 `????`，JSON 数据本身为 UTF-8，不影响存储与测试。

```powershell
cd C:\AI\cognitive-ai
py -m pip install -e ".[test]"
py -m cognitive_ai.cli doctor
py -m cognitive_ai.cli concepts
py -m cognitive_ai.cli inspect cup
py -m cognitive_ai.cli gaps
py -m cognitive_ai.cli questions
py -m cognitive_ai.cli explore --steps 5
py -m cognitive_ai.cli history
py -m cognitive_ai.cli experiences --limit 10
pytest -q
```

以 `src` 直接运行（未安装时）：

```powershell
$env:PYTHONPATH = "C:\AI\cognitive-ai\src"
python -m cognitive_ai.cli doctor
```

## 演示场景

```powershell
# 场景一：基本分类（杯子的多重归属存储在概念体系中）
py -m cognitive_ai.cli inspect cup

# 场景二：发现未知概念（结构化观察，用 --json-file 避免 PowerShell 引号剥离）
py -m cognitive_ai.cli observe --json-file obs_novel.json
# obs_novel.json 内容示例：
# {"name": "teapot", "parents": ["vessel"],
#  "attributes": {"material": "clay", "function": "brewing"}}

# 场景三：拆分过于宽泛的类别（测试覆盖 vessel→open/closed split + 回滚，
# 见 tests/test_learning_loop.py::test_split_scenario）；交互式拆分：
py -m cognitive_ai.cli history
#   在 Python 中：
#   from cognitive_ai.memory import Memory
#   from cognitive_ai.schema_evolver import SchemaEvolver
#   from cognitive_ai.learner import default_regression
#   m = Memory("data/cognitive.db")
#   e = SchemaEvolver(m, default_regression)
#   cid = e.propose("split_category",
#       {"category": "vessel",
#        "groups": {"open_vessel": ["bowl"], "closed_vessel": ["bottle"]}},
#       reason="bowl/bottle 在 has_lid 上存在稳定差异")
#   print(e.apply_proposal(cid))

# 场景四：自主发现研究问题并研究
py -m cognitive_ai.cli gaps
py -m cognitive_ai.cli explore --steps 5

# 场景五：持久化（退出重进后知识仍在；每次 CLI 调用都是新进程，直接证明）
py -m cognitive_ai.cli concepts
py -m cognitive_ai.cli doctor

# 场景六：冲突与回滚（verified 值不被覆盖；回滚移除 contested 行）
py -m cognitive_ai.cli observe --json-file obs_conflict.json
# obs_conflict.json 内容示例：{"name": "杯子", "attributes": {"material": "glass"}}
py -m cognitive_ai.cli history
py -m cognitive_ai.cli rollback 7
```

## CLI 命令

| 命令 | 说明 |
|---|---|
| `concepts` | 查看当前概念体系 |
| `inspect <id>` | 查看概念属性/关系/继承 |
| `observe --json '{...}'` / `--json-file f` | 添加结构化观察并学习 |
| `parse <句子>` | 有限中文规则解析演示 |
| `query --json '{...}'` / `--json-file f` | 推理查询 (`is_a`/`get_attribute`/`relations`/`conflicts`) |
| `gaps` | 查看知识缺口 |
| `questions` | 生成候选研究问题 |
| `explore --steps N` | 运行 N 步自主探索（有预算上限） |
| `experiences --limit N` | 查看学习经历 |
| `history` | 查看结构修改历史与版本 |
| `rollback <version>` | 回滚至指定版本 |
| `doctor` | 健康检查 |

## 三种学习（本项目严格区分）

- A. 情景记忆更新 → `experiences` 表。
- B. 认知结构更新 → `schema_changes` + `schema_versions`，可回滚。
- C. 神经网络参数更新 → Phase 1 未实现，未来独立模块。

## 已知限制

1. 中文理解仅限结构化输入 + 三种固定句式；任意句子会诚实返回未解析。
2. 证据独立性是启发式，不是语义理解：`content_hash` 识别逐字复制，
   excerpt Jaccard（≥0.85）识别轻度改写；改写充分的同源文本、同一事实的
   多手转述会被误判为独立——上限已在测试与文档中声明。相关性分级
   （direct/indirect/unrelated）只是"是否提及目标+是否有 key 模式"的表面
   检查。研究证据仅来自本地语料；证据不足时保持 `inconclusive`，
   不写库、不升版本。写入需 ≥2 独立来源相同的短原子值（`policy.py`）。
   审计前旧版本曾写入少量来源不明的 hypothesis 属性，历史数据未自动删除，
   如需清理可先备份 `data/cognitive.db` 再 `rollback`。
3. 属性值冲突默认保留双方 + 标记 `contested`，不自动覆盖 `verified`；
   verified 只能经显式 `promote ... --confirmed-by` 达成。
4. 神经网络、在线搜索尚未实现（见 AGENTS.md）。
5. `claims`/`evidences`/`questions` 记录不在概念快照内，回滚恢复概念结构，
   但保留冲突与研究的历史记录（有意为之：历史不可篡改）。

## 评测与对照实验（Phase 2，复现命令）

```powershell
pytest tests/test_eval_concept_learning.py -q   # 10 案例评测集（tests/eval_data/）
pytest tests/test_transfer_experiment.py -q -s  # A/B/C 对照（-s 显示对照表）
pytest tests/test_lifecycle.py tests/test_evidence_independence.py tests/test_migration.py -q
```

评测基线结果（隔离库，`pytest -s` 可见明细表）：
`EVAL accuracy=1.00, false_writes=0, abstention=3/3, unknown_detection=yes`；
对照实验 A/B/C 在 5 个 held-out 测试上同得 5/5，仅 C 通过验证式拆分
额外获得子组 bonus——结构学习带来可验证的迁移增益，但基础分类无差异
（诚实结论：小规模评测上动态结构不总是最优）。

## 神经网络（Phase 3A：概念识别网，CPU 可运行）

`src/cognitive_ai/neural/`：字符级 CNN（默认）与双向 GRU，随机初始化，
无预训练权重、无外部词向量。torch 为可选依赖（`pip install -e ".[neural]"`），
缺失时规则系统不受影响。

```powershell
py -m cognitive_ai.cli neural-train --model cnn --tag cnn-v1 --seed 7 --epochs 300
py -m cognitive_ai.cli neural-eval --tag cnn-v1
py -m cognitive_ai.cli neural-compare --tags cnn-v1,gru-v1
py -m cognitive_ai.cli neural-predict 杯子 --tag cnn-v1
pytest tests/test_neural.py tests/test_neural_continual.py -q
```

实测（`data/neural/` 手工种子数据：train 174 / val 54 / test 77，
组级隔离零泄漏；权重在 `models/`，git 忽略）：

| 模型 | val L1 | test L1 | test L2 | test 基线 | 未知拒绝率 |
|---|---|---|---|---|---|
| char-CNN | 0.574 | **0.420** | 0.261 | 0.435 | 0.75 |
| char-GRU | 0.556 | 0.391 | 0.275 | 0.435 | 0.00 |

诚实结论：小数据+测试词完全未见的设定下，CNN 与粗糙关键词基线
基本打平（差 1 个样本），GRU 更差且过度自信（拒绝率 0）。
CNN 的价值在于校准的弃权（0.75）而非精度；L2 尚弱。
网络输出恒为 `candidate`，经证据+生命周期门后才能入库，
`neural-predict --submit` 只建 Claim，不断言知识、不升版本。
3B 预留：`continual.py`（未知日志/样本检疫/版本化并入/回放门禁），
无在线权重更新。

能力边界：数百汉字词汇、短句、人工分布；未见词多靠共享单字猜测，
常过度自信；不得称为通用中文理解。

## 下一阶段计划

1. 更严格的研究证据门槛：同一结论需两个独立来源才进入知识库。
2. 规则学习：从稳定的属性共现中提议新规则（走提案+回归验证）。
3. 类别拆分/合并的自动触发条件（基于持续的反例计数，而非单次观察）。
4. 神经网络模块（随机初始化，独立 `C` 类学习评估，与规则引擎对比测试）。
5. `ResearchProvider` 的只读本地网页镜像源（仍不伪造结果、无密钥可运行）。
