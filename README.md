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

## 持续学习（Phase 3B：门禁式模型更新，CPU 可运行）

未知输入→隔离队列→人工审核→增量训练（新样本+旧回放）→冻结回归门禁→
指针切换（active.json），失败保留现状、可回滚。`models/` 不入库。

```powershell
py -m cognitive_ai.cli neural-cycle 激光打印机
py -m cognitive_ai.cli neural-queue --status pending
py -m cognitive_ai.cli neural-review <id> --l1 entity --l2 electronics --accept
py -m cognitive_ai.cli neural-incremental --base cnn-v1 --out cnn-cand1 --epochs 30
py -m cognitive_ai.cli neural-queue-export --out reviewed.jsonl
py -m cognitive_ai.cli neural-gate --base cnn-v1 --candidate cnn-cand1 --new reviewed.jsonl --promote
py -m cognitive_ai.cli neural-active
py -m cognitive_ai.cli neural-rollback
pytest tests/test_neural_3b.py tests/test_neural_transfer3b.py tests/test_neural_continual.py -q
```

真实运行（5 个审核新词 + 10 条回放，30/100 epoch 两候选）：
门禁均**拒绝晋升**——回归 0.574→0.463/0.444（超 0.02 容忍度），
新词准确率仅 0.2/0.6。活动模型保持 `cnn-v1`。
结论：小批量回放不足以在吸收新词的同时保住旧知识，门禁按设计拦下；
对照实验（全量回放）显示 replay 版旧知识保持 0.917 优于纯新样本版 0.75。
A/B/C/D 小实验：A(冻)新0.33/旧0.58，B(新)C(新+回放)D(重训)新词皆0.67，
旧知识 C 0.917 > D 0.833 > B 0.75——回放有效，但样本量小，属初步信号。

## 交互教学（Phase 4A：`chat` 多轮教学，UTF-8）

```powershell
py -m cognitive_ai.cli chat
```

首屏显示知识版本与活动模型；中文快捷（帮助/概念/查看/分类/教/
观察/标签/假设/纠正/探索/状态/队列/审核/经历/历史/模型/退出）与英文
命令等效。`教 <名称>` 分步要属性→功能→类别（可跳过，未知保持未知）；
`观察/标签/假设/纠正/探索` 五种操作分别落 observation/审核/Claim/
留痕纠正/研究队列，互不替代；不可解析输入明确拒绝且**零写入**；
`分类` 同时给出规则解析、KB 匹配、模型候选、未知判断与缺口。
教学演示：教紫砂壶（属性→功能→类别）→`状态`可见暂定概念与缺口；
`纠正` 保留原值；`neural-*` 训练/门禁命令复用原有接口（chat 内不直训）。
`COGNITIVE_AI_DATA` 可指向隔离数据目录（测试即用此隔离）。

限制：只懂帮助列出的命令与三种简单句式（X是Y/X有Y/X可以Z）；
自由中文闲聊、指代消解、跨轮指代均不支持；类别名按 id/名称双查，
未知归属永不虚构；单字/全未知串判 unjudgeable。

## 组合式中文理解（Phase 4B：语义帧 + 上下文 + 意图网，CPU 可运行）

```powershell
py -m cognitive_ai.cli neural-intent-train --tag intent-v1
py -m cognitive_ai.cli neural-intent-eval --tag intent-v1
pytest tests/test_semantics.py -q
```

`semantics.py`：槽位组合规则（词法 markers + 通用槽 `{NP}是{NP}` 等，
非整句硬编码）输出统一帧
（intent/entities/predicates/modality/scope/source/confidence/
unresolved/ambiguities）；否定用结构前瞻（不锈钢/不同不误伤）、
hedge 提升后做槽匹配、纠正去嵌套、条件逗号兜底、量词清理。
`chat` 内 `理解 <句>` 显示帧与将执行操作，`确认` 执行（未消解指代
拒绝执行，曾抓到真 bug：代词成概念）；`纠正` 关联会话 last_observation；
代词按会话上下文消解（多候选必问）；上下文落库可看可清，会话隔离。
空属性归类恒 novel 已修：无属性/无冲突判 insufficient，无匹配有冲突
才判 novel（缺属性≠新类别）。
冻结 test 对比：规则 47/47=1.0，意图网 0.745（negate 0.375、
unknown 0.0——无 unknown 训练样本，恒强制已知意图；职责分离的代价，
如实报告）。

## 开放世界发现（Phase 3C：未知检测 + 候选概念，CPU 可运行）

```powershell
py -m cognitive_ai.cli neural-ood 跑步机 --second gru-v1
py -m cognitive_ai.cli neural-discover --tag cnn-v1
pytest tests/test_neural_3c.py tests/test_neural_discovery_xfer.py -q
```

未知检测（原型由 train 定、阈值由 VAL 定，TEST 只报告）：
 verdict 分布（69 有标）similar 24 / confident 20 / unjudgeable 21 /
 novel 2 / uncertain 2；接受精度 12/20=0.60（裸模型 29/69=0.42，
 以覆盖 29% 为代价将错误率减半）；未知 FAR 1/8（激光打印机两网一致
 误判 entity——诚实失败）；校准温度 T=2.3 降 NLL 但 ECE 0.170→0.194
 未改善，故 verdict 用原始概率（VAL 上二者接受精度相当）。
 漏网：跑步机/花钱/跳舞两网一致误判；代价：自行车等可知词被连带弃权。
候选发现：审核样本满足内聚+同族 L2 全清+LOO≥0.75 才提 new_l2，
 否则 insufficient；L1 新类只存候选、永不自动扩展。
结构对照（held-out 铁锅/12 冻结旧例，同预算）：
 A 只检测（铁锅 unjudgeable，旧 0.583），B 只加事实（L1 entity，旧 0.917），
 C 验证后扩展（L2=kitchenware 命中，旧 0.917，门禁 delta +0.33）——
 小样本上的真实迁移增益；若验证失败 C 自动退回 base（已测试）。

## 通用认知执行（Phase 4C：任务→操作→轨迹→答案，CPU 可运行）

```powershell
py -m cognitive_ai.cli chat
# 解题 计算3+4*2 / 解题 杯子的材质是什么 / 解题 如果下雨地面就会湿。已知下雨。
pytest tests/test_cogexec.py -q
```

`CognitiveTask`（已知/目标/可用知识/操作/未决/验证标准）→9 种内部
操作（检索/查属性/用规则/比解释/计算/写工作记忆/检查/提缺口/改计划，
全部记录输入·依据·输出·状态）→执行循环（验证→改计划/答/转缺口）→
`explainer` 只按轨迹生成答案（未验证不出确定结论）。
自研符号引擎：精确有理数算术、一元一次方程（回代验证）、多项式求导
（幂/和/积规则逐项记录，独立复算验证），禁 `eval`、无外部求解器。
三域同流程：数学（3+4*2=11、2x+3=7、x^3+2x 未见组合）、概念
（verified/继承/hypothesis/unknown 四态区分，实例覆盖阻止盲继承）、
逻辑（前提具备则推出、前提缺失/ contested 则拒断+缺口）。
12 项验收全过；失败转 curiosity 缺口、神经权重零触碰、轨迹跨重启可查。
人工规则与网络能力：操作 origin 明确标记 hand_rule；神经网络未参与
解题推理（只做分类候选），如实区分、不混为一谈。

## 可学习的认知技能（Phase 4D：归纳→验证→组合，CPU 可运行）

```powershell
py -m cognitive_ai.cli rule-induce --train ex.jsonl --domain math --name double --save
py -m cognitive_ai.cli rule-validate lr_double --train ex.jsonl --heldout ho.jsonl --activate
py -m cognitive_ai.cli rule-list
pytest tests/test_rulelearn.py -q
```

`LearnableRule` 结构化存 DB（`learnable_rules` 表）：slot/transform 两族
受限 DSL（bind/check/compute/emit，禁 eval，无代码生成）；归纳器只拟合
参数式（仿射精确拟合、mod 分类需每剩余类≥2 例防记忆、合取贪心覆盖），
歧义多候选必须 `rule-select` 人工选择，晋升要独立验证（训练 100% +
held-out≥0.75 + 反例全拒 + 非法输入安全）+ 版本/回滚；重复验证与重复
激活是显式幂等（只刷新报告，不 bump 版本、不改历史）；执行器经
`ApplyLearnedRule`（origin=induced_rule，调用计数）实际调用，支持链式
组合（_prev 传递，轨迹记录）；seed 规则经 `apply_rule(*)` 接入概念流程。
实验：A 双规则组合解未见输入（6→13、9→19）；B 反例否决过宽规则；
C 奇偶迁移新实例；D 未见组合。`学规则` 聊天分步收集示例→候选（不自动晋升）。
能力归属：人工操作（operations/reasoner/mathengine/seed 规则）/用户监督
（示例与选择）/系统归纳（候选+验证报告）/独立验证能力（held-out 通过项）
四者分离记录（experiences outcome 字段）。

## 本地认知工作台（Phase 5A：浏览器 UI，127.0.0.1:8765）

```powershell
py -m pip install -e ".[ui]"
py -m cognitive_ai.cli ui
# 打开 http://127.0.0.1:8765/ ；端口占用会明确报错；只绑本地回环
```

8 页：对话与解题（语义帧/任务/轨迹/验证/答案全展示）/教授知识
（预览→确认，假设永不直写 verified）/规则实验室（示例→反例→验证→
选择→激活→调用计数，全走 rulelearn 真流程）/概念网络（搜索/详情/
版本）/待审队列/自主探索（限步）/认知历史（经历/版本/任务轨迹）/
模型与状态（活动模型/候选/门禁说明，无直接激活通道）。
顶栏显示活动模型、知识版本、服务与最近任务状态。
演示（隔离库，真 HTTP）：提问得答案11→教学紫砂壶→归纳 double→
验证通过激活 v3→历史/任务/探索均可见。测试 14 项全过（含中文多轮、
会话隔离、参数校验、重开持久化）。

## 下一阶段计划

1. 更严格的研究证据门槛：同一结论需两个独立来源才进入知识库。
2. 规则学习：从稳定的属性共现中提议新规则（走提案+回归验证）。
3. 类别拆分/合并的自动触发条件（基于持续的反例计数，而非单次观察）。
4. 神经网络模块（随机初始化，独立 `C` 类学习评估，与规则引擎对比测试）。
5. `ResearchProvider` 的只读本地网页镜像源（仍不伪造结果、无密钥可运行）。
