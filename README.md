# Cognitive AI

一个用 Python 标准库 + SQLite 写的**符号化概念知识库**，外加一个可选的、从随机初始化开始训练的
**字符级 CNN 文本分类器**。不使用任何预训练模型或在线 LLM。

项目名里的 "Cognitive" 只是一个目标方向，不代表现有代码具备认知、理解或推理能力。
进度、实测数字和已移除的部分见 [PROGRESS.md](PROGRESS.md)。

## 它实际是什么

**1. 概念知识库（`src/cognitive_ai/`，无第三方依赖）**

- 概念存在 SQLite 里，支持多重继承（带环检测），属性沿祖先链继承，离得最近的定义优先。
- 一个很小的前向规则引擎：条件只有 `is_a` / `has_attr` / `relation` 三种，支持例外条件；
  缺数据时返回 `unknown`，不会去猜。
- 结构修改（加概念、改父类、拆分/合并类别等）都会先存快照、跑回归检查，再生成新版本号，
  之后可以回滚到任意一个版本。
- 每条属性带状态：`candidate → hypothesis → supported → verified`，另有
  `contested / rejected / superseded`。自动流程最高只能写到 `hypothesis`，
  再往上必须显式调用 `promote`。
- "自主探索"：根据知识缺口（缺属性、没关系、孤立概念、冲突）生成问题，然后在本地 `data/corpus/`
  的 3 个小文本文件里做关键词检索，用正则抽出 `key: value` / `key是value`。
  必须有 ≥2 个独立来源（按内容哈希 + 摘录相似度去重）给出同一个短值，才会以 `hypothesis` 写入。
  每次运行都有步数和时间上限。
- 按属性重合度给结构化观察打分，结论是 `match / ambiguous / novel / insufficient` 之一。

**2. 神经分类器（`src/cognitive_ai/neural/`，需要 PyTorch）**

- 字符嵌入 → 1-D 卷积（核宽 2/3/4）→ max-pool → 两个输出头（5 个一级类别 + 二级类别）。
  另附一个双向 GRU 用来对比。
- 训练数据是手写的中文短词/短句：train 174 / val 54 / test 77（`data/neural/`），
  按语义组划分，训练前会检查有没有组泄漏。
- 测试集上一级准确率 0.42，**比手写关键词基线（0.435）还低**。详见 PROGRESS.md。
- 外围工具：未知样本隔离队列（模型预测不会自动变成标签）、人工审核、增量训练（新样本 + 旧样本回放）、
  回归门禁（指针切换，不覆盖旧权重，可回滚）、OOD 判定、候选类别发现。
- 网络输出只会以 `candidate` Claim 的形式进库，不会直接改属性或版本。

注意：这个 CNN 处理的是汉字序列（文本分类），不是图像。它和知识库之间只通过 `candidate` Claim 连接，
知识库的任何功能都不依赖它。

## 它不是什么

- 它不懂中文。自由文本输入只支持 `X是Y` / `X有Y` / `X可以Z` 三个正则（`cli parse`），
  其余输入都需要写成结构化 JSON。
- 它不做语义检索。"研究"就是在约 1 KB 的本地文本里做关键词匹配加正则抽取；"独立来源"
  判断只能识别逐字复制和轻度改写。
- 它不是在持续学习。知识库的更新就是往数据库插行、改状态；神经网络只有在你手动跑训练命令时才会更新。
- `WebSearchProvider` 只是个接口，调用时直接抛 `NotImplementedError`。

## 安装与运行

需要 Python ≥ 3.12。Windows 上把 `python` 换成 `py`。

```bash
pip install -e ".[test]"          # 核心 + pytest
pip install -e ".[neural]"        # 可选：PyTorch，神经分类器要用
pytest -q
```

默认数据库是 `data/cognitive.db`，第一次运行时用 `data/concepts.json` 初始化。
想在隔离环境里跑，可以设置 `COGNITIVE_AI_DATA`（目录里要有 `concepts.json` 和 `corpus/`）
和 `COGNITIVE_AI_MODELS`。

### 知识库命令

```bash
python -m cognitive_ai.cli doctor                 # 健康检查（环检测 + 回归门）
python -m cognitive_ai.cli concepts               # 列出概念
python -m cognitive_ai.cli inspect cup            # 概念详情（含继承属性）
python -m cognitive_ai.cli query --json '{"type":"get_attribute","concept":"cup","key":"function"}'
python -m cognitive_ai.cli observe --json-file obs.json   # 结构化观察 → 学习
python -m cognitive_ai.cli parse 杯子是容器        # 三种句式的正则解析
python -m cognitive_ai.cli gaps                   # 知识缺口
python -m cognitive_ai.cli questions              # 生成并打分研究问题
python -m cognitive_ai.cli explore --steps 5      # 有预算的自主探索
python -m cognitive_ai.cli consolidate            # 尝试按稳定的属性分歧拆分类别（走回归门）
python -m cognitive_ai.cli history                # 版本与结构修改记录
python -m cognitive_ai.cli rollback 1             # 回滚到某个版本
python -m cognitive_ai.cli lifecycle --describe   # 生命周期规则
python -m cognitive_ai.cli promote <attr_id> supported --evidence '[...]'
python -m cognitive_ai.cli experiences --limit 20
```

`observe` 的输入格式：

```json
{"name": "teapot", "parents": ["vessel"],
 "attributes": {"material": "clay", "function": "brewing"}}
```

### 神经网络命令（需要 torch）

```bash
python -m cognitive_ai.cli neural-train --model cnn --tag cnn-v1 --seed 7 --epochs 300
python -m cognitive_ai.cli neural-eval --tag cnn-v1
python -m cognitive_ai.cli neural-compare --tags cnn-v1,gru-v1
python -m cognitive_ai.cli neural-predict 杯子 --tag cnn-v1 [--submit]
python -m cognitive_ai.cli neural-ood-build --tag cnn-v1     # 原型 + 阈值（只用 train/val）
python -m cognitive_ai.cli neural-ood 跑步机 --tag cnn-v1
python -m cognitive_ai.cli neural-cycle 激光打印机            # 预测；低置信进隔离队列
python -m cognitive_ai.cli neural-queue --status pending
python -m cognitive_ai.cli neural-review <id> --l1 entity --l2 electronics --accept
python -m cognitive_ai.cli neural-incremental --base cnn-v1 --out cnn-cand1
python -m cognitive_ai.cli neural-queue-export --out reviewed.jsonl
python -m cognitive_ai.cli neural-gate --base cnn-v1 --candidate cnn-cand1 --new reviewed.jsonl --promote
python -m cognitive_ai.cli neural-active
python -m cognitive_ai.cli neural-rollback
python -m cognitive_ai.cli neural-discover --tag cnn-v1
```

权重保存在 `models/`，不提交到仓库。

## 代码结构

| 模块 | 作用 |
|---|---|
| `models.py` | 数据类，不含 IO |
| `memory.py` | 唯一的 SQLite 访问层，含增量迁移 |
| `ontology.py` | 概念增删改查、多重继承、环检测、属性继承 |
| `reasoner.py` | `is_a` / 属性 / 关系查询，前向规则，冲突检查 |
| `novelty.py` | 观察的新颖性检查、知识缺口、三句式正则解析 |
| `schema_evolver.py` | 唯一能改概念结构的模块：提案 → 回归 → 版本 / 回滚 |
| `curiosity.py` | 从缺口生成问题，用加权公式打分 |
| `research.py` | `LocalCorpusProvider`（关键词 + 内容哈希 + 相关性分级） |
| `lifecycle.py` | 知识状态机与晋升条件 |
| `learner.py` | 观察 → 研究 → 写库；`classify`；`consolidate` |
| `loop.py` | 带步数 / 时间预算的探索循环 |
| `policy.py` | 所有阈值集中在这里 |
| `cli.py` | 命令行入口，不放业务逻辑 |
| `neural/` | 词表、数据集、模型、训练、评测、适配器、队列、门禁、OOD、类别发现 |

## 已知问题

- 回归门 `learner.default_regression` 只检查"没有环"和"`cup` 仍是 `container` 的后代"，
  和种子数据绑得很死，算不上真正的回归测试。
- `curiosity.IMPORTANT_CONCEPTS` 写死了 4 个种子概念 id，用来加相关性分。
- `classify` 的打分是手写公式，阈值 0.3 是拍出来的。
- 评测集和对照实验每组只有 3–12 个样本，只能当冒烟测试看，不能作为结论。

## 许可证

MPL-2.0，见 [LICENSE](LICENSE)。
