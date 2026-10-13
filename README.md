# Cognitive AI
> **项目名说明：** “Cognitive AI” 容易让人联想到具备认知、理解、推理或自主学习能力的人工智能系统。本项目目前并不具备这些广义能力；项目名表示探索方向，不是功能承诺。它实际是一个基于 Python 标准库和 SQLite 的符号化概念知识库，以及一个可选的、从随机初始化开始训练的字符级文本分类器。

本项目不使用预训练模型或在线 LLM。当前功能、实测结果和历史清理记录见 [PROGRESS.md](PROGRESS.md)。
## 项目定位

项目由两个相互独立的部分组成：

1. **概念知识库**：保存概念、属性、关系、规则和证据，提供有限的继承、查询、学习和版本管理能力。
2. **可选神经分类器**：对手写的中文短词和短句进行字符级分类。它不是图像模型，也不是中文理解模型；它只通过 `candidate` Claim 与知识库连接，知识库的核心功能不依赖 PyTorch。
知识库中的更新需要区分三种情况：

- **情景记录**：向 `experiences` 表插入一条观察记录。
- **知识结构修改**：改变概念、属性、关系或规则，并生成版本号，可回滚。
- **神经网络参数更新**：只有手动执行 `neural-*` 训练命令才会发生，从随机初始化开始训练。

## 核心能力

### 概念知识库

核心代码位于 `src/cognitive_ai/`，运行时不需要第三方依赖。

- 概念支持多重继承，并进行环检测；属性沿祖先链继承，距离最近的定义优先。
- 规则引擎支持 `is_a`、`has_attr` 和 `relation` 三种条件，并支持例外条件。数据不足时返回 `unknown`，不会用猜测代替缺失信息。
- 概念结构修改先保存快照、执行回归检查，再生成新版本；版本可以回滚。高影响修改由 `SchemaEvolver` 统一处理。
- 属性拥有生命周期：`candidate → hypothesis → supported → verified`，并支持 `contested`、`rejected`、`superseded`。自动流程最多写入 `hypothesis`，更高状态必须显式调用 `promote`。
- 观察可以按属性重合度分类为 `match`、`ambiguous`、`novel` 或 `insufficient`。
- 探索循环会从缺少属性、缺少关系、孤立概念和冲突中生成问题，并受步数和时间预算限制。
- 本地研究流程在 `data/corpus/` 中做关键词匹配，再用正则抽取 `key: value` 或 `key是value`。至少需要两个独立来源给出相同的短值，才会把结果写成 `hypothesis`；只有一个来源时只保留 `candidate` Claim。

这些能力是结构化数据处理和有限规则推理，不等同于开放域知识获取、自然语言理解或通用推理。

### 字符级神经分类器

代码位于 `src/cognitive_ai/neural/`，安装可选的 PyTorch 依赖后使用。

- CNN 结构为字符嵌入、核宽为 2/3/4 的一维卷积、最大池化和两个输出头（5 个一级类别及二级类别）；另有双向 GRU 用于对比。
- 训练数据是手写的中文短词和短句，按语义组划分训练、验证和测试集，并在训练前检查组泄漏。
- 模型从随机初始化开始训练，不使用预训练权重。
- 未知样本可以进入隔离队列，等待人工审核；支持增量训练、旧样本回放、回归门禁、活动模型指针、回滚、OOD 判定和候选类别发现。
- 网络输出只能作为 `candidate` Claim 进入知识库，不会直接修改属性或知识库版本。

当前数据规模较小，CNN 的一级测试表现低于手写关键词基线；因此这部分目前适合用于实验和流程验证，不适合宣称具有可靠的中文分类或理解能力。可复现实测数字见 [PROGRESS.md](PROGRESS.md)。

## 明确的边界与局限

- 自由文本只支持 `X是Y`、`X有Y`、`X可以Z` 三种正则句式，对应 `cli parse`；其他输入需要使用结构化 JSON。
- 系统不理解中文语义。字符级分类器学习的是有限数据集中的字符模式，不是语言模型。
- 系统不进行语义检索。所谓“研究”只是在约 1 KB 的本地文本中做关键词匹配和正则抽取；独立来源判断主要能识别逐字复制和轻度改写。
- 系统不是持续学习系统。知识库更新是数据库插入和状态变化；神经网络只有在手动执行训练命令时才更新。
- `WebSearchProvider` 目前只是接口，调用会直接抛出 `NotImplementedError`；它没有联网能力，也没有现成的在线来源审核方案。
- 评测集和对照实验每组只有 3–12 个样本，结果只能说明代码路径能够运行，不能证明学习带来稳定增益。

## 安装

需要 Python ≥ 3.12。Windows 上将 `python` 替换为 `py`。

```bash
pip install -e ".[test]"          # 核心 + pytest
pip install -e ".[neural]"        # 可选：PyTorch，神经分类器要用
pytest -q
```

默认数据库为 `data/cognitive.db`，首次运行时使用 `data/concepts.json` 初始化。要在隔离环境中运行，可设置：

- `COGNITIVE_AI_DATA`：数据目录，必须包含 `concepts.json` 和 `corpus/`。
- `COGNITIVE_AI_MODELS`：神经网络权重和活动模型指针所在目录。

## 使用方式

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

### 神经网络命令

以下命令需要安装 PyTorch：

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

权重保存在 `models/`，不会提交到仓库。OOD 阈值只使用训练集和验证集确定，测试集只用于报告结果。

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

## 当前已知问题

以下问题不影响基本运行，但会限制系统的泛化能力：

- `learner.default_regression` 只检查“没有环”和“`cup` 仍是 `container` 的后代”，与种子数据绑定较紧，尚不是真正通用的回归测试。
- `curiosity.IMPORTANT_CONCEPTS` 写死了 4 个种子概念 ID，用来增加相关性分。
- `classify` 使用手写评分公式，阈值 `0.3` 尚未通过系统校准。
- 神经网络数据量过小，继续调模型的收益有限，优先级应是扩充训练数据。

## 保留信息：历史清理与暂未实现内容

本栏集中保留在文档重构中不宜丢失的历史信息。2026-10-12 清理时移除了约 1.2 万行代码和 34 个测试文件；旧版本可在 branch `backup/old-str` 以及 tag `first-fix` 中查看。

### 计划中的内容

- `research.WebSearchProvider` 只有接口，调用会抛 `NotImplementedError`。
- 计划将回归门改为基于种子数据和历史快照的通用检查。
- 计划让 `curiosity.IMPORTANT_CONCEPTS` 可配置，或改为从数据计算。
- 计划扩充神经网络训练数据；在数据规模不变时，继续调参缺少实际意义。

## 许可证

MPL-2.0，见 [LICENSE](LICENSE)。