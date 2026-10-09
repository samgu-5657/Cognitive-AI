# 语义解析数据（handcrafted-sem1）

组合式规则解析器 + 微型意图网络的训练/评测数据。句式覆盖 9 意图
（classify/attribute/function/negate/hedge/correct/causal/observe/
unknown）及 ref 指代句（needs_context，仅单独评测指代消解）。

| 切分 | 样本数 | 组数 |
|---|---|---|
| train | 74 | 37 |
| val | 32 | 16 |
| test | 47 | 23 + 3 歧义/未知 + 2 指代 |

组级隔离（改写不出组），`check_group_isolation` 在训练前强制检查。
意图网只学意图标签（slots/实体/模态一律由规则负责）；unknown 意图
无训练样本，网络恒不能输出 unknown（规则负责弃权），已如实报告。
