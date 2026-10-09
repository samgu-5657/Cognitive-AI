# 神经训练数据（handcrafted-v1，2026-10）

手工编写的小规模中文概念分类数据，**合成种子数据，不是大规模语料**。
每行一个 JSON：`{id, group, text, level1, level2, ambiguous?, unknown?, note?, source}`。

- `level1`: entity / event / action / attribute / abstract（未知样本为 null）。
- `level2`: 各一级下的子类别；不适用记 `none`；未知样本为 null。
- `group`: 语义改写组。**整个组只属于一个切分**（train/val/test 组不相交），
  同义改写不会同时出现在训练集和测试集。
- `ambiguous: true`：有歧义样本（计入评测但单独报告）。
- `unknown: true`：仅 test，用于未知拒绝评测，不计入准确率。

| 切分 | 样本数 | 组数 | 说明 |
|---|---|---|---|
| train | 174 | 58 | 5 类大致均衡 |
| val | 54 | 18 | 与 train 组不相交 |
| test | 77（69 有标 + 8 未知） | 23 + 8 未知组 | 含近义反例/双关/未知 |

限制：词汇量小（数百汉字）、句式短、分布高度人工化。
结论仅适用于"小规模受控中文概念分类"，不得外推为通用中文理解。
