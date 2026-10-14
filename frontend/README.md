# Cognitive AI · Web 前端

本地 Cognitive AI Web API 的对话式前端，技术栈为 Bun + React（纯 JS，不使用 TypeScript）+ [Beer CSS](https://www.beercss.com/)（Material Design 3）。

前端调用 `src/cognitive_ai/web_ui.py` 中已有的 Python API，用来替代旧的单页 `web_ui_static/index.html`。Python 代码没有改动。

## 运行

最简单的方式是在仓库根目录执行：

```bash
./run.sh              # 同时启动后端（8765）和前端（5173），Ctrl+C 一起停止
MODE=prod ./run.sh    # 生产模式：只打包一次，不热重载
```

`run.sh` 需要 `python3`（≥ 3.12）和 `bun`；首次运行会自动执行 `bun install`。它通过 `PYTHONPATH` 加载 `src/`，不要求先 `pip install`。可用环境变量：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `BACKEND_HOST` / `BACKEND_PORT` | `127.0.0.1` / `8765` | 后端地址 |
| `FRONTEND_HOST` / `FRONTEND_PORT` | `127.0.0.1` / `5173` | 前端地址 |
| `PYTHON` | `python3` | Python 解释器 |
| `MODE` | `dev` | `dev`（热重载）或 `prod` |
| `COGNITIVE_AI_DATA` | `data/` | 后端数据目录（沿用项目原有变量） |

也可以分别手动启动：

```bash
python -m cognitive_ai.cli ui                 # 后端，127.0.0.1:8765
cd frontend && bun install && bun run dev     # 前端，127.0.0.1:5173
```

浏览器打开 http://127.0.0.1:5173 。Bun 服务器负责提供页面，并把 `/api/*` 转发给后端（地址由 `COGNITIVE_AI_API` 指定，默认 `http://127.0.0.1:8765`）。

其他命令：

- `bun run build`：把静态文件打包到 `dist/`。可以用任意静态服务器托管，但需要把 `/api/*` 转发到后端。
- `bun test`：运行 API 客户端、对话状态和响应整理函数的单元测试。

## 功能

旧页面的全部功能都保留了下来，改为对话形式：

| 旧页面 | 新界面 |
|---|---|
| 知识问答（`/api/ask`） | **问答**模式：显示回答、结论标签，以及可展开的「查证过程」（主语 / 宾语概念、查询、证据）。 |
| 语义分析（`/api/analyze`） | **分析**模式：把句子显示为可折叠的树，标出并列 / 条件结构（前件、后件），每个节点都有状态、原子和证据。 |
| 上下文实验（`/api/interact`、`/api/answer`） | **上下文**模式：先前陈述以可编辑的标签显示在输入框上方；指代有歧义时直接在回复里选择选项，也可以取消。指代解析结果来自 `/api/context`。 |
| 提出知识候选 / 教学（`/api/propose`） | **教学**模式：在对话里直接显示候选卡片，带「确认写入 / 拒绝」按钮。 |
| 候选列表（`/api/pending`、`/api/confirm`、`/api/reject`） | **知识审核**页：按状态筛选、搜索、批量确认 / 拒绝、提交新候选。在这里做的决定会同步到对应的对话消息。 |
| JSON 诊断 | 每条回答都有 `{}` 按钮，可查看并复制原始响应。 |

相比旧页面新增的内容：

- 侧边栏里的对话历史：可重命名、删除后可撤销、可搜索。历史只保存在本浏览器的 `localStorage` 中，不会发送给后端。
- 每个对话有自己的上下文。开启「自动加入」后，理解成功的句子会以替换掉代词后的形式加入上下文，追问时可以继续用"它"。
- 每轮对话都可以重试、编辑后重发、换一种模式运行。
- 侧边栏显示后端状态（概念数 / 规则数 / 版本）、待审核数量角标和离线提示。
- 浅色 / 深色 / 跟随系统主题，以及根据种子色生成的 Material 3 动态配色。
- 快捷键：`Enter` 发送，`Shift+Enter` 换行，`Alt+1…4` 切换模式，`Ctrl+Shift+O` 新对话，`/` 聚焦输入框。中文输入法选字时按 Enter 不会发送。
- 响应式布局：桌面端侧边栏可收起成窄图标栏，手机上变为滑出抽屉。

只有审核按钮（确认 / 拒绝）会写入知识库，调用的也是旧页面使用的同一组后端接口；其余操作都是只读的。

## 已知情况

- `GET /api/pending` 会忽略状态筛选参数，所以审核页先加载全部候选，再在浏览器中筛选。
- 上下文模式每条消息会发两个只读请求：`/api/interact` 判断是否需要澄清，`/api/context` 提供要显示的指代解析结果。
- 确认 / 拒绝失败时，后端可能返回 HTTP 200 加 `{ok:false, reason}`（例如候选已经处理过）。前端把 `reason` 当作错误显示。
- JS 包压缩后约 0.56 MB；Beer CSS 自带三套 Material Symbols 图标字体，每套约 0.5 MB。

## 目录结构

```
frontend/
  server.js                 Bun 服务器：页面打包 + /api 转发
  src/index.html            入口
  src/main.jsx              引入 Beer CSS、material-dynamic-colors 和样式
  src/App.jsx               状态管理、API 调用、快捷键
  src/components/           Sidebar、Thread、Composer、Welcome、ReviewPage、SettingsDialog、results、common
  src/lib/api.js            11 个接口的 fetch 客户端
  src/lib/conversations.js  对话 reducer 与 localStorage 持久化
  src/lib/summarize.js      把响应整理成界面数据的纯函数
  src/lib/labels.js         后端枚举值的中文标签和配色
  src/lib/prefs.js          主题 / 模式偏好
  tests/lib.test.js         bun test
```
