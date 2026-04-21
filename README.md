# Care Chat

一个面向癌症患者与照护者的医疗陪伴 Agent 骨架，基于 `openai-agents-python` 构建，使用 `uv` 管理依赖，使用本地 `.env` 管理模型和 API Endpoint。

当前仓库默认按 `Kimi-K2.5 + OpenAI-compatible endpoint + chat_completions` 的方式配置，更适合你当前的网关环境。默认模型 alias 改成了 `moonshot/kimi-k2.5`，避免你当前网关里 `moonshotai/...` 那条更容易触发 provider 侧限流。

当前脚手架包含：

- `OpenAI Agents SDK` 会话编排
- 面向 `患者 / 患者家属 / 社区志愿者` 的角色路由 Agent
- 支持 `MCP` server 的共享挂载与 Agent 专用挂载
- 本地 `.env` 配置加载
- SQLite 会话记忆
- 本地关键词兜底 + 可选 LLM guardrail
- 面向肿瘤陪伴场景的工具与提示词
- 基于 `Rich` 的命令行聊天入口
- 可选流式输出与 Thinking 展示
- 接入官方 `OpenAI Agents SDK tracing`

## 快速开始

1. 复制环境变量模板

```bash
cp .env.example .env
```

2. 编辑 `.env`

- 填入 `OPENAI_API_KEY`
- 如果你接的是兼容端点，填入 `OPENAI_BASE_URL`
- 如果兼容端点使用自定义模型名，覆盖 `CARE_CHAT_MODEL`
- 对 Kimi 兼容网关，优先使用 `CARE_CHAT_OPENAI_API=chat_completions`
- 如果想在 CLI 里展示 Kimi 的思考流，保持 `CARE_CHAT_ENABLE_THINKING=true`
- 如果你要启用 MCP，设置 `CARE_CHAT_MCP_ENABLED=true`，并指定 `CARE_CHAT_MCP_CONFIG_PATH`
- 如果你要把 traces 发到 OpenAI 官方 Traces dashboard：
  - 官方 OpenAI 直连：默认可直接复用 `OPENAI_API_KEY`
  - 第三方兼容网关：额外设置 `CARE_CHAT_TRACING_API_KEY` 为官方 OpenAI API key

3. 安装依赖

```bash
uv sync
```

4. 查看当前配置

```bash
uv run care-chat --show-config
```

5. 启动交互式会话

```bash
uv run care-chat --session-id patient-demo
```

6. 如果希望边生成边输出

```bash
uv run care-chat --session-id patient-demo --stream
```

7. 如果希望展示 Kimi Thinking

```bash
uv run care-chat --session-id patient-demo --stream --show-reasoning
```

8. 如果希望在 CLI 里看到 handoff / tool calling 日志

```bash
uv run care-chat --session-id patient-demo --stream --show-tools
```

9. 或者单轮调用

```bash
uv run care-chat "我今天很害怕，明天要做化疗。"
```

10. 显式指定当前对象角色

```bash
uv run care-chat --session-id family-demo --role-hint caregiver
uv run care-chat --session-id volunteer-demo --role-hint volunteer
```

11. 清空某个会话的历史

```bash
uv run care-chat --session-id patient-demo --clear-session
```

## 会话上下文

这个项目已经是会话式的：

- 同一个 `--session-id` 会复用同一个 `SQLiteSession`
- 会话内容默认保存在 `.local/care_chat_sessions.sqlite3`
- 即使你退出 CLI，只要下次继续使用同一个 `session-id`，上下文仍会保留

例如：

```bash
uv run care-chat --session-id patient-001
uv run care-chat --session-id patient-001 "继续刚才的话题"
```

如果你想限制每轮带回模型的历史条数，可以在 `.env` 中设置 `CARE_CHAT_SESSION_HISTORY_LIMIT`。

如果同一台机器上同时服务患者、家属和志愿者，建议分别使用不同的 `session-id`，避免不同角色的上下文混在一起。

## 流式说明

CLI 提供 `--stream`，会优先尝试 Agents SDK 的 `Runner.run_streamed()`；对 Kimi 还可以叠加 `--show-reasoning` 查看 thinking 文本，也可以用 `--show-tools` 查看 handoff / tool calling 事件。

- 如果当前 provider 对 SDK 流式事件兼容良好，会实时输出增量文本
- 如果当前网关对 Agents SDK 流式兼容性一般，CLI 会自动回退到普通回答，而不是直接中断

对你当前的 Kimi 网关环境，`chat_completions` 明显比 `responses` 更稳定。

如果你看到的是 `429 rate limit`，那是 provider/gateway 的限流，不是 Agents SDK 自己的超时。为了减小这种风险，当前默认运行模式已经收敛为：

- 单 Agent
- 默认关闭 LLM input/output guardrail
- 默认开启轻量级 audience router
- 默认关闭 legacy specialist handoff
- 默认禁止并行 tool calls

## 官方 Tracing

这个项目现在不再内置自定义调试面板，而是直接使用 `OpenAI Agents SDK` 官方 tracing。

官方 tracing 会记录：

- 整个 `Runner.run / run_sync / run_streamed`
- agent run
- LLM generation
- tool call 和 tool output
- handoff
- guardrail

同一个 `session-id` 的多轮对话会被写进同一个 `group_id`，方便你在 OpenAI Traces dashboard 里按会话串起来看。

对当前项目，推荐这样理解：

- 如果你走的是官方 OpenAI 模型，直接开启 tracing 即可
- 如果你走的是第三方 OpenAI-compatible 网关，模型请求仍然可以走第三方，但 tracing 导出最好单独配置 `CARE_CHAT_TRACING_API_KEY`

示例：

```env
CARE_CHAT_TRACING_DISABLED=false
CARE_CHAT_TRACING_API_KEY=sk-your-official-openai-key
CARE_CHAT_TRACE_INCLUDE_SENSITIVE_DATA=false
```

说明：

- `CARE_CHAT_TRACING_DISABLED=false`
  打开官方 tracing
- `CARE_CHAT_TRACING_API_KEY`
  给第三方网关场景单独指定 OpenAI 官方 trace export key
- `CARE_CHAT_TRACE_INCLUDE_SENSITIVE_DATA=false`
  默认不把模型输入输出全文发到 tracing，比较适合医疗陪伴场景

如果你希望 traces 尽快出现在 dashboard，本项目会在每次运行结束后主动 `flush_traces()`。

## 角色路由

当前默认会先经过一个很小的路由 Agent，再 handoff 到三个对象化 specialist 之一：

- `Patient Companion`
  面向患者本人，偏重情绪支持、症状记录、就诊准备和保守升级建议
- `Caregiver Support Coach`
  面向患者家属或照护者，偏重家庭协调、观察记录、与医生沟通、向外求助
- `Community Volunteer Guide`
  面向社区志愿者或非家属帮助者，偏重边界、隐私、接送/跑腿/陪伴和异常升级

这样做的原因很直接：同一个“化疗后吃不下怎么办”的问题，患者、家属和志愿者真正需要的工具、边界和措辞都不一样。

你也可以通过 `.env` 控制：

```env
CARE_CHAT_DEFAULT_ROLE_HINT=auto
CARE_CHAT_ENABLE_ROLE_ROUTER=true
```

如果某个入口已经知道用户角色，比如患者端、小程序家属端、志愿者工作台，可以直接传入 `role_hint`，避免路由 Agent 再猜一遍。

如果你单独给 router 指定了 `CARE_CHAT_ROLE_ROUTER_MODEL`，要优先选一个 handoff/tool calling 比较稳定的模型。否则常见现象是：router 文本里说“我来转接”，但没有真的发出 handoff tool call。

## MCP 集成

这个项目现在支持把 MCP server 作为 `Agent(..., mcp_servers=[...])` 注入到不同层级的 Agent 里。

- 共用 MCP：把一个 server 的 `targets` 写成多个角色组，例如 `["patient", "caregiver", "volunteer"]`
- 专用 MCP：把 `targets` 写成单个 Agent key，例如 `["patient_navigation"]`
- 如果同一个物理 MCP endpoint 需要给不同 Agent 暴露不同工具子集，直接在配置里声明成多个不同的逻辑 server，并分别设置不同的 `name`、`targets`、`allowed_tool_names`

启用方式：

```env
CARE_CHAT_MCP_ENABLED=true
CARE_CHAT_MCP_CONFIG_PATH=docs/mcp.servers.example.json
CARE_CHAT_MCP_STRICT=false
CARE_CHAT_MCP_CONNECT_IN_PARALLEL=true
```

示例配置文件见 [docs/mcp.servers.example.json](/Users/liueic/Documents/Code/Care-Chat/docs/mcp.servers.example.json:1)。

`targets` 支持两类值：

- 角色组：`all`、`patient`、`caregiver`、`volunteer`、`coordinators`、`specialists`
- 具体 Agent：`router`、`patient_coordinator`、`patient_emotional`、`patient_navigation`、`patient_urgent`、`caregiver_coordinator`、`caregiver_emotional`、`caregiver_coordination`、`caregiver_urgent`、`volunteer_coordinator`、`volunteer_task`、`volunteer_boundary`、`volunteer_escalation`

实现上有两个关键点：

- service 首次运行前会统一连接 MCP server；如果 `CARE_CHAT_MCP_STRICT=false`，连接失败的 server 会被自动剔除，其他 server 继续可用
- Agent 只会拿到“当前成功连接”的 MCP server，因此不会把失联的 server 继续暴露给模型

## 项目结构

```text
src/care_chat/
  agents.py      # Agent 组装、handoff、guardrail、OpenAI runtime 配置
  cli.py         # 命令行入口
  config.py      # .env / settings
  mcp.py         # MCP 配置解析、shared/dedicated 挂载与 lifecycle manager
  prompts.py     # 陪伴场景提示词
  safety.py      # 本地紧急风险兜底
  service.py     # 会话与运行封装
  tools.py       # 肿瘤陪伴相关工具
```

## 安全边界

这个脚手架是“医疗陪伴”而不是“在线诊断”：

- 不做诊断
- 不给个体化用药调整建议
- 有紧急风险时优先引导线下求助

本地兜底会直接拦截明显的自伤风险和急症关键词；模型侧还增加了一层输入/输出 guardrail。

## 后续扩展建议

- 接入医院知识库或患者教育资料
- 增加结构化用户画像与长期记忆
- 加入量表、症状打分和随访工作流
- 为患者端、家属端、志愿者端继续拆分 UI 和工作流

## 参考文档

- OpenAI Agents SDK: https://openai.github.io/openai-agents-python/zh/
- OpenAI Agents SDK tracing: https://openai.github.io/openai-agents-python/tracing/
- OpenAI integrations and observability: https://developers.openai.com/api/docs/guides/agents/integrations-observability#tracing
- OpenAI agent building guidance: https://developers.openai.com/tracks/building-agents
