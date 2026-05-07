# Coding Agent

基于 LangChain/LangGraph 的单Agent编码智能体，实现从需求分析到代码交付的全自动化软件工程流水线。

## 项目概述

Coding Agent 是一个**自主单Agent系统**，能够端到端地完成软件开发任务。它接收用户的自然语言需求，经过需求澄清、架构设计、代码生成、质量检测、失败修复、性能评估等六个阶段，最终输出完整的工程代码和交付报告。

**核心能力**：需求分析 → 架构设计 → 代码生成 → 自动测试 → 失败修复 → 交付报告

**框架基础**：[LangChain](https://github.com/langchain-ai/langchain) + [LangGraph](https://github.com/langchain-ai/langgraph) + Pydantic

---

## 目录结构

```
coding_agent/
├── config.py                    # 配置管理（Settings 单例，SecretStr API key）
├── logger.py                    # 日志模块（文件 + 控制台双通道）
├── state.py                     # 状态定义（AgentState Pydantic 模型 + 断点持久化）
├── prompts.py                   # 提示词模板（10 个 ChatPromptTemplate）
├── tool.py                      # 自定义工具集（7 个 LangChain @tool）
├── agent.py                     # 主Agent逻辑（LangGraph StateGraph 状态图）
├── main.py                      # 入口文件（async CLI）
├── requirements.txt             # Python 依赖
├── AGENTS.md                    # Agent 行为规范文档
├── .env / .env.example          # 环境变量配置
├── .gitignore                   # Git 忽略规则（含 .env 排除）
│
├── phases/                      # 阶段模块
│   ├── __init__.py              # 模块导出
│   ├── requirement.py           # 阶段一：需求分析与澄清
│   ├── develop.py               # 阶段二：骨架设计 + 代码填充
│   ├── test.py                  # 阶段三/四：轻量测试 + 整体测试
│   ├── fix.py                   # 阶段五：失败分析与修复
│   └── deliver.py               # 阶段六：指标收集与交付
│
└── workspace/                   # 工作区（运行时生成）
    ├── checkpoints/             # 阶段断点（每阶段自动保存）
    │   ├── checkpoint_latest.json
    │   ├── checkpoint_requirement.json
    │   ├── checkpoint_develop.json
    │   ├── checkpoint_light_test.json
    │   ├── checkpoint_full_test.json
    │   ├── checkpoint_fix.json
    │   └── checkpoint_deliver.json
    ├── logs/agent.log           # 运行日志
    ├── tests/                   # 测试套件
    │   ├── conftest.py
    │   ├── test_config.py       # 配置加载测试
    │   ├── test_state.py        # 状态模型与断点测试
    │   ├── test_tools.py        # 工具集测试
    │   ├── test_phases.py       # 阶段模块测试
    │   └── test_agent.py        # 状态图与路由测试
    └── src/                     # 目标代码生成目录
```

---

## 架构设计

### 整体架构

```
┌──────────┐    ┌─────────────────────────────────────┐
│  main.py │───▶│           LangGraph StateGraph       │
│  (入口)   │    │                                     │
└──────────┘    │  ┌──────────┐   ┌──────────┐        │
                │  │requirement│──▶│ develop  │        │
                │  │   (循环)   │   └────┬─────┘        │
                │  └──────────┘        │              │
                │                      ▼              │
                │                ┌──────────┐         │
                │         ┌──────│light_test│──┐      │
                │         │      └──────────┘  │      │
                │         │ 通过               │ 失败  │
                │         ▼                    ▼      │
                │  ┌──────────┐         ┌──────────┐  │
                │  │full_test │────────▶│   fix    │  │
                │  └────┬─────┘  失败   │ (计数器) │  │
                │       │ 通过          └────┬─────┘  │
                │       ▼                    │        │
                │  ┌──────────┐     ┌───────┘        │
                │  │ deliver  │     │ 回到失败测试    │
                │  └────┬─────┘     │ / END(超限)    │
                │       │           │                │
                │       ▼           │                │
                │      END ◀────────┘                │
                └─────────────────────────────────────┘
```

### 状态管理

`AgentState`（Pydantic BaseModel）包含 16 个字段，贯穿整个工作流：

| 字段 | 类型 | 说明 |
|------|------|------|
| `requirement` | `str` | 原始用户需求 |
| `clarified_requirement` | `str` | 明确后的需求 |
| `tech_plan` | `str` | 技术方案（JSON字符串） |
| `task_template` | `dict` | 结构化任务模板 |
| `project_skeleton` | `dict` | 项目骨架定义 |
| `current_phase` | `str` | 当前阶段标识 |
| `fix_rounds` | `int` | 当前修复轮数 |
| `max_fix_rounds` | `int` | 最大修复轮数 |
| `test_results` | `list` | 测试结果列表 |
| `failure_analysis` | `list` | 失败分析记录 |
| `performance_metrics` | `dict` | 性能指标 |
| `phase_history` | `list` | 阶段流转历史 |
| `logs` | `list` | 操作日志 |
| `should_continue` | `bool` | 是否继续执行 |
| `human_intervention` | `bool` | 是否需要人工介入 |
| `failed_at` | `Optional[str]` | 记录失败的测试阶段 |

### 条件路由

| 路由函数 | 判断逻辑 | 分支 |
|----------|---------|------|
| `route_after_requirement` | `clarified_requirement` 是否有值 | `develop` / `requirement`(循环) |
| `route_after_light_test` | 最近测试是否通过 | `full_test` / `fix` |
| `route_after_full_test` | 全部测试是否通过 | `deliver` / `fix` |
| `route_after_fix` | 是否超限 / `failed_at` 值 | `light_test` / `full_test` / `END` |

---

## 功能清单

### 阶段一：需求分析 (`phases/requirement.py`)
- LLM 分析需求，识别技术选型和功能边界
- 生成结构化任务模板（输入 / 目标文件 / 测试命令 / 验收标准）
- **交互式需求澄清**：CLI 多行输入对话，循环直至需求明确

### 阶段二：开发 (`phases/develop.py`)
- LLM 设计项目骨架（目录结构、配置文件、主入口、测试目录）
- 自动生成 `AGENTS.md`（角色定义、代码风格、提交规范、测试要求）
- 单Agent模式直接填充代码
- 轻量测试（ruff → black → mypy → lint 顺序执行）

### 阶段三/四：测试 (`phases/test.py`)
- **轻量测试**：代码质量门禁（ruff / black / mypy / lint）
- **整体测试**：隔离测试环境 + 单元测试 + 集成测试 + E2E测试 + 覆盖率采集
- **CI/CD 模拟**：按 GitHub Actions 流程依次执行 ruff/black/mypy/pytest

### 阶段五：修复 (`phases/fix.py`)
- 计数器自动递增，超限则标记人工介入
- **5 类失败分析**：
  | 类型 | 说明 | 修复策略 |
  |------|------|---------|
  | `unread_required_file` | 未读取必要文件 | `repair_code` |
  | `wrong_code_modified` | 误改无关代码 | `repair_code` |
  | `test_not_executed` | 测试未执行 | `add_tests` |
  | `over_engineering` | 过度设计 | `simplify` |
  | `incomplete_fix` | 修复不完整 | `complete_fix` |
- LLM 辅助策略选择，自动回到失败测试阶段

### 阶段六：交付 (`phases/deliver.py`)
- **4 项性能指标**：
  - Pass Rate（通过率）
  - First-Pass Rate（一次通过率）
  - Average Fix Rounds（平均修复轮数）
  - Mis-Change Rate（误改率）
- 生成 `DELIVERY_REPORT.json` 交付报告
- CLI 打印摘要，自动结束工作流

---

## 模块亮点

### 1. 断点持久化
每个阶段执行完毕后自动保存 checkpoint 到 `workspace/checkpoints/`：
- `checkpoint_latest.json` — 最新状态（始终覆盖）
- `checkpoint_{phase}.json` — 各阶段独立快照
- 支持从断点恢复：`load_checkpoint(path)` → `AgentState`
- 异常容错：缺失/损坏文件返回 `None`，不中断主流程

### 2. SecretStr 安全存储
`config.py` 使用 Pydantic `SecretStr` 封装 API key：
```python
self.OPENAI_API_KEY: SecretStr = SecretStr(os.getenv("OPENAI_API_KEY", ""))
```
- mypy 类型安全，防止意外泄露
- `.env` 已加入 `.gitignore`，不会提交到仓库

### 3. 双通道日志
`logger.py` 同时输出到：
- **控制台**：实时观察运行状态
- **文件**：`workspace/logs/agent.log` 持久保存
- 格式：`[YYYY-MM-DD HH:MM:SS] [LEVEL] [模块名] 消息内容`
- 通过 `get_logger(__name__)` 工厂获取模块级 logger

### 4. 完整的工具生态
7 个 LangChain `@tool` 装饰器工具，覆盖全流程：
| 工具 | 功能 | 关键特性 |
|------|------|---------|
| `read_file_tool` | 读取文件 | offset / limit 分段读取 |
| `write_file_tool` | 写入文件 | 自动创建父目录 |
| `edit_file_tool` | 精确编辑 | 多重匹配拒绝（要求更多上下文） |
| `execute_command_tool` | 执行命令 | timeout 超时控制 + 工作目录 |
| `run_test_tool` | 运行测试 | 支持 ruff/black/mypy/pytest/all |
| `git_commit_tool` | Git 提交 | 暂存 + 提交一步完成 |
| `log_tool` | 记录日志 | 复用项目 logger 基础设施 |

### 5. DeepSeek V4 集成
默认配置使用 DeepSeek V4 API（`api.deepseek.com`），兼容 OpenAI 接口协议，开箱即用。

### 6. 标准化提示词体系
10 个 `ChatPromptTemplate`，每个都包含：
- `system` 消息（角色定义 + JSON 输出格式约束）
- `human` 消息（动态输入变量）
- 结构化的 JSON 输出协议，配合 `json.loads` 解析和异常回退

---

## 上手与使用

### 前置条件
- Python 3.10+
- pip 包管理器

### 1. 安装依赖
```bash
pip install -r requirements.txt
# 额外安装（测试运行需要）
pip install langgraph pytest pytest-asyncio pytest-cov ruff black mypy
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek（或其他 OpenAI 兼容 API）凭证
```

`.env` 文件内容：
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
WORKSPACE_DIR=./workspace
MAX_FIX_ROUNDS=5
LOG_LEVEL=INFO
```

> **注意**：`.env` 已加入 `.gitignore`，不会被提交到仓库。

### 3. 启动 Agent
```bash
python main.py
```

交互流程：
1. 显示已加载的工具列表
2. 输入多行需求，输入 `done` 结束
3. 如有疑问，Agent 会提问澄清需求
4. 自动执行全部 6 个阶段
5. 打印最终交付报告和性能指标

### 4. 运行测试
```bash
# 运行全部测试（64 个用例）
python -m pytest workspace/tests/ -v

# 带覆盖率
python -m pytest --cov=. --cov-report=term workspace/tests/

# 仅运行特定模块
python -m pytest workspace/tests/test_tools.py -v
```

### 5. 运行质量门禁
```bash
ruff check .           # 代码规范检查
black --check .        # 格式化检查
mypy . --ignore-missing-imports  # 类型检查
```

### 6. 断点恢复
```python
from state import load_checkpoint
from agent import build_graph
import asyncio

state = load_checkpoint("workspace/checkpoints/checkpoint_latest.json")
if state:
    graph = build_graph()
    asyncio.run(graph.ainvoke(state))
```

---

## 工作流详解

```
用户输入
  │
  ▼
┌─────────────────┐
│  需求分析         │ ◀── 循环直至需求明确，交互式 Q&A
│  requirement.py  │
└────────┬────────┘
         │ 需求明确
         ▼
┌─────────────────┐
│  开发             │ 骨架设计 → 代码生成 → AGENTS.md → 轻量测试
│  develop.py      │ (ruff → black → mypy → lint)
└────────┬────────┘
         │
    ┌────▼────┐
    │轻量测试?  │
    └─┬───┬───┘
  通过│   │失败
      │   ▼
      │ ┌─────────────────┐
      │ │  修复             │ 失败分析(5类) → 策略选择 → 执行修复
      │ │  fix.py          │ 计数器++ → 超限? → 人工介入
      │ └────────┬────────┘
      │          │ 回退到失败测试
      ▼          │
┌─────────────────┐
│  整体测试         │ 隔离环境 → 单元 → 集成 → E2E → 覆盖率 → CI/CD
│  test.py         │
└────────┬────────┘
         │
    ┌────▼────┐
    │整体测试?  │
    └─┬───┬───┘
  通过│   │失败 → fix
      ▼
┌─────────────────┐
│  交付             │ 指标收集 → 报告生成 → DELIVERY_REPORT.json
│  deliver.py      │
└────────┬────────┘
         │
         ▼
        END
```

---

## 配置参考

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `OPENAI_API_KEY` | _(必填)_ | API 密钥 |
| `OPENAI_BASE_URL` | _(必填)_ | API 端点地址 |
| `MODEL_NAME` | `gpt-4` | 模型名称 |
| `WORKSPACE_DIR` | `./workspace` | 工作目录 |
| `MAX_FIX_ROUNDS` | `5` | 最大修复轮数 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

---

## 技术栈

| 组件 | 用途 |
|------|------|
| `langchain` + `langchain-openai` | LLM 调用与链式编程 |
| `langgraph` | 状态图构建与条件路由 |
| `pydantic` | 数据模型、SecretStr、序列化 |
| `python-dotenv` | 环境变量管理 |
| `pytest` + `pytest-asyncio` | 测试框架（含异步支持） |
| `ruff` | 代码规范检查 |
| `black` | 代码格式化 |
| `mypy` | 静态类型检查 |

---

## 扩展指南

### 添加新工具
在 `tool.py` 中使用 `@tool` 装饰器定义新工具，并加入 `ALL_TOOLS` 列表：
```python
@tool
def my_new_tool(param: str) -> str:
    """Tool description."""
    # implementation
    return result

ALL_TOOLS = [..., my_new_tool]
```

### 添加新阶段
1. 在 `phases/` 下创建新模块（如 `phases/new_phase.py`）
2. 实现 `async def new_phase(state: AgentState) -> dict`
3. 在 `phases/__init__.py` 中导出
4. 在 `agent.py` 的 `build_graph()` 中注册节点和路由

### 修改提示词
在 `prompts.py` 中修改对应的 `ChatPromptTemplate`，调整 system/human 消息和 JSON 输出格式。

### 切换 LLM 提供商
只需修改 `.env` 中的 `OPENAI_BASE_URL` 和 `MODEL_NAME`，兼容任何 OpenAI 接口协议的 API。

---

## 许可证

MIT License
