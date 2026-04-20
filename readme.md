# Home Auto Agent - 快速上手指南

## 项目简介

Home Auto Agent 是一个基于 **LangGraph** 的智能家居控制系统，支持通过自然语言与智能设备交互。核心能力包括：

- 自然语言理解与设备控制（开灯、调空调温度等）
- 基于向量数据库的语义化设备检索
- 联网搜索补充上下文信息
- 智能场景规划（起床模式、离家模式等）
- 多 LLM 供应商支持（OpenAI / Anthropic / Groq / DeepSeek / 通义千问）

## 项目结构

```
home-auto-agent/
├── HomeBuddyAgent/          # 主智能助手模块（核心入口）
│   ├── agent.py             # 主工作流图定义
│   └── utils/
│       ├── nodes.py         # 核心节点逻辑（filter、agent、generate 等）
│       ├── tools.py         # 设备检索工具（向量搜索 + Redis 缓存）
│       ├── state.py         # 状态定义
│       ├── prompts.py       # Prompt 模板
│       └── structs.py       # 数据模型（路由评分、清晰度评分等）
│
├── basic_executor/          # 设备执行模块
│   ├── agent.py             # 执行器工作流
│   └── utils/
│       ├── nodes.py         # 执行节点（generate → device_call 循环）
│       ├── tools.py         # 执行工具
│       ├── prompts.py       # 执行 Prompt
│       └── state.py         # 执行器状态
│
├── deep_planner_v1/         # 智能场景规划模块
│   ├── agent.py             # 规划工作流
│   └── utils/
│       ├── nodes.py         # 规划节点（生成场景 → 联网搜索 → 设计方案）
│       ├── states.py        # 规划状态
│       ├── structs.py       # 场景数据模型
│       ├── prompts.py       # 规划 Prompt
│       └── redis_cache.py   # Redis 缓存
│
├── common/                  # 公共模块
│   ├── configuration.py     # 全局配置（LLM 供应商、模型选择等）
│   ├── structs.py           # 核心数据结构（Device、DeviceCall、DeviceResult）
│   ├── common_utils.py      # 工具函数（get_model、rag_loader、tavily_search）
│   └── VectorStore/         # Chroma 向量数据库（本地持久化存储）
│
├── oneNetConfig.json        # 设备配置文件（模拟的 OneNET 设备列表）
├── langgraph.json           # LangGraph 入口配置
├── requirements.txt         # Python 依赖
├── .env.example             # 环境变量模板
└── readme.md                # 基础说明
```

## 环境要求

- **Python** 3.11+
- **Redis**（用于设备检索缓存，需本地或远程部署）
- **API Keys**：至少需要 Tavily 的 API Key，以及至少一个 LLM 提供商的 API Key（如 OpenAI、Anthropic、Groq、DeepSeek 或通义千问）

## 快速启动

### 1. 克隆项目

```bash
git clone <仓库地址>
cd home-auto-agent
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

安装 LangGraph CLI（仅本地开发需要）：

```bash
pip install -U "langgraph-cli[inmem]"
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env.dev`，并填入你的 API Key：

```bash
cp .env.example .env.dev
```

编辑 `.env.dev`：

```dotenv
# 可选 - OpenAI 用于设备控制的 tool call 和结构化输出
OPENAI_API_KEY=sk-xxx

# 必填 - Tavily 用于联网搜索
TAVILY_API_KEY=tvly-xxx

# 可选 - Anthropic 用于场景规划
ANTHROPIC_API_KEY=sk-ant-xxx

# 可选 - 其他 LLM 供应商
GROQ_API_KEY=gsk_xxx
DEEPSEEK_API_KEY=xxx
DEEPSEEK_API_BASE=https://api.siliconflow.cn/v1

# 可选 - 通义千问（可替代 OpenAI）
DASHSCOPE_API_KEY=xxx
DASHSCOPE_BASE=xxx

# Redis 密码（如果 Redis 设置了密码）
REDIS_PASSWORD=xxx

# 向量数据库存储路径（一般不需要修改）
VECTOR_STORE_PATH=common/VectorStore
```

> **注意**：最低配置只需 `TAVILY_API_KEY` 和至少一个 LLM 提供商的 API Key（如 OpenAI、Anthropic、Groq、DeepSeek 或通义千问）即可运行基本功能。

### 5. 初始化向量数据库

首次运行需要将设备配置导入向量数据库。以下是使用不同嵌入模型的示例：

**使用 OpenAI 嵌入模型：**

```python
# init_vector_store.py 或在 Python/Jupyter 中执行
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import json, os

# 初始化 Embeddings（需要 OPENAI_API_KEY）
embeddings = OpenAIEmbeddings()

# 创建向量数据库
vector_store = Chroma(
    collection_name="vector_collection_for_agent",
    embedding_function=embeddings,
    persist_directory=os.path.join(os.getcwd(), "common", "VectorStore"),
    collection_metadata={"vs_name": "test"}
)

# 加载设备配置
with open("oneNetConfig.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [
    Document(page_content=json.dumps(device, indent=2, ensure_ascii=False))
    for device in data
]

ids = vector_store.add_documents(documents)
print(f"成功导入 {len(ids)} 个设备配置")
```

**使用 HuggingFace 开源嵌入模型（无需 API Key）：**

```python
# init_vector_store.py 或在 Python/Jupyter 中执行
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json, os

# 初始化 Embeddings（无需 API Key）
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 创建向量数据库
vector_store = Chroma(
    collection_name="vector_collection_for_agent",
    embedding_function=embeddings,
    persist_directory=os.path.join(os.getcwd(), "common", "VectorStore"),
    collection_metadata={"vs_name": "test"}
)

# 加载设备配置
with open("oneNetConfig.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [
    Document(page_content=json.dumps(device, indent=2, ensure_ascii=False))
    for device in data
]

ids = vector_store.add_documents(documents)
print(f"成功导入 {len(ids)} 个设备配置")
```

### 6. 启动开发服务器

```bash
langgraph dev
```

启动后访问 **LangGraph Studio**（默认 http://localhost:8000）即可通过可视化界面与智能家居助手交互。

## 系统架构

### 核心工作流

```
用户输入（自然语言）
    │
    ▼
  filter ─── 提取问题、初始化状态
    │
    ▼
  agent ──── LLM 判断：是否需要检索设备？
    │
    ├── 需要检索 ──► retriever（向量语义搜索）
    │                    │
    │                    ▼
    │              command_router（评估指令清晰度）
    │                    │
    │         ┌──────────┴──────────┐
    │         ▼                     ▼
    │    指令清晰              指令模糊
    │    executor             info_graph（联网搜索补充）
    │    （执行设备控制）           │
    │         │                    ▼
    │         ▼               generate（生成回复）
    │       返回结果                │
    │                              ▼
    │                           返回回复
    │
    └── 不需要检索 ──► 直接结束
```

### 三个 Agent 模块

| 模块 | 功能 | 对应 Graph |
|------|------|-----------|
| **HomeBuddyAgent** | 主控 Agent，负责意图理解和路由 | `HomeBuddyAgent/agent.py:graph` |
| **basic_executor** | 设备执行器，将指令转化为设备调用 | 作为子图被 HomeBuddyAgent 调用 |
| **deep_planner_v1** | 场景规划器，生成一日生活方案 | `deep_planner_v1/agent.py:graph` |

## 配置说明

### LLM 供应商切换

在 `common/configuration.py` 中可配置各功能使用的 LLM：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `tool_call_provider` | OpenAI | 设备检索的 tool call，可替换为 qwen、anthropic、groq 或 deepseek |
| `tool_call_model` | gpt-4o-mini | tool call 使用的模型 |
| `structured_output_provider` | OpenAI | 结构化输出（设备指令生成），可替换为 qwen、anthropic、groq 或 deepseek |
| `structured_output_model` | gpt-4o-mini | 结构化输出模型 |
| `planner_provider` | Anthropic | 场景规划，可替换为 openai、qwen、groq 或 deepseek |
| `planner_model` | claude-3-7-sonnet-latest | 规划模型 |
| `search_api` | Tavily | 联网搜索 API |
| `number_of_queries` | 2 | 每次联网搜索生成的查询数 |

也可以通过 LangGraph Studio 的 UI 在运行时动态调整这些配置。

**示例配置**：使用通义千问替代 OpenAI

```python
# 在 common/configuration.py 中修改默认值
class Configuration:
    # ...
    writer_provider: WriterProvider = WriterProvider.QWEN
    writer_model: str = "qwen3-72b-instruct"
    structured_output_provider: StructuredOutputProvider = StructuredOutputProvider.QWEN
    structured_output_model: str = "qwen3-72b-instruct"
    tool_call_provider: ToolCallProvider = ToolCallProvider.QWEN
    tool_call_model: str = "qwen3-72b-instruct"
    # ...
```

### 设备配置

设备信息在 `oneNetConfig.json` 中定义。每个设备包含：

```json
{
  "product_id": { "type": "string", "value": "设备产品ID" },
  "device_name": { "type": "string", "value": "设备名称" },
  "device_type": "设备类型（如 Air Conditioner）",
  "params": {
    "type": "object",
    "properties": {
      "参数名": {
        "type": "类型",
        "description": "参数描述",
        "value_range": "取值范围"
      }
    }
  }
}
```

新增设备时，编辑此文件并重新运行**步骤 5**初始化向量数据库。

## 常见问题

### Q: Redis 连接失败怎么办？

确保本地 Redis 服务已启动。Windows 用户可以使用 [Memurai](https://www.memurai.com/) 或 WSL 中安装 Redis。如果 Redis 设置了密码，确保 `.env.dev` 中的 `REDIS_PASSWORD` 正确配置。

### Q: 向量数据库初始化报错？

如果使用 OpenAI 嵌入模型，请确认 `OPENAI_API_KEY` 已正确配置且网络可以访问 OpenAI API（可能需要代理）。

如果不想使用 OpenAI，可以使用 HuggingFace 开源嵌入模型，无需 API Key。

### Q: 如何添加新的智能设备？

1. 在 `oneNetConfig.json` 中添加设备的 JSON 配置
2. 重新运行向量数据库初始化脚本
3. 重启 `langgraph dev`

### Q: 如何切换 LLM 供应商？

修改 `common/configuration.py` 中的默认值，或在 LangGraph Studio 运行时通过 configurable 参数动态切换。

## 开发指南

### 添加新的 Agent 模块

1. 创建新目录（如 `my_new_agent/`）
2. 定义 `agent.py`（包含 LangGraph 工作流）和 `utils/`（nodes、state、prompts 等）
3. 在 `langgraph.json` 的 `graphs` 中注册新 graph
4. 如需作为子图，在父图中 import 并添加节点

### 调试技巧

- 使用 `langgraph dev` 启动后，在 LangGraph Studio 中可以可视化查看 graph 执行流程
- 每个节点的输入输出状态都可以在 Studio 中查看
- 可以断点调试单个节点的执行
