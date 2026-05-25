# Home Auto Agent - 智能家居控制系统

## 项目简介

Home Auto Agent 是一个基于 **LangGraph** 的智能家居控制系统，支持通过自然语言与智能设备交互。核心能力包括：

- 自然语言理解与设备控制（开灯、调空调温度等）
- 基于向量数据库的语义化设备检索
- 智能场景规划（起床模式、离家模式等）
- 多 LLM 供应商支持（OpenAI / Anthropic / Groq / DeepSeek / 通义千问）
- **MCP协议支持**：通过标准 Model Context Protocol 与外部设备控制Server对接

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
│       ├── structs.py       # 数据模型（路由评分、清晰度评分等）
│       └── mcp_integration.py # MCP协议集成模块
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
│   ├── VectorStore/         # Chroma 向量数据库（本地持久化存储）
│   └── mcp/                 # MCP协议相关
│       ├── protocol.py      # MCP协议数据结构定义
│       ├── client.py        # MCP客户端实现
│       └── device_mappings.json # 设备类型映射配置
│
├── test/                    # 测试模块
│   ├── mcp_mock_server.py   # MCP模拟服务器（用于测试）
│   └── test_mcp_client.py   # MCP客户端测试
│
├── oneNetConfig.json        # 设备配置文件（模拟的设备列表）
├── init_vector_store.py     # 向量数据库初始化脚本
├── langgraph.json           # LangGraph 入口配置
├── requirements.txt         # Python 依赖
└── .env.example             # 环境变量模板
```

## 环境要求

- **Python** 3.11+
- **Redis**（用于设备检索缓存）
- **API Keys**：至少需要一个 LLM 提供商的 API Key（如 OpenAI、Anthropic、Groq、DeepSeek 或通义千问）

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

# MCP Server 配置（对接外部设备控制Server）
MCP_SERVER_URL=http://localhost:8080
MCP_API_KEY=xxx

# Redis 密码（如果 Redis 设置了密码）
REDIS_PASSWORD=xxx

# 向量数据库存储路径
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
    │    executor             info_graph（补充信息）
    │    （执行设备控制）           │
    │         │                    ▼
    │         ▼               generate（生成回复）
    │       返回结果                │
    │                              ▼
    │                           返回回复
    │
    └── 不需要检索 ──► 直接结束
```

### MCP协议集成

本项目作为 **MCP Client**，通过标准 MCP 协议与外部设备控制 Server 通信：

```
Home Auto Agent (MCP Client)
        │
        ▼ HTTP POST /mcp/v1/control
        │
    External MCP Server
        │
        ▼
    设备控制执行
```

## MCP协议接口

### 设备控制接口

**Endpoint**: `POST /mcp/v1/control`

**请求格式**:

```json
{
  "version": "1.0",
  "payload_type": "request",
  "request_id": "uuid-string",
  "timestamp": "2024-01-01T00:00:00Z",
  "source": "home-auto-agent",
  "controls": [
    {
      "device_id": "device-uuid",
      "device_name": "空调",
      "device_type": "air_conditioner",
      "action": "set_value",
      "parameters": { "temperature": 24 }
    }
  ],
  "timeout": 30,
  "execute_strategy": "sequential"
}
```

**响应格式**:

```json
{
  "version": "1.0",
  "payload_type": "response",
  "request_id": "uuid-string",
  "timestamp": "2024-01-01T00:00:00Z",
  "overall_status": "success",
  "results": [
    {
      "device_id": "device-uuid",
      "device_name": "空调",
      "status": "success",
      "message": "设备控制成功",
      "action": "set_value",
      "device_status": { "power": "on", "temperature": 24 }
    }
  ]
}
```

### 支持的设备类型

| 类型              | 说明   |
| ----------------- | ------ |
| `air_conditioner` | 空调   |
| `light`           | 灯光   |
| `curtain`         | 窗帘   |
| `heater`          | 加热器 |
| `fan`             | 风扇   |
| `tv`              | 电视   |
| `speaker`         | 音箱   |
| `switch`          | 开关   |
| `sensor`          | 传感器 |
| `scene`           | 场景   |
| `humidifier`      | 加湿器 |

### 支持的动作类型

| 动作             | 说明         |
| ---------------- | ------------ |
| `turn_on`        | 打开设备     |
| `turn_off`       | 关闭设备     |
| `set_value`      | 设置参数值   |
| `get_status`     | 获取设备状态 |
| `toggle`         | 切换设备状态 |
| `scene_activate` | 激活场景     |

## 设备配置

设备信息在 `oneNetConfig.json` 中定义。每个设备包含：

```json
{
  "product_id": { "type": "string", "value": "设备产品ID" },
  "device_name": { "type": "string", "value": "设备名称" },
  "device_type": "设备类型",
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

新增设备时，编辑此文件并重新运行初始化脚本。

## 测试

### 启动 MCP Mock Server

```bash
python test/mcp_mock_server.py
```

Mock Server 运行在 `http://localhost:8080`，支持所有设备类型和动作。

## 常见问题

### Q: 如何添加新设备？

1. 在 `oneNetConfig.json` 中添加设备配置
2. 运行 `python init_vector_store.py` 重新初始化向量数据库
3. 如果需要测试，在 `test/mcp_mock_server.py` 中添加模拟设备

### Q: 如何对接外部 MCP Server？

1. 设置环境变量 `MCP_SERVER_URL` 为外部 Server 地址
2. 确保外部 Server 实现了 `/mcp/v1/control` 接口
3. 如果需要认证，设置 `MCP_API_KEY` 环境变量

### Q: MCP 请求失败怎么办？

检查以下几点：

- `MCP_SERVER_URL` 是否正确设置
- 外部 Server 是否正常运行
- 请求格式是否符合 MCP 协议规范
