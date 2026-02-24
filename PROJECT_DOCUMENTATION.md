# Home Auto Agent 项目说明文档

## 项目概述

**Home Auto Agent** 是一个基于 LangGraph 的智能家居控制系统，通过多层级 AI 代理实现自然语言交互的设备控制和智能场景规划。系统整合了向量检索、Web 搜索和多 LLM 提供商，提供灵活且强大的智能家居解决方案。

### 核心特性

- **自然语言交互**：用户可以用自然语言描述需求，系统自动理解并执行
- **智能设备检索**：基于向量数据库的语义搜索，快速定位相关设备
- **上下文感知**：集成 Web 搜索获取实时信息，增强决策能力
- **场景规划**：自动生成一整天的智能家居方案
- **多 LLM 支持**：支持 OpenAI、Anthropic、Groq、DeepSeek、Qwen 等多个提供商
- **可扩展架构**：模块化设计，易于添加新功能和设备类型

---

## 技术栈

### 核心框架
- **LangGraph 0.3.25** - AI 工作流编排引擎
- **LangChain 0.3.23** - LLM 应用开发框架
- **Pydantic 2.11.2** - 数据验证和建模

### 数据存储
- **Chroma 0.6.3** - 向量数据库，存储设备配置
- **Redis 5.2.1** - 缓存和状态管理
- **SQLAlchemy 2.0.40** - ORM 支持

### LLM 提供商
- **OpenAI** - gpt-4o-mini（工具调用、结构化输出）
- **Anthropic** - claude-3-7-sonnet（规划任务）
- **Groq、DeepSeek、Qwen** - 备选提供商

### 其他服务
- **Tavily API** - Web 搜索
- **FastAPI 0.115.12** - API 服务框架
- **OpenTelemetry** - 可观测性

---

## 项目结构

```
home-auto-agent/
├── HomeBuddyAgent/          # 主智能家居助手模块
│   ├── agent.py             # 主工作流图定义
│   └── utils/
│       ├── state.py         # 状态定义
│       ├── nodes.py         # 核心节点实现
│       ├── tools.py         # 检索工具
│       ├── prompts.py       # 提示词模板
│       └── structs.py       # 数据模型
│
├── basic_executor/          # 设备执行器模块
│   ├── agent.py             # 执行器工作流
│   └── utils/
│       ├── state.py         # 执行器状态
│       └── nodes.py         # 执行节点
│
├── deep_planner_v1/         # 智能场景规划模块
│   ├── agent.py             # 规划工作流
│   └── utils/
│       ├── states.py        # 规划状态
│       ├── nodes.py         # 规划节点
│       ├── structs.py       # 场景数据模型
│       └── redis_cache.py   # Redis 缓存管理
│
├── common/                  # 共享模块
│   ├── configuration.py     # 配置管理
│   ├── structs.py          # 核心数据结构
│   ├── common_utils.py     # 工具函数
│   └── VectorStore/        # Chroma 向量数据库
│
├── oneNetConfig.json        # 设备配置文件
├── langgraph.json          # LangGraph 配置
├── requirements.txt        # 项目依赖
└── .env.dev                # 环境变量
```

---

## 核心功能

### 1. 智能家居助手（HomeBuddyAgent）

**功能描述**：主要的用户交互接口，处理自然语言指令并控制智能设备。

**核心能力**：
- 理解用户的自然语言指令
- 从向量数据库检索相关设备配置
- 评估指令清晰度并智能路由
- 生成结构化的设备控制命令
- 当信息不足时，自动搜索补充信息
- 提供用户友好的反馈

**工作流程**：
```
用户输入 → filter(提取问题) → agent(决策) → retriever(设备检索)
  → command_router(路由决策)
    ├→ executor: 执行设备控制
    └→ agent: 收集额外信息 → Web 搜索
  → generate(生成响应) → 返回结果
```

**关键文件**：
- `HomeBuddyAgent/agent.py:graph` - 主工作流定义
- `HomeBuddyAgent/utils/nodes.py` - 核心节点实现

---

### 2. 设备执行器（basic_executor）

**功能描述**：负责执行具体的设备控制命令。

**核心能力**：
- 将高层指令转换为设备调用
- 生成符合设备配置的参数
- 模拟设备执行并返回结果
- 支持多设备顺序执行

**工作流程**：
```
输入设备配置 → generate(生成 DeviceCalls)
  → should_continue(检查是否继续)
    ├→ continue: action(执行设备调用) → 循环
    └→ end: 返回结果
```

**关键文件**：
- `basic_executor/agent.py` - 执行器工作流
- `basic_executor/utils/nodes.py` - 执行节点

---

### 3. 智能场景规划器（deep_planner_v1）

**功能描述**：生成一整天的智能家居场景方案。

**核心能力**：
- 根据日期和设备配置生成场景列表
- 为每个场景设计具体方案
- 并行处理多个场景（使用 Send API）
- 集成 Web 搜索获取最佳实践
- 生成时间轴和设备调用计划

**工作流程**：
```
初始化 → generate_scenes(生成场景)
  → Send 到 generate_scheme(并行处理每个场景)
    ├→ generate_queries(生成搜索查询)
    ├→ search_web(Web 搜索)
    └→ design_scheme(设计方案)
  → gather_completed_schemes(汇总) → 返回
```

**场景示例**：
- 早晨起床：开窗帘、调节灯光、播放音乐
- 离家模式：关闭所有灯光、设置安防
- 回家模式：开启空调、调节温度、打开灯光
- 睡眠模式：关闭所有设备、设置夜灯

**关键文件**：
- `deep_planner_v1/agent.py` - 规划工作流
- `deep_planner_v1/utils/structs.py` - 场景数据模型

---

### 4. 向量检索系统

**功能描述**：基于语义相似度检索设备配置。

**核心能力**：
- 使用 OpenAI Embeddings 向量化设备配置
- Chroma 向量数据库存储和检索
- Redis 缓存加速查询
- 支持多关键词并行检索

**工作原理**：
```python
# 1. 从 oneNetConfig.json 加载设备配置
# 2. 转换为 Document 并向量化
# 3. 存储到 Chroma 数据库
# 4. 查询时使用语义搜索匹配最相关的设备
```

**关键文件**：
- `HomeBuddyAgent/utils/tools.py:retriever_tool` - 检索工具
- `common/common_utils.py:rag_loader` - 向量数据库加载器

---

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                      用户接口层                          │
│            (自然语言输入/对话交互)                       │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │   HomeBuddyAgent          │
         │   (主控制器)              │
         │                            │
         │  ┌──────────────────────┐ │
         │  │ Filter → Agent      │ │
         │  │ → Retriever         │ │
         │  │ → Command Router    │ │
         │  │ → Generate          │ │
         │  └──────────────────────┘ │
         └───┬──────────────────┬────┘
             │                  │
    ┌────────▼────────┐  ┌─────▼──────────────┐
    │ basic_executor  │  │ Info Collector     │
    │ (设备控制)      │  │ (信息收集)         │
    │                 │  │                    │
    │ Generate →      │  │ Generate Queries   │
    │ Action Loop     │  │ → Web Search       │
    └────────┬────────┘  └─────┬──────────────┘
             │                 │
    ┌────────▼─────────────────▼──────────┐
    │         外部服务层                   │
    │                                      │
    │  ┌──────────┐  ┌──────────┐        │
    │  │ Chroma   │  │ Tavily   │        │
    │  │ Vector DB│  │ Search   │        │
    │  └──────────┘  └──────────┘        │
    │                                      │
    │  ┌──────────┐  ┌──────────┐        │
    │  │ Redis    │  │ LLM      │        │
    │  │ Cache    │  │ Providers│        │
    │  └──────────┘  └──────────┘        │
    └──────────────────────────────────────┘
```

### 数据流

```
1. 用户输入
   ↓
2. State 初始化（question, location, time_now）
   ↓
3. LLM 决策（是否需要检索设备）
   ↓
4. 向量检索（Chroma + Redis 缓存）
   ↓
5. 路由决策
   ├→ 清晰指令: basic_executor
   └→ 模糊指令: 信息收集 → Web 搜索
   ↓
6. 生成 DeviceCalls（结构化输出）
   ↓
7. 设备执行 → DeviceResult
   ↓
8. 生成用户反馈
   ↓
9. 返回结果
```

---

## 核心数据模型

### State（状态）

```python
class State(MessagesState):
    question: str                      # 用户问题
    device_configs: List[Document]     # 设备配置文档
    answer: str                        # 最终答案
    location: str                      # 用户位置
    time_now: datetime                 # 当前时间
    feed_back: bool                    # 反馈标志
    device_call_results: List[DeviceResult]  # 设备调用结果
    device_calls: DeviceCalls          # 待执行的设备调用
    additional_info: List[AdditionalInfo]  # 附加信息
```

### Device（设备）

```python
class Device[ConfigT]:
    id: str           # 设备 ID（product_id）
    type: str         # 设备类型（如 "LED Light"）
    config: ConfigT   # 配置对象（动态生成的 Pydantic 模型）
```

### DeviceCall（设备调用）

```python
class DeviceCall[ConfigT]:
    device_name: str  # 设备名称
    device_id: str    # 设备 ID
    config: ConfigT   # 参数配置
    order: int        # 执行顺序
```

### Scene（场景）

```python
class Scene:
    name: str                        # 场景名称（如 "起床场景"）
    description: str                 # 描述
    research: bool                   # 是否需要研究
    start_time: str                  # 开始时间（HH:MM:SS）
    end_time: str                    # 结束时间（HH:MM:SS）
    involved_devices: List[Device]   # 涉及的设备
```

---

## 配置系统

### Configuration 类

系统使用统一的配置类管理所有 LLM 和服务提供商：

```python
@dataclass
class Configuration:
    # 查询生成
    number_of_queries: int = 2

    # 规划任务（推荐 Anthropic Claude）
    planner_provider: PlannerProvider = PlannerProvider.ANTHROPIC
    planner_model: str = "claude-3-7-sonnet-latest"

    # 文本生成（推荐 OpenAI）
    writer_provider: WriterProvider = WriterProvider.OPENAI
    writer_model: str = "gpt-4o-mini"

    # 结构化输出（推荐 OpenAI）
    structured_output_provider: StructuredOutputProvider = ...
    structured_output_model: str = "gpt-4o-mini"

    # 工具调用（推荐 OpenAI）
    tool_call_provider: ToolCallProvider = ToolCallProvider.OPENAI
    tool_call_model: str = "gpt-4o-mini"

    # 搜索服务
    search_api: SearchAPI = SearchAPI.TAVILY
```

### 环境变量

在 `.env.dev` 中配置：

```bash
# LLM API Keys
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-xxx
GROQ_API_KEY=xxx
DEEPSEEK_API_KEY=xxx
DASHSCOPE_API_KEY=xxx  # Qwen

# 服务配置
TAVILY_API_KEY=xxx
REDIS_PASSWORD=xxx
VECTOR_STORE_PATH=common/VectorStore

# API 基础 URL（可选）
DEEPSEEK_API_BASE=https://xxx
DASHSCOPE_BASE=https://xxx
```

---

## 使用指南

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env.dev
# 编辑 .env.dev 填入 API Keys
```

### 2. 初始化向量数据库

```bash
# 从 oneNetConfig.json 加载设备配置
python test2.py
```

这会将设备配置向量化并存储到 `common/VectorStore/` 目录。

### 3. 启动服务

```bash
# 使用 LangGraph CLI 启动开发服务器
langgraph dev
```

### 4. 访问 LangGraph Studio

浏览器打开 LangGraph Studio UI，可以：
- 可视化查看工作流图
- 交互式测试 HomeBuddyAgent
- 调试和追踪执行过程

### 5. 使用示例

**控制设备**：
```
用户: "打开客厅的灯"
系统: [检索设备] → [生成控制命令] → [执行] → "已为您打开客厅的灯"
```

**场景规划**：
```
用户: "生成明天的智能家居方案"
系统: [生成场景列表] → [并行设计方案] → [返回完整时间表]
```

**信息补充**：
```
用户: "什么温度比较舒适"
系统: [判断需要额外信息] → [Web 搜索] → [结合搜索结果回答]
```

---

## 关键技术亮点

### 1. 动态模型生成

使用 `DeviceModelFactory` 根据设备配置 JSON 动态生成 Pydantic 验证模型：

```python
# 自动从 JSON schema 生成 Python 类
# 支持 boolean、integer、string、object、array 等类型
# 用于 LLM structured_output 验证
```

### 2. 智能路由

`command_router` 节点评估指令清晰度：
- **清晰指令** → 直接执行（basic_executor）
- **模糊指令** → 收集信息（Web 搜索）→ 再处理

### 3. 并行处理

deep_planner_v1 使用 LangGraph 的 `Send` API：
- 生成多个场景
- 并行为每个场景设计方案
- 提高整体性能

### 4. 缓存优化

- **Redis 缓存**：检索结果缓存，加速重复查询
- **LRU 缓存**：向量数据库和 LLM 模型实例缓存

### 5. 多提供商支持

灵活配置不同任务使用不同的 LLM：
- 规划任务 → Anthropic Claude（推理能力强）
- 工具调用 → OpenAI（function calling 稳定）
- 结构化输出 → OpenAI（JSON mode 可靠）

---

## 扩展指南

### 添加新设备类型

1. 在 `oneNetConfig.json` 中添加设备配置：

```json
{
  "product_id": {"value": "xxx"},
  "device_name": {"value": "新设备"},
  "device_type": "新设备类型",
  "params": {
    "properties": {
      "param1": {"type": "boolean", "value_range": "true/false"}
    }
  }
}
```

2. 运行 `python test2.py` 重新加载向量数据库

### 添加新的 LLM 提供商

在 `common/common_utils.py` 的 `get_model` 函数中添加：

```python
elif model_provider == "new_provider":
    return NewProviderChatModel(model_name=model_name)
```

### 添加新节点

1. 在对应模块的 `utils/nodes.py` 中定义节点函数
2. 在 `agent.py` 中添加到工作流图：

```python
graph_builder.add_node("new_node", new_node_function)
graph_builder.add_edge("previous_node", "new_node")
```

---

## 已知问题

### 1. 硬编码路径

`basic_executor/utils/nodes.py` 中存在硬编码路径：

```python
persist_directory = r"D:\DevelopFiles\pycharms\..."
```

**解决方案**：使用环境变量 `VECTOR_STORE_PATH`

### 2. RAG Loader 异步阻塞

`rag_loader()` 在初始化 Chroma 时可能导致事件循环阻塞。

**已解决**：在 `HomeBuddyAgent/utils/tools.py` 中使用 `await to_thread(rag_loader)` 解决

### 3. API 密钥安全

避免在代码中硬编码 API 密钥，统一使用环境变量。

---

## 项目依赖

关键依赖版本：

```
langgraph==0.3.25
langchain==0.3.23
langchain-community==0.3.23
langchain-openai==0.3.7
langchain-anthropic==0.3.8
langchain-chroma==0.2.2
chromadb==0.6.3
pydantic==2.11.2
redis==5.2.1
tavily-python==0.5.1
fastapi==0.115.12
opentelemetry-api==1.29.0
```

完整依赖列表见 `requirements.txt`。

---

## 文件路径索引

| 功能 | 文件路径 |
|------|---------|
| 主工作流 | `HomeBuddyAgent/agent.py` |
| 状态定义 | `HomeBuddyAgent/utils/state.py` |
| 核心节点 | `HomeBuddyAgent/utils/nodes.py` |
| 检索工具 | `HomeBuddyAgent/utils/tools.py` |
| 提示词库 | `HomeBuddyAgent/utils/prompts.py` |
| 执行器 | `basic_executor/agent.py` |
| 规划器 | `deep_planner_v1/agent.py` |
| 配置类 | `common/configuration.py` |
| 数据结构 | `common/structs.py` |
| 工具函数 | `common/common_utils.py` |
| 向量数据库 | `common/VectorStore/` |
| 设备配置 | `oneNetConfig.json` |
| LangGraph 配置 | `langgraph.json` |

---

## 总结

Home Auto Agent 是一个功能完整、架构清晰的智能家居控制系统。通过 LangGraph 的工作流编排能力，结合向量检索、Web 搜索和多 LLM 支持，实现了从自然语言理解到设备控制的完整闭环。

系统的模块化设计使其易于扩展，支持添加新的设备类型、LLM 提供商和功能节点。无论是简单的设备控制还是复杂的场景规划，Home Auto Agent 都能提供智能、高效的解决方案。
