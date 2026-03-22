# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Home Auto Agent — a LangGraph-based smart home assistant that uses natural language (Chinese) to control IoT devices. It combines vector search (Chroma), web search (Tavily), Redis caching, and multi-LLM support.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -U "langgraph-cli[inmem]"

# Run development server (starts LangGraph Studio at http://localhost:8000)
langgraph dev

# Initialize vector store (required on first run)
python test2.py
```

No test suite exists in this project.

## Architecture

Three agent modules, each following the same structure: `agent.py` (graph definition) + `utils/` (nodes, state, prompts, structs).

### Agent Modules

- **HomeBuddyAgent** — Main controller. Routes user intent through: `filter → agent → retriever → command_router → executor|info_graph → generate`. Entry point in `langgraph.json`.
- **basic_executor** — Device execution subgraph. Converts instructions to device API calls via `generate → should_continue → device_call` loop. Called as a subgraph by HomeBuddyAgent.
- **deep_planner_v1** — Scene planner. Generates daily smart home scenarios using LangGraph's Send API for parallel scene design. Registered as a separate graph in `langgraph.json`.

### Shared Code (`common/`)

- `configuration.py` — Runtime-configurable dataclass with enum providers for LLM selection (OpenAI, Anthropic, Groq, DeepSeek, Qwen). Configurable via LangGraph Studio UI.
- `common_utils.py` — `get_model()` factory, `rag_loader()` for Chroma, `tavily_search_async()` for web search.
- `structs.py` — `DeviceModelFactory` dynamically generates Pydantic models from `oneNetConfig.json` device schemas at runtime for structured LLM output.

### Key Patterns

- **Dynamic Pydantic model generation**: `DeviceModelFactory` reads JSON device configs and creates typed models + Union types for LLM structured output.
- **Structured output (multi-LLM compatible)**: `basic_executor/utils/nodes.py` 的 `generate` 节点使用 `json_mode` + 手动 JSON 解析实现结构化输出，而非 `with_structured_output()`。原因：各 LLM 提供商对 `json_schema`/`function_calling` 模式支持不一致（Qwen 思考模型不支持 `tool_choice=required`，`json_schema` 模式下多数模型会嵌套输出）。当前方案三层保障：prompt 注入动态 JSON Schema → `response_format=json_object` → 手动解析兼容嵌套。
- **Prompt templates**: Jinja2 templates in each module's `prompts.py`. Qwen models use `/no_think` suffix.
- **Caching**: Redis for device retrieval results; `@lru_cache` for model instances and vector store loader.
- **Async throughout**: Async Redis client, `asyncio.gather` for concurrent Tavily searches, `asyncio.to_thread` for blocking vector store operations.

## Configuration

- `langgraph.json` — Graph entry points and env file path (`.env.dev`)
- `oneNetConfig.json` — Device definitions (product_id, device_name, device_type, params schema). Adding a new device requires editing this file and re-running vector store initialization.
- `.env.dev` — API keys. Minimum required: `OPENAI_API_KEY` + `TAVILY_API_KEY`.

## Language

All user-facing prompts, comments, and documentation are in Chinese.