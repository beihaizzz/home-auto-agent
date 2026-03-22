import os
from enum import Enum
from dataclasses import  fields
from typing import Any, Optional, Dict

from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass


class BaseProvider(Enum):
    pass


class SearchAPI(BaseProvider):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"


class PlannerProvider(BaseProvider):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


class WriterProvider(BaseProvider):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


class ToolCallProvider(BaseProvider):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


class StructuredOutputProvider(BaseProvider):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    number_of_queries: int = 2  # 每次生成的查询数
    # planner其实还没有实装
    planner_provider: PlannerProvider = PlannerProvider.QWEN
    planner_model: str = "qwen3.5-plus"
    
    writer_provider: WriterProvider = WriterProvider.QWEN
    writer_model: str = "qwen3.5-plus"

    structured_output_provider: StructuredOutputProvider = StructuredOutputProvider.QWEN
    structured_output_model: str = "qwen3.5-plus"
    
    tool_call_provider: ToolCallProvider = ToolCallProvider.QWEN
    tool_call_model: str = "qwen3.5-plus"

    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    think_switch: bool = False  # 专门为Qwen3设计的开关

    @classmethod
    def from_runnable_config(
            cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
