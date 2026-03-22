import asyncio
import os
from functools import lru_cache
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings

from langsmith import traceable
from tavily import AsyncTavilyClient
from common.configuration import BaseProvider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=1)
def rag_loader():
    persist_directory = os.getenv("VECTOR_STORE_PATH")
    #embeddings = OpenAIEmbeddings()
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",  # 最新最强嵌入模型
    )
    vector_store = Chroma(
        collection_name="vector_collection_for_agent",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vector_store


@lru_cache(maxsize=4)
def get_model(model_provider: BaseProvider, model_name: str):
    env = os.getenv("ENV", "dev")  # 默认测试环境
    load_dotenv(f"../../.env.dev.{env}", override=True)
    print(f"环境变量加载+++++++++++++++++++")
    if model_provider == "groq":
        model = ChatGroq(model_name=model_name)
    elif model_provider == "anthropic":
        model = ChatAnthropic(model_name=model_name)
    elif model_provider == "openai":
        model = ChatOpenAI(model_name=model_name, )
    elif model_provider == "deepseek":
        # 注意这里的deepseek是硅基流动的
        model = init_chat_model(model_name)
    elif model_provider == "qwen":
        model = ChatOpenAI(
            model=model_name,
            streaming=False,
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            base_url=os.getenv('DASHSCOPE_API_BASE')
        )

    else:
        raise ValueError(f"Unsupported model type: {model_provider}")
    return model


@traceable
async def tavily_search_async(search_queries):
    """
    Performs concurrent web searches using the Tavily API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process

    Returns:
            List[dict]: List of search responses from Tavily API, one per query. Each response has format:
                {
                    'query': str, # The original search query
                    'follow_up_questions': None,
                    'answer': None,
                    'images': list,
                    'results': [                     # List of search results
                        {
                            'title': str,            # Title of the webpage
                            'url': str,              # URL of the result
                            'content': str,          # Summary/snippet of content
                            'score': float,          # Relevance score
                            'raw_content': str|None  # Full page content if available
                        },
                        ...
                    ]
                }
    """
    tavily_async_client = AsyncTavilyClient()
    search_tasks = []
    for query in search_queries:
        search_tasks.append(
            tavily_async_client.search(
                query,
                max_results=5,
                include_raw_content=True,
                topic="general"
            )
        )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    return search_docs


def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=True):
    """
    Takes a list of search responses and formats them into a readable string.
    Limits the raw_content to approximately max_tokens_per_source tokens.

    Args:
        search_response: List of search response dicts, each containing:
            - query: str
            - results: List of dicts with fields:
                - title: str
                - url: str
                - content: str
                - score: float
                - raw_content: str|None
        max_tokens_per_source: int
        include_raw_content: bool

    Returns:
        str: Formatted string with deduplicated sources
    """
    # Collect all results
    sources_list = []
    for response in search_response:
        sources_list.extend(response['results'])

    # Deduplicate by URL
    unique_sources = {source['url']: source for source in sources_list}

    # Format output
    formatted_text = "Content from sources:\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"{'=' * 80}\n"  # Clear section separator
        formatted_text += f"Source: {source['title']}\n"
        formatted_text += f"{'-' * 80}\n"  # Subsection separator
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
        formatted_text += f"{'=' * 80}\n\n"  # End section separator

    return formatted_text.strip()


def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filters the search_api_config dictionary to include only parameters accepted by the specified search API.

    Args:
        search_api (str): The search API identifier (e.g., "exa", "tavily").
        search_api_config (Optional[Dict[str, Any]]): The configuration dictionary for the search API.

    Returns:
        Dict[str, Any]: A dictionary of parameters to pass to the search function.
    """
    # Define accepted parameters for each search API
    SEARCH_API_PARAMS = {
        "exa": ["max_characters", "num_results", "include_domains", "exclude_domains", "subpages"],
        "tavily": [],  # Tavily currently accepts no additional parameters
        "perplexity": [],  # Perplexity accepts no additional parameters
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
        "linkup": ["depth"],
    }

    # Get the list of accepted parameters for the given search API
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # If no config provided, return an empty dict
    if not search_api_config:
        return {}

    # Filter the config to only include accepted parameters
    return {k: v for k, v in search_api_config.items() if k in accepted_params}


async def select_and_execute_search(search_api: str, query_list: list[str], params_to_pass: dict) -> str:
    """Select and execute the appropriate search API.

    Args:
        search_api: Name of the search API to use
        query_list: List of search queries to execute
        params_to_pass: Parameters to pass to the search API

    Returns:
        Formatted string containing search results

    Raises:
        ValueError: If an unsupported search API is specified
    """
    if search_api == "tavily":
        print("开始搜索")
        search_results = await tavily_search_async(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000, include_raw_content=False)
    # elif search_api == "perplexity":
    #     search_results = perplexity_search(query_list, **params_to_pass)
    #     return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    # elif search_api == "exa":
    #     search_results = await exa_search(query_list, **params_to_pass)
    #     return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    # elif search_api == "arxiv":
    #     search_results = await arxiv_search_async(query_list, **params_to_pass)
    #     return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    # elif search_api == "pubmed":
    #     search_results = await pubmed_search_async(query_list, **params_to_pass)
    #     return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    # elif search_api == "linkup":
    #     search_results = await linkup_search(query_list, **params_to_pass)
    #     return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    # else:
    #     raise ValueError(f"Unsupported search API: {search_api}")
