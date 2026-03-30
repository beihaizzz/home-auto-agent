import json
import os

from typing import Annotated, List

from redis.asyncio import Redis as AsyncRedis  # 使用 redis.asyncio 的异步客户端

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from HomeBuddyAgent.utils.state import State
from common.common_utils import rag_loader
from asyncio import to_thread


async def get_redis_client():
    return await AsyncRedis(host='localhost', port=6379, db=0, decode_responses=True,
                            password=os.getenv("REDIS_PASSWORD"))


@tool
async def retriever_tool(query_list: List[str], tool_call_id: Annotated[str, InjectedToolCallId],
                         state: Annotated[State, InjectedState]
                         ):
    """
        Search for device configuration data in a document based on a list of query keywords and return the most relevant device information.

        :param query_list: A list of strings (List[str]) containing keywords to search for device-related data.
                           Multiple queries can be provided to refine the search (e.g., ["living room", "light"]).
                           Each keyword is searched independently, and results are combined.
        :param tool_call_id: The unique identifier for the tool call, injected automatically.
        :param state: The current state of the agent, injected automatically.
        :return: A Command object updating the state with search results and tool invocation details.
        """

    # 将 query_list 转换为 JSON 字符串作为 Redis 键
    # 获取异步 Redis 客户端
    redis_client = await get_redis_client()

    # 将 query_list 转换为 JSON 字符串作为 Redis 键
    query_key = json.dumps(query_list, ensure_ascii=False)

    # 检查 Redis 缓存（异步操作）
    cached_result = await redis_client.get(query_key)
    if cached_result:
        # 如果缓存存在，直接加载 search_results
        search_results = [Document(page_content=content) for content in json.loads(cached_result)]
    else:
        # 如果缓存不存在，执行搜索
        # 从state中获取当前的工具调用提供商
        model_provider = state.get('config', {}).get('tool_call_provider', 'qwen')
        # 提取枚举值的字符串表示
        if hasattr(model_provider, 'value'):
            model_provider = model_provider.value
        vectorstore = await to_thread(rag_loader, model_provider)

        # vectorstore = state['vector_store']
        search_results: List[Document] = []
        for query in query_list:
            # 使用向量存储进行相似性搜索，获取最相关的1个结果
            search_result = await vectorstore.asimilarity_search(query=query, k=1)
            search_results += search_result

        # 将搜索结果存入 Redis 缓存（异步操作）
        await redis_client.set(query_key, json.dumps([doc.page_content for doc in search_results]))

    # 关闭 Redis 连接
    await redis_client.aclose()  # redis.asyncio 使用 aclose() 而不是 close()

    # 提取文档内容并解析为 JSON
    doc_contents = [json.loads(doc.page_content) for doc in search_results]

    # 将 doc_contents 转换为格式化的字符串，避免列表格式问题
    content_str = json.dumps(doc_contents, ensure_ascii=False, indent=2)

    # 创建 ToolMessage，content 为字符串
    tool_message = ToolMessage(
        content=content_str,
        tool_call_id=tool_call_id
    )

    # 返回 Command 对象，更新状态
    return Command(
        update={
            "messages": [tool_message],
            "device_configs": search_results,
        }
    )
