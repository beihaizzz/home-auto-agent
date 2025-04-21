import asyncio
import json
import os

from typing import Annotated, List

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from redis.asyncio import Redis as AsyncRedis  # 使用 redis.asyncio 的异步客户端
import requests

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_community.tools.tavily_search import TavilySearchResults
from HomeBuddyAgent.utils.state import State


async def get_redis_client():
    return await AsyncRedis(host='localhost', port=6379, db=0, decode_responses=True, password=os.getenv("REDIS_PASSWORD"))


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
        vectorstore =state['vector_store']
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
    # vectorstore = rag_loder()
    #
    # search_results: List[Document] = []
    # for query in query_list:
    #     # 使用向量存储进行相似性搜索，获取最相关的1个结果
    #     search_result = vectorstore.similarity_search(query=query, k=1)  # 后续优化，根据用户实际设备数量
    #     search_results += search_result
    #
    # # 提取文档内容并解析为JSON
    # doc_contents = [json.loads(doc.page_content) for doc in search_results]
    #
    # # 将doc_contents转换为格式化的字符串，避免列表格式问题
    # content_str = json.dumps(doc_contents, ensure_ascii=False, indent=2)
    #
    # # 创建ToolMessage，content为字符串
    # tool_message = ToolMessage(
    #     content=content_str,
    #     tool_call_id=tool_call_id
    # )
    #
    # # 将条件边给迁移过来
    # return Command(
    #     update={
    #         "messages": [
    #             tool_message
    #         ],
    #         "device_configs": search_results,
    #     }
    # )


# 不使用这个api了，使用taivly
@tool
def get_weather(tool_call_id: Annotated[str, InjectedToolCallId],
                state: Annotated[State, InjectedState]) -> Command:
    """
    获取成都市郫都区的实时天气信息。

    该工具通过腾讯天气 API 获取指定地区的天气数据，并返回格式化的天气信息。
    如果请求失败或数据异常，返回错误提示。

    :param tool_call_id: 工具调用的唯一标识符，由系统自动注入。
    :param state: 当前代理状态，由系统自动注入。
    :return: Command 对象，包含更新后的状态。
    """
    # 固定参数
    province = "四川"
    city = "成都"
    county = "郫都区"
    base_url = "https://wis.qq.com/weather/common"
    params = {
        "source": "pc",
        "weather_type": "observe",
        "province": province,
        "city": city,
        "county": county
    }

    # 默认错误消息
    error_message = ToolMessage(
        content=f"无法获取 {city}{county} 的天气信息，返回的数据状态异常。",
        tool_call_id=tool_call_id
    )
    tool_messages: List[ToolMessage] = state["tool_cache"]

    try:
        # 发送请求
        response = requests.get(base_url, params=params, timeout=10)  # 添加超时
        response.raise_for_status()  # 检查 HTTP 状态码

        # 解析响应
        data = response.json()
        if data.get("status") == 200 and "data" in data and "observe" in data["data"]:
            observe = data["data"]["observe"]
            weather_info = {
                "weather": observe.get("weather", "未知"),
                "temperature": observe.get("degree", "未知"),
                "humidity": observe.get("humidity", "未知"),
                "wind_direction": observe.get("wind_direction_name", "未知"),
                "wind_power": observe.get("wind_power", "未知")
            }

            info = (f"{city}{county} 的天气是 {weather_info['weather']}，"
                    f"温度是 {weather_info['temperature']}°C，"
                    f"湿度为 {weather_info['humidity']}%，"
                    f"风向为 {weather_info['wind_direction']}，"
                    f"风力为 {weather_info['wind_power']}。")
            tool_message = ToolMessage(
                content=(
                    f"{city}{county} 的天气是 {weather_info['weather']}，"
                    f"温度是 {weather_info['temperature']}°C，"
                    f"湿度为 {weather_info['humidity']}%，"
                    f"风向为 {weather_info['wind_direction']}，"
                    f"风力为 {weather_info['wind_power']}。"
                ),
                tool_call_id=tool_call_id
            )
            print("====== Tool message from weather ====")
            print(tool_message)
        else:
            return Command(
                update={
                    "tool_cache": tool_messages + [error_message],
                    "messages": [error_message],
                    "mark": state["mark"] + ["weather_error"],
                    "tool_using": True
                }
            )

        # 成功时的状态更新
        return Command(
            update={
                "tool_cache": tool_messages + [tool_message],
                "messages": [tool_message],
                "mark": state["mark"] + ["weather"],
                "tool_using": True,
                "additional_info": [info]
            }
        )

    except requests.RequestException as e:
        print(f"天气请求失败：{str(e)}")
        return Command(
            update={
                "tool_cache": tool_messages + [error_message],
                "messages": [error_message],
                "mark": state["mark"] + ["weather_error"],
                "tool_using": True
            }
        )
    except Exception as e:
        print(f"发生未知错误：{str(e)}")
        return Command(
            update={
                "tool_cache": tool_messages + [error_message],
                "messages": [error_message],
                "mark": state["mark"] + ["weather_error"],
                "tool_using": True
            }
        )


tools_for_info = [TavilySearchResults(max_results=3)]
# # 为每个工具生成一个唯一的ID，并将其添加到tool_registry字典中。
# tool_registry = {
#     str(uuid.uuid4()): tool for company in tools
# }
#
# tool_documents = [
#     Document(
#         page_content=tool.__doc__,
#         id=id,
#         metadata={"tool_name": tool.__name__},
#     )
#     for id, tool in tool_registry.items()
# ]
# vector_store = InMemoryVectorStore(embedding=OpenAIEmbeddings())
# document_ids = vector_store.add_documents(tool_documents)
#
#
#
# # 实际上是个node，暂时放在这，避免过多的调用文件反复引入造成延迟
# def select_tools(state: State):
#     last_user_message = state["messages"][-1]
#     query = last_user_message.content
#     tool_documents = vector_store.similarity_search(query)
#     return {"selected_tools": [document.id for document in tool_documents]}
