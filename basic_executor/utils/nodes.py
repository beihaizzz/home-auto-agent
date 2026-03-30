from typing import List, Literal
import os
import json as _json
from jinja2 import Template
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from functools import lru_cache

from langgraph.types import Command

from HomeBuddyAgent.utils.prompts import node_generate_prompt_device_call, prompt_for_feedback
from basic_executor.utils.state import State
from langgraph.prebuilt import ToolNode

from common.common_utils import get_model, get_embedding_model
from common.structs import ConfigT, DeviceModelFactory, DeviceCall, DeviceResult, DeviceCalls
from common.configuration import Configuration

@lru_cache(maxsize=4)
def _rag_loder(model_provider: str = "qwen"):
    # with open("D:\DevelopFiles\pycharms\Stimulate\AI\files\function_file.json", encoding="UTF-8") as f:
    #     file = f.read()
    #
    # text_splitter = CharacterTextSplitter(
    #     separator="\n\n",
    #     chunk_size=5,
    #     chunk_overlap=2,
    #     length_function=len,
    #     is_separator_regex=False
    # )
    #
    # texts = text_splitter.create_documents([file])
    # vector_store = Chroma.from_documents(texts, OpenAIEmbeddings())
    persist_directory = r"./ChromaDB/test"

    embeddings = get_embedding_model(model_provider)

    vector_store = Chroma(
        collection_name="vector_collection_for_agent",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vector_store


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


def retrieve(state: State):
    # print("retrieve")
    # 从state中获取当前的模型提供商
    model_provider = state.get('config', {}).get('structured_output_provider', 'qwen')
    # 提取枚举值的字符串表示
    if hasattr(model_provider, 'value'):
        model_provider = model_provider.value
    vector_store = _rag_loder(model_provider)
    retrieved_docs = vector_store.similarity_search(state["question"], k=4)
    return {"context": retrieved_docs, "tool_using": False}


def generate(state: State, config: RunnableConfig) -> Command[Literal["action", "__end__"]]:
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
         :param state:
         :param config:
    """
    configurable = Configuration.from_runnable_config(config)
    docs_content = state["device_configs"]
    messages = state["messages"]
    print(state["feed_back"])
    if not state["feed_back"]:
        print("---调用工具---")
        # model = ChatOpenAI(model="gpt-4o")
        model = get_model(
            model_provider=configurable.structured_output_provider,
            model_name=configurable.structured_output_model
        )
        factory = DeviceModelFactory()
        factory.generate_all(state['device_configs'])
        ConfigUnion = factory.get_union_type()
        DeviceCallsDynamic = DeviceCalls[ConfigUnion]
        DeviceCallsDynamic.__name__ = "DeviceCallsDynamic"
        print(f"{DeviceCallsDynamic.__name__}+++++++++++++++++++++++++++++++++++++++++++++++++")
        model = model.bind(response_format={"type": "json_object"})

        template = Template(node_generate_prompt_device_call, autoescape=False)
        query = template.render(
            {"question": state["question"], "device_configs": [docs_content],
             "additional_info": " ",
             "json_schema": _json.dumps(DeviceCallsDynamic.model_json_schema(), ensure_ascii=False, indent=2)}
        )
        print(state["question"])
        print(query)
        raw_response = model.invoke(messages + [HumanMessage(content=query)])
        # 手动解析 JSON，兼容各模型可能的嵌套输出
        data = _json.loads(raw_response.content)
        # 处理嵌套：模型可能返回 {"device_calls": {"device_calls": [...]}} 而非 {"device_calls": [...]}
        if "device_calls" in data and isinstance(data["device_calls"], dict):
            data = data["device_calls"]
        response = DeviceCallsDynamic.model_validate(data)
        print(response)
        return Command(
            update={
                "device_calls": response,
                "messages": [AIMessage(content=f"正在调用设备：{response}....")]
            },
            goto="action"
        )

    else:
        model = get_model(
            model_provider=configurable.writer_provider,
            model_name=configurable.writer_model
        )
        print("===========反馈=============")
        template = Template(prompt_for_feedback, autoescape=False)
        query = template.render(
            {"question": [state["question"]],
             "device_configs": [docs_content],
             # "additional_info": state["additional_info"],
             "device_call_result": state["device_call_results"]
             }
        )
        response = model.invoke(messages + [SystemMessage(content=query)])
        return Command(
            update={
                "answer": response.content,
                "messages": messages + [SystemMessage(content=query)] + [response],
                "feed_back": False
            },
            goto="__end__"
        )


# def generate(state: State, config: RunnableConfig):
#     print("generate")
#     configurable = Configuration.from_runnable_config(config)
#     model = get_model(
#         model_provider=configurable.tool_call_provider,
#         model_name=configurable.tool_call_model
#     ).bind_tools(tools)
#     docs_content = "\n\n".join(doc.page_content for doc in state["device_configs"])
#     messages = state["messages"]
#     if not state["tool_using"]:
#         messages = messages + prompt_for_tool.invoke(
#             {"question": [state["question"]], "context": [docs_content]}).to_messages()
#         # print("messages",messages)
#     response = model.invoke(messages)
#     # llm.with_structured_output(
#     #
#     # )
#     # print(response)
#     return {
#         "answer": response.content,
#         "device_configs": [],
#         "messages": messages + [response]
#     }


def _process_device_call(device_commands: List[DeviceCall]) -> DeviceResult:
    """
    处理设备调用请求，并生成详细的操作结果反馈。

    参数：
    - device_commands: 设备操作命令列表，每个命令包含设备信息和参数。

    返回：
    - 字符串格式的操作结果，包含每个设备的名称、参数和状态。
    """

    print("==================设备模拟执行===================")
    print(device_commands)
    response = DeviceResult(success=True,
                            message="设备都已成功执行",
                            data={})
    return response


def device_call(
        state: State
):
    """
    工具描述：根据用户的自然语言命令控制智能家居设备，并返回操作结果供生成反馈。
    :param devices: 一个设备操作命令列表，每个命令必须包含以下字段：
        - device_id (str): 设备的唯一产品ID，例如 'aX23Jrf5xy' 表示空调。
        - device_name (str): 设备的具体名称，例如 '空调' 或 '卧室灯'。
        - params (dict): 操作参数，键值对形式，必须提供，例如 {'power': 'true', 'temperature': '24'}。
        - order (int): 调用顺序。
    :param tool_call_id:
    :param state:
    :return:
    """
    # 处理设备调用并获取结果
    result = _process_device_call(state['device_calls'].device_calls)
    if result.success:
        print("====== Device Call =====")
        # 返回 Command 对象，更新状态
        return {
            "feed_back": True,
            "device_call_results": result
        }
    else:
        print("====== Device Call Failed =====")
        return {
            "feed_back": False,
            "messages": [AIMessage(content=f"设备调用失败：{result}")]
        }

# tool_node = ToolNode(tools)
