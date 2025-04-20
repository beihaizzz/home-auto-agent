import datetime
from typing import List

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Send

from HomeBuddyAgent.utils.structs import RouterScore
from common.common_utils import get_search_params, select_and_execute_search,get_model
from common.structs import Queries, DeviceCalls, DeviceCall, DeviceResult,DeviceModelFactory
from HomeBuddyAgent.utils.prompts import node_agent_prompt, \
    node_retrieve_missing_info_prompt, additional_info_prompt, \
    node_generate_prompt_device_call, prompt_for_feedback, command_router_prompt, query_gen_prompt
from HomeBuddyAgent.utils.state import State, InfoState
from HomeBuddyAgent.utils.structs import AdditionalInfo, AdditionalInfos
from HomeBuddyAgent.utils.tools import retriever_tool, tools_for_info
from common.configuration import Configuration

from typing import Literal

from jinja2 import Template
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from HomeBuddyAgent.utils.structs import ClarityScore





def filter(state: State):
    """
    在消息传到正式处理的节点之前，将消息提取为question，初始化retriever_next为false
    :param state:
    :return:
    """
    return {
        "messages": [HumanMessage(content=state['question'])],
        "tool_cache": [],
        "device_configs": [],
        "answer": "",
        "feed_back": False,
        "time_now": datetime.datetime.now(),
        "location": "中国四川省成都市郫都区",
        "additional_info": [],
        "device_call_results": [],
        "device_calls": DeviceCalls(
            device_calls=[]
        ),
    }


def agent(state: State, config: RunnableConfig):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
        :param state:
        :param config:
    """
    print("---CALL AGENT---")
    configurable = Configuration.from_runnable_config(config)
    messages = state["messages"]
    # model = ChatGroq(model_name="qwen-2.5-32b", stream=True)
    # model = ChatOpenAI(model="gpt-4o-mini")
    # 使用配置信息获取模型
    model = get_model(
        model_provider=configurable.tool_call_provider,
        model_name=configurable.tool_call_model
    )
    # 只绑定retriever_tool，如果需要检索设备信息，就调用工具在向量数据库中检索
    model = model.bind_tools([retriever_tool])
    prompt = SystemMessage(
        content=node_agent_prompt
    )
    messages = [prompt] + messages
    print(messages)
    response = model.invoke(messages)
    return {"messages": [response]}


def command_router(state: State, config: RunnableConfig) -> Command[Literal["executor", "additional_info_collect"]]:
    print("===========路由==============")
    # model = ChatGroq(model_name="qwen-2.5-32b")
    # model = ChatOpenAI(model="gpt-4o-mini")
    configurable = Configuration.from_runnable_config(config)
    model = get_model(
        model_provider=configurable.structured_output_provider,
        model_name=configurable.structured_output_model
    )
    llm_with_structured_output = model.with_structured_output(RouterScore)
    template = Template(command_router_prompt, autoescape=False)
    query = template.render(
        {"question": [state["question"]],
         "device_configs": state['device_configs'],
         }
    )
    score = llm_with_structured_output.invoke([SystemMessage(content=query)])
    if score.command_score == "agent":
        return Command(
            goto=Send("additional_info_collect",
                      {
                          "device_configs": state['device_configs'],
                          "question": f"user:{state['question']},info for agent:{score.info_for_agent}",
                          "location": state['location'],
                          "time_now": state['time_now']
                      }
                      )
        )
    else:
        return Command(
            goto=Send(
                "executor",
                {
                    "question": state['question'],
                    "device_configs": state['device_configs'],
                    "tool_using": False,
                    "feed_back":False
                }
            )
        )


def check_command_clarity(state: State, config: RunnableConfig) -> Literal["generate", "additional_info_collect"]:
    """
    根据上下文判断用户控制设备的自然语言指令是否清晰完整。

    Args:
        state (messages): The current state containing the user's command and context

    Returns:
        str: 'complete' if command is clear and complete, 'retrieve_more_info' if ambiguous or incomplete
    """
    print("---检查指令清晰度---")

    # 数据模型

    # 初始化LLM
    # model = ChatGroq(model_name="qwen-2.5-32b", stream=True)
    configurable = Configuration.from_runnable_config(config)
    model = get_model(
        model_provider=configurable.structured_output_provider,
        model_name=configurable.structured_output_model
    )
    print(ClarityScore.schema())
    llm_with_tool = model.with_structured_output(ClarityScore)

    # 提示模板，加入上下文
    prompt = PromptTemplate(
        template="""你是一个 AI 助手，负责评估用户自然语言命令在特定设备控制上下文中的清晰度和完整性。\n
        以下是上下文（设备配置、环境状态等）：\n\n {context} \n\n
        以下是用户的命令：{question} \n\n
        根据提供的上下文，评估该命令是否清晰且包含执行设备控制动作所需的所有必要参数。\n
        请考虑以下因素：
        - 目标设备是否明确指定，并且与上下文中的设备匹配？
        - 是否提供了上下文所需的必要参数（例如，空调的温度、开关状态等）？
        - 在此上下文中，用户意图是否 unambiguous（无歧义）？\n\n
        输出一个清晰度评分（'clear' 或 'ambiguous'），并在必要时说明缺少的信息，基于上下文提示的需要。\n
        如果命令中涉及的参数信息（即 `params` 中的属性）不完整或模糊不清，明确指出这些参数的缺失或不明确。\n\n
        示例：
        1. 上下文：
           {{
             "devices": {{
               "curtain": {{
                 "type": "curtain",
                 "params": {{
                   "open_close": {{ "type": "boolean", "description": "窗帘开合状态", "value_range": "true/false" }},
                   "position": {{ "type": "integer", "description": "窗帘位置百分比", "value_range": "0-100" }}
                 }}
               }}
             }}
           }}
           命令："把窗帘打开"
           评估：clarity_score='clear', missing_info='' (备注：尽管未指定位置百分比，但‘打开’默认可视为 open_close=true，因此无需额外参数)
        2. 上下文：
           {{
             "devices": {{
               "humidifier": {{
                 "type": "humidifier",
                 "params": {{
                   "power": {{ "type": "boolean", "description": "开关状态", "value_range": "true/false" }},
                   "humidity_level": {{ "type": "integer", "description": "设定湿度", "value_range": "30-80" }},
                   "mist_output": {{ "type": "integer", "description": "雾量大小", "value_range": "1-3" }}
                 }}
               }}
             }}
           }}
           命令："打开加湿器"
           评估：clarity_score='ambiguous', missing_info='缺少湿度级别和雾量参数，params 信息不完整'""",
        input_variables=["context", "question"],
    )

    # 创建处理链
    chain = prompt | llm_with_tool

    # 获取用户指令和上下文
    question = state["question"]
    context = state["device_configs"]

    print("---用户指令---")
    print(question)
    print("---上下文---")
    print(context)

    # 获取评分结果
    result = chain.invoke({"question": question, "context": context})
    clarity = result.clarity_score
    missing_info = result.missing_info

    print("---评分结果---")
    print(f"Clarity: {clarity}")
    if missing_info:
        print(f"Missing info: {missing_info}")

    # 根据评分决定下一步
    if clarity == "clear":
        print("---DECISION: 指令完整清晰---")
        return "generate"
    else:
        print("---DECISION: 指令模糊，需要更多信息---")
        return "additional_info_collect"


def should_feed_back(state: State) -> Literal["device_call", "end"]:
    """"
    根据反馈判断是否再次调用重试
    """


def clearly_check(state: State) -> Command[Literal["generate", "additional_info_collect"]]:
    target = check_command_clarity(state)
    if target == "additional_info_collect":
        return Command(
            goto=Send("additional_info_collect",
                      {
                          "question": state['question'],
                          "device_configs": state['device_configs'],
                      }
                      )
        )
    elif target == "generate":
        return Command(
            goto="generate"
        )


# sub graph===================================================


def generate_queries(state: InfoState, config: RunnableConfig):
    """
    根据已经生成的场景和是否需要额外信息来进行判断，如果需要额外信息就编写网络搜索词条来获取准确参数
    搜集完成或者不收集之后就将所有信息汇总由规划模型生成合适的参数信息
    :param state:
    :param config:
    :return:
    """
    configurable = Configuration.from_runnable_config(config)
    model = get_model(
        model_provider=configurable.structured_output_provider,
        model_name=configurable.structured_output_model
    )
    model_with_structured_output = model.with_structured_output(Queries)
    template = Template(query_gen_prompt, autoescape=False)
    query = template.render({
        "question": state['question'],
        "location": state['location'],
        "device_configs": state['device_configs']
    })
    response = model_with_structured_output.invoke([SystemMessage(content=query)])
    print(response)
    return {
        "search_queries": response.queries
    }


async def search_web(state: InfoState, config: RunnableConfig):
    """为部分查询执行网络搜索。

    此节点：
    1. 获取生成的查询
    2. 使用配置的搜索API执行搜索
    3. 将结果格式化为可用的上下文

    参数：
        state: 当前状态，包含搜索查询
        config: 搜索API配置

    返回：
        包含搜索结果和更新迭代次数的字典
    """

    # 获取状态
    search_queries = state['search_queries']

    # 获取配置
    # search_api = get_config_value(configurable.search_api)
    search_api = "tavily"
    search_api_config = {}  # 获取配置字典，默认为空
    params_to_pass = get_search_params(search_api, search_api_config)  # 过滤参数

    # 网络搜索
    query_list = [query.search_query for query in search_queries]
    print(query_list)
    print(params_to_pass)
    # 使用参数进行网络搜索
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    return {
        # "source_str": source_str,
        # "search_iterations": state["search_iterations"] + 1,
        "additional_info": [AdditionalInfo(type="tavily_search_results_json", content=source_str)]
    }


## 下面的原先检索的环节效果不佳
def retrieve_missing_info(state: InfoState, config: RunnableConfig):
    """
    当指令不清晰时，智能调用工具搜寻缺失信息并更新状态。

    Args:
        state (messages): The current state containing the user's command and context

    Returns:
        dict: The updated state with retrieved information
        :param config:
    """
    print("---检索缺失信息---")
    # 初始化LLM
    configurable = Configuration.from_runnable_config(config)
    messages = state['sub_messages']
    if messages and not messages[-1].tool_call_id:
        # 如果最后一条消息并不是工具调用
        # model = ChatOpenAI(model="gpt-4o")
        model = get_model(
            model_provider=configurable.tool_call_provider,
            model_name=configurable.tool_call_model
        )
        model = model.bind_tools(tools_for_info, tool_choice="required")
        query = node_retrieve_missing_info_prompt.format(
            DEVICE_CONFIGS=state["device_configs"],
            QUESTION=state["question"],
            TIME_NOW=state['time_now'],
            LOCATION=state['location']
        )
        # 调用LLM生成检索策略
        response = model.invoke([SystemMessage(content=query)])
        return {"sub_messages": [response]}
    else:
        # model = ChatGroq(model_name="qwen-2.5-32b", stream=True)
        # model = ChatOpenAI(model="gpt-4o")
        model = get_model(
            model_provider=configurable.structured_output_provider,
            model_name=configurable.structured_output_model
        )
        #TODO：这里仅仅只把web search作为了信息来源
        model = model.with_structured_output(AdditionalInfos)

        # 在这次调用中使用到的工具列表
        tools = []
        index: int = len(messages) - 1
        while True and messages:
            if messages[index].tool_call_id:
                tools.append(messages[index])
                index -= 1
            else:
                break

        query = [SystemMessage(content=additional_info_prompt)] + tools
        response = model.invoke(query)

        return {"sub_messages": [AIMessage(content=str(response))], "additional_info": response}


def should_continue(state):
    messages = state["sub_messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


collect_info = ToolNode(tools_for_info)


def generate(state: State, config: RunnableConfig) -> Command[Literal["call_devices", "__end__"]]:
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
        model = model.with_structured_output(DeviceCallsDynamic)
        template = Template(node_generate_prompt_device_call, autoescape=False)
        query = template.render(
            {"question": state["question"], "device_configs": [docs_content],
             "additional_info": state["additional_info"]}
        )
        print(state["question"])
        print(query)
        # messages = messages + prompt_for_tool.invoke(
        #     {"question": [state["question"]], "device_configs": [docs_content],
        #      "additional_info": state["additional_info"]}).to_messages()
        response = model.invoke(messages + [HumanMessage(content=query)])
        print(response)
        return Command(
            update={
                "device_calls": response,
                "messages": [AIMessage(content=f"正在调用设备：{response}....")]
            },
            goto="call_devices"
        )

        # messages = messages + prompt_for_tool.invoke(
        #     {"question": [state["question"]], "device_configs": [docs_content],
        #      "additional_info": state["additional_info"]}).to_messages()
        # print("messages",messages)
    else:
        # model = ChatGroq(model_name="qwen-2.5-32b", stream=True)
        model = get_model(
            model_provider=configurable.writer_provider,
            model_name=configurable.writer_model
        )
        print("===========反馈=============")
        template = Template(prompt_for_feedback, autoescape=False)
        query = template.render(
            {"question": [state["question"]],
             "device_configs": [docs_content],
             "additional_info": state["additional_info"],
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


#
#
# def device_call(state: State):
#     """
#     控制设备
#     :param state:
#     :return:
#     """
#     response = _process_device_call(state["device_commands"])
#     return {
#         "feed_back": True,
#         "device_commands": [],
#         "device_result": response
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
