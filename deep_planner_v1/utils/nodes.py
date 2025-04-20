import json
from datetime import datetime, date
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.types import Command, Send

from common.common_utils import get_search_params, select_and_execute_search, rag_loder, get_model
from common.configuration import Configuration
from deep_planner_v1.utils.prompts import scheme_plan_prompt, query_gen_prompt, scheme_gen_prompt
from deep_planner_v1.utils.states import SchemePlanState, SceneState
from deep_planner_v1.utils.redis_cache import RedisSchemeCache

from deep_planner_v1.utils.structs import Scene, Scenes, Scheme, Schemes
from common.structs import Queries, DeviceModelFactory, DeviceCalls
from pydantic import create_model, Field
from typing import Dict, Any, Literal


def _generate_device_model(device_json: Dict[str, Any]) -> type:
    """根据设备 JSON 配置生成 Pydantic 模型"""
    device_name = device_json['device_name']['value']
    params = device_json['params']['properties']

    # 定义模型字段
    fields = {}
    for param_name, param_info in params.items():
        param_type = param_info['type']
        description = param_info['description']
        value_range = param_info['value_range']

        # 根据类型和取值范围设置字段
        if param_type == 'boolean':
            fields[param_name] = (bool, Field(description=description))
        elif param_type == 'integer':
            min_val, max_val = map(int, value_range.split('-'))
            fields[param_name] = (int, Field(description=description, ge=min_val, le=max_val))
        elif param_type == 'string':
            allowed_values = value_range.split('/')
            fields[param_name] = (str, Field(description=description, pattern=f"^(?:{'|'.join(allowed_values)})$"))
        # 可以根据需要扩展其他类型（如 float、list 等）

    # 创建模型类，命名格式为设备名称首字母大写 + "Params"
    model_name = f"{device_name.capitalize()}Params"
    return create_model(model_name, **fields)


from typing import List


def _add_config_for_scenes(device_configs: List[dict], scenes: Scenes) -> Scenes:
    """
    为 scenes 中的设备添加对应的配置信息。
    :param device_configs: 设备配置列表，每个配置是一个字典
    :param scenes: Scenes 对象，包含场景和相关设备
    :return: 更新后的 Scenes 对象
    """
    # 构建设备 ID 到配置的映射
    # 直接使用字典推导式创建映射，提高效率并简化代码
    print(device_configs, type(device_configs))
    device_to_config_map = {
        config['product_id']['value']: config
        for config in device_configs
        if isinstance(config.get('product_id'), dict) and 'value' in config['product_id']
    }

    # 直接更新设备配置，避免多余的循环和条件检查
    for scene in scenes.scenes:
        for device in scene.involved_devices:
            device.config = device_to_config_map.get(device.id, {})

    return scenes


def init_state(state: SceneState):
    """
    初始化（主要是今天的日期）
    :param state:
    :return:
    """
    vector_store = rag_loder()

    docs: List[Document] = []
    for s in vector_store.get()['documents']:
        docs.append(Document(page_content=s))
    device_configs = docs
    date_now = datetime.now()
    return {
        "date": date_now,
        "device_configs": device_configs,
        "location": "中国四川省成都市郫都区"
    }


def generate_scenes(state: SceneState, config: RunnableConfig) -> Command[
    Literal["generate_scheme"]]:
    """
    生成接下来一天智能家居操控的各种场景。

    :param state:
    :param config:
    :return:
    """
    device_configs = state["device_configs"]
    today_date: datetime = state["date"]
    configurable = Configuration.from_runnable_config(config)

    # model = ChatGroq(
    #     model="qwen-qwq-32b"
    # )
    model = get_model(
        model_provider=configurable.planner_provider,
        model_name=configurable.planner_model
    )
    factory = DeviceModelFactory()
    factory.generate_all(device_configs)
    ConfigUnion = factory.get_union_type()
    ScenesDynamic = Scenes[ConfigUnion]
    ScenesDynamic.__name__ = "ScenesDynamic"
    model_with_structured_output = model.with_structured_output(ScenesDynamic)
    print(scheme_plan_prompt)
    query = scheme_plan_prompt.format(
        date=today_date,
        device_configs=device_configs,
        location=state['location']
    )
    scenes: Scenes = model_with_structured_output.invoke([
        SystemMessage(content=query)
    ])
    # scenes = _add_config_for_scenes(device_configs, response)
    print("---------------------生成的场景的数量--------------------------------")
    print(len(scenes.scenes))
    return Command(
        update={
            "scenes": scenes,
        },
        goto=[
            Send("generate_scheme", {
                "scene": s,
                "device_configs": state['device_configs'],
                "date": state['date'],
                "location": state['location'],
            },
                 )
            for s in scenes.scenes
        ]
    )


# 从下面开始就都是子图的节点了
# ======================================================================================================
def generate_queries(state: SchemePlanState, config: RunnableConfig):
    """
    根据已经生成的场景和是否需要额外信息来进行判断，如果需要额外信息就编写网络搜索词条来获取准确参数
    搜集完成或者不收集之后就将所有信息汇总由规划模型生成合适的参数信息
    :param state:
    :param config:
    :return:
    """
    configurable = Configuration.from_runnable_config(config)
    scene: Scene = state['scene']
    model = get_model(
        model_provider=configurable.planner_provider,
        model_name=configurable.planner_model
    )
    device_configs = state["device_configs"]
    today_date: datetime = state["date"]
    model_with_structured_output = model.with_structured_output(Queries)
    query = query_gen_prompt.format(
        scene=scene,
        location=state['location'],
        device_configs=[device.config for device in scene.involved_devices],
    )
    response = model_with_structured_output.invoke([SystemMessage(content=query)])
    print(response)
    return {
        "search_queries": response.queries
    }


async def search_web(state: SchemePlanState, config: RunnableConfig):
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
    configurable = Configuration.from_runnable_config(config)
    # search_api = get_config_value(configurable.search_api)
    search_api = "tavily"
    search_api_config = configurable.search_api_config or {}  # 获取配置字典，默认为空
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
        "source_strs": [source_str]
    }


def design_scheme(state: SchemePlanState, config: RunnableConfig):
    """
    结合场景和搜寻到的信息综合设计设备调用方案
    """
    source_strs = state['source_strs']
    scene = state['scene']
    configurable = Configuration.from_runnable_config(config)
    model = get_model(
        model_provider=configurable.planner_provider,
        model_name=configurable.planner_model
    )
    factory = DeviceModelFactory()
    factory.generate_all(state['device_configs'])
    ConfigUnion = factory.get_union_type()
    SchemeDynamic = Scheme[ConfigUnion]
    SchemeDynamic.__name__ = "SchemeDynamic"
    model_with_structured_output = model.with_structured_output(SchemeDynamic)
    query = scheme_gen_prompt.format(
        scene=scene,
        source_strs=source_strs
    )
    response = model_with_structured_output.invoke([SystemMessage(content=query)])
    return {
        "scheme": response,
        "schemes_list": [response]
    }


# ======================================================================================================
def gather_completed_schemes(state: SceneState):
    """将已完成部分格式化为上下文以撰写最终部分。

    此节点获取所有已完成的研究部分，并将它们格式化为单个上下文字符串，用于撰写总结部分。

    参数：
        state: 当前状态，包含已完成部分

    返回：
        包含格式化部分的上下文字典
    """
    # 已完成部分列表
    completed_schemes = state['schemes_list']

    schemes: Schemes = Schemes(schemes=completed_schemes)
    cache = RedisSchemeCache()
    cache.save_schemes(schemes, target_date=date.today())
    return {"schemes_list": completed_schemes}
