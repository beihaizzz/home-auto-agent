from datetime import datetime
from typing import Annotated, TypedDict, List
import operator

from langchain_core.documents import Document

from deep_planner_v1.utils.structs import Scene, Scenes,  Scheme
from common.structs import SearchQuery,ConfigT


class SchemePlanState(TypedDict):
    """
    子图的state
    """

    # 预设方案计划的state
    device_configs: List[Document]
    # schemes: list[Scheme]
    date: datetime
    location: str
    scene: Scene[ConfigT]  # 分配给每个方案的场景
    scheme: Scheme[ConfigT]
    schemes_list: Annotated[list[Scheme[ConfigT]], operator.add]  # 收集汇总所有的方案,Final key we duplicate in outer state for Send() API
    search_iterations: int  # 搜索的迭代次数
    search_queries: list[SearchQuery]
    source_str: str  # 格式化的web search源内容
    source_strs: list[str]


class SceneState(TypedDict):
    """
    暂时想不出起什么名字好，是父图的state
    """
    device_configs: List[Document]
    # schemes: list[Scheme]
    date: datetime
    location: str
    scenes: Scenes[ConfigT]  #所有的场景
    # search_iterations: int  # Number of search iterations done
    # search_queries: list[SearchQuery]  # List of search queries
    source_str: str  # String of formatted source content from web search
    source_strs: list[str]
    schemes_list: Annotated[list[Scheme[ConfigT]], operator.add]  # Send() API key,收集汇总所有的方案


class SchemeStateInput(TypedDict):
    any: str  # 任何值，后面会使用具体的输入，目前就是无论输入什么都会进行这个定时任务
    # device_configs: List[dict]
    # date: datetime


class SchemeStateOutPut(TypedDict):
    # schemes:List[Scheme]
    #中间输出测试
    # scenes: Scenes
    # queries: Queries
    # source_strs: List[str]
    schemes_list: Annotated[list[Scheme[ConfigT]], operator.add]


class SchemePlanStateOutPut(TypedDict):
    # schemes:List[Scheme]
    #中间输出测试
    # scenes: Scenes
    # queries: Queries
    # source_strs: List[str]
    schemes_list: Annotated[list[Scheme[ConfigT]], operator.add]
