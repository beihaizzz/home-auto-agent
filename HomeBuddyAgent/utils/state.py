from datetime import datetime
import operator
from typing import List, Annotated, Optional, Sequence, TypedDict

from langchain_chroma import Chroma
from langchain_core.messages import ToolMessage, AnyMessage
from langgraph.graph import MessagesState, add_messages
from langchain_core.documents import Document

from HomeBuddyAgent.utils.structs import AdditionalInfo
from common.structs import DeviceCall, DeviceResult, DeviceCalls, SearchQuery, ConfigT


def reduce_feed_back(existing: Optional[bool], new: Optional[bool]) -> bool:
    """归约 feed_back 的布尔值，取最后一个非 None 值或默认 False"""
    if new is not None:
        return new
    return existing if existing is not None else False


def reduce_tool_messages(existing: Sequence[ToolMessage], new: Sequence[ToolMessage]) -> Sequence[ToolMessage]:
    """合并并行分支的工具消息列表，确保消息按顺序追加"""
    return list(existing) + list(new)


def reduce_additional_info(existing: Sequence[List[AdditionalInfo]], new: Sequence[List[AdditionalInfo]]) -> Sequence[
    List[AdditionalInfo]]:
    """合并并行分支的工具消息列表，确保消息按顺序追加"""
    return list(existing) + list(new)


def reduce_device_results(existing: Sequence[List[DeviceResult]], new: Sequence[List[DeviceResult]]) -> Sequence[
    List[DeviceResult]]:
    """合并并行分支的工具消息列表，确保消息按顺序追加"""
    return list(existing) + list(new)


def reduce_device_calls(existing: Sequence[List[DeviceCall]], new: Sequence[List[DeviceCall]]) -> Sequence[
    List[DeviceCall]]:
    """合并并行分支的工具消息列表，确保消息按顺序追加"""
    return list(existing) + list(new)


class State(MessagesState):
    question: str
    device_configs: List[Document]  # 改为支持多值
    answer: str
    location: str  # 后续要添加到长期记忆或者配置信息中供修改
    time_now: datetime  # init的时候进行初始化
    feed_back: Annotated[bool, reduce_feed_back]
    device_call_results: Annotated[List[DeviceResult], reduce_device_results]
    device_calls: DeviceCalls[ConfigT]
    additional_info: Annotated[List[AdditionalInfo], reduce_additional_info]
    vector_store: Chroma


class InfoState(TypedDict):
    question: str
    device_configs: List[Document]  # 改为支持多值
    location: str  # 后续要添加到长期记忆或者配置信息中供修改
    time_now: datetime  # init的时候进行初始化
    sub_messages: Annotated[list[AnyMessage], add_messages]
    search_queries: List[SearchQuery]
    additional_info: Annotated[List[AdditionalInfo], reduce_additional_info]


class InfoStateOutPut(TypedDict):
    # 用于和父图对接的节点
    additional_info: Annotated[List[AdditionalInfo], reduce_additional_info]


class StateInput(TypedDict):
    question: str


class StateOutput(TypedDict):
    answer: str
