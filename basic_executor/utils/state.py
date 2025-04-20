from langchain_core.documents import Document
from typing_extensions import List, TypedDict, Annotated
from langgraph.graph.message import MessagesState

from HomeBuddyAgent.utils.state import reduce_device_results
from common.structs import DeviceModelFactory, DeviceCalls, ConfigT, DeviceResult


class State(MessagesState):
    question: str
    device_configs: List[Document]
    answer: str
    feed_back: bool
    factory: DeviceModelFactory
    device_calls: DeviceCalls[ConfigT]
    device_call_results: Annotated[List[DeviceResult], reduce_device_results]

class StateInput(TypedDict):
    question: str


class StateOutput(TypedDict):
    answer: str
