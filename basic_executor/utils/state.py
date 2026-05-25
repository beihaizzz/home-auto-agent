from langchain_core.documents import Document
from typing_extensions import List, TypedDict, Annotated, Optional
from langgraph.graph.message import MessagesState

from HomeBuddyAgent.utils.state import reduce_device_results, reduce_feed_back
from common.structs import DeviceModelFactory, DeviceCalls, ConfigT, DeviceResult
from common.mcp.protocol import MCPResponse


class State(MessagesState):
    question: str
    device_configs: List[Document]
    answer: str
    feed_back: Annotated[bool, reduce_feed_back]
    factory: DeviceModelFactory
    device_calls: DeviceCalls[ConfigT]
    device_call_results: Annotated[List[DeviceResult], reduce_device_results]
    mcp_response: Optional[MCPResponse] = None

class StateInput(TypedDict):
    question: str


class StateOutput(TypedDict):
    answer: str
    device_calls: DeviceCalls[ConfigT]
