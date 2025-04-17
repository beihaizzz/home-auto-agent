from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph.message import MessagesState


class State(MessagesState):
    question: str
    device_configs: List[Document]
    answer: str
    tool_using: bool


class StateInput(TypedDict):
    question: str


class StateOutput(TypedDict):
    answer: str
