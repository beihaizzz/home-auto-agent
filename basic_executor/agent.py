from typing import TypedDict, Literal

from langgraph.constants import START
from langgraph.graph import StateGraph, END
from basic_executor.utils.nodes import generate, should_continue, tool_node
from basic_executor.utils.state import State,StateOutput
from common.configuration import Configuration


graph_builder = StateGraph(State, config_schema=Configuration,output=StateOutput)

graph_builder.add_node("generate", generate)
graph_builder.add_node("action", tool_node)

graph_builder.add_conditional_edges(
    "generate",
    should_continue,
    {
        "continue": "action",
        "end": END
    },
)

graph_builder.add_edge(START, "generate")
graph_builder.add_edge("action", "generate")
graph = graph_builder.compile()
