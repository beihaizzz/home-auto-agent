from HomeBuddyAgent.utils.nodes import agent, filter, command_router, generate_queries, search_web
from HomeBuddyAgent.utils.state import State, StateInput, StateOutput, InfoStateOutPut, InfoState
from HomeBuddyAgent.utils.tools import retriever_tool
from common.configuration import Configuration
from basic_executor.agent import graph_builder as executor
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

# 定义子图
info_graph = StateGraph(
    state_schema=InfoState,
    config_schema=Configuration,
    output=InfoStateOutPut
)
info_graph.add_node(generate_queries)
info_graph.add_node(search_web)
info_graph.add_edge(START, "generate_queries")
info_graph.add_edge("generate_queries", "search_web")
info_graph.add_edge("search_web", END)

info_graph = info_graph.compile()

# Define a new graph
workflow = StateGraph(State, config_schema=Configuration, input=StateInput, output=StateOutput)

workflow.add_node("filter", filter)
workflow.add_node("agent", agent)  # agent

retriever = ToolNode([retriever_tool])

# 变为固定的节点
workflow.add_node("retriever", retriever)  # retrieval
workflow.add_node("additional_info_collect", info_graph)
workflow.add_node(command_router)

# executor子图包含完整的设备控制逻辑（generate + call_devices）
workflow.add_node("executor", executor.compile())

workflow.add_edge(START, "filter")
workflow.add_edge("filter", "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retriever",
        END: END,
    },
)
workflow.add_edge("retriever", "command_router")
workflow.add_edge("additional_info_collect", "executor")

# command_router 通过 Send 对象控制路由，不需要额外的边

# executor完成后直接结束
workflow.add_edge("executor", END)

# Compile
graph = workflow.compile()