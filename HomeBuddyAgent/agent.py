from HomeBuddyAgent.utils.nodes import generate, agent, filter, command_router, generate_queries, search_web
from HomeBuddyAgent.utils.mcp_integration import mcp_device_call
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

# 添加设备调用节点 - 使用MCP协议调用外部设备控制服务
# 使用标准的call_tool方法与对方MCP Server通信
workflow.add_node("call_devices", mcp_device_call)

# 变为固定的节点
workflow.add_node("retriever", retriever)  # retrieval
workflow.add_node("additional_info_collect", info_graph)

workflow.add_node(command_router)
workflow.add_node("executor", executor.compile())

workflow.add_node(
    "generate", generate
)

workflow.add_edge(START, "filter")
workflow.add_edge("filter", "agent")


workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retriever",
        END: END,
    },
)
workflow.add_edge("retriever", "command_router")
workflow.add_edge("additional_info_collect", "generate")

# MCP协议工作流：executor生成设备控制指令后，调用外部MCP Server
workflow.add_edge("executor", "call_devices")

# 设备调用完成后生成最终回复
workflow.add_edge("call_devices", "generate")

# 生成完成后结束
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()