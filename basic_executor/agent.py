
from langgraph.constants import START
from langgraph.graph import StateGraph, END
from basic_executor.utils.nodes import generate, should_continue
from basic_executor.utils.state import State,StateOutput
from common.configuration import Configuration
from HomeBuddyAgent.utils.mcp_integration import mcp_device_call


graph_builder = StateGraph(State, config_schema=Configuration,output=StateOutput)

graph_builder.add_node("generate", generate)
# 使用 MCP 协议调用外部设备控制服务，替换原有的 device_call
graph_builder.add_node("call_devices", mcp_device_call)

graph_builder.add_conditional_edges(
    "generate",
    should_continue,
    {
        "call_devices": "call_devices",  # should_continue 返回 "call_devices" 时跳转
        "end": END
    },
)

graph_builder.add_edge(START, "generate")
# 设备调用完成后回到 generate 生成最终回复
graph_builder.add_edge("call_devices", "generate")
graph = graph_builder.compile()
