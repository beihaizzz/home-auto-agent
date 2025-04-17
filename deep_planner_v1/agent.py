from langgraph.constants import START, END
from langgraph.graph import StateGraph

from deep_planner_v1.utils.states import SchemePlanState, SchemeStateInput, SchemeStateOutPut, SchemePlanStateOutPut
from common.configuration import Configuration
from deep_planner_v1.utils.nodes import (
    init_state, generate_scenes, generate_queries, search_web, design_scheme, gather_completed_schemes
)

subgraph = StateGraph(
    SchemePlanState,
    output=SchemePlanStateOutPut  # 没有这个，在Langgraph studio中就显示不了子图
)
subgraph.add_node("generate_queries", generate_queries)
subgraph.add_node("search_web", search_web)
subgraph.add_node("design_scheme", design_scheme)
subgraph.add_edge(START, "generate_queries")
subgraph.add_edge("generate_queries", "search_web")
subgraph.add_edge("search_web", "design_scheme")
subgraph.add_edge("design_scheme", END)

builder = StateGraph(
    SchemePlanState,
    input=SchemeStateInput,
    output=SchemeStateOutPut,
    config_schema=Configuration
)

builder.add_node("init", init_state)
builder.add_node("generate_scenes", generate_scenes)
builder.add_node("generate_scheme", subgraph.compile())
builder.add_node("gather_completed_schemes", gather_completed_schemes)

builder.set_entry_point("init")
builder.add_edge("init", "generate_scenes")
# builder.add_edge("generate_scenes", "generate_scheme")
builder.add_edge("generate_scheme", "gather_completed_schemes")
builder.add_edge("gather_completed_schemes", END)
graph = builder.compile()
