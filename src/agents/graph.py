"""
Task 8: LangGraph Multi-Agent Workflow Orchestrator.
Connects Planner -> Retriever -> Analyst -> Validator into a compiled state machine.
"""
from typing import Optional
from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.nodes import AgentNodes


def build_agent_graph(
    api_key: str,
    llm_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    provider_override: str = "auto",
):
    nodes = AgentNodes(
        api_key=api_key,
        llm_model=llm_model,
        embedding_model=embedding_model,
        provider_override=provider_override,
    )
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", nodes.planner_node)
    workflow.add_node("retriever", nodes.retriever_node)
    workflow.add_node("analyst", nodes.analyst_node)
    workflow.add_node("validator", nodes.validator_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "analyst")
    workflow.add_edge("analyst", "validator")
    workflow.add_edge("validator", END)

    return workflow.compile()
