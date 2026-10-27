"""
Task 8: Shared Agent State Schema for LangGraph Orchestration.
"""
from typing import List, Dict, Any, TypedDict


class AgentState(TypedDict):
    user_query: str
    search_queries: List[str]
    retrieved_docs: List[Dict[str, Any]]
    draft_answer: str
    validation_status: str
    validation_critique: str
    final_answer: str
    agent_trace: List[str]
