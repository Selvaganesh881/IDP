from __future__ import annotations

from langgraph.graph import END, StateGraph

from pipeline.nodes import (
    extract_llm_node,
    hash_and_check_cache_node,
    ingest_document_node,
    mask_pii_node,
    unmask_pii_node,
)
from pipeline.state import Pipeline_State


def build_pipeline_graph():
    """Compiles the LangGraph for the Intelligent Document Processing PoC."""

    # 1. Initialize the Graph with our specific TypedDict State
    workflow = StateGraph(Pipeline_State)

    # 2. Add all executable nodes
    workflow.add_node("check_cache", hash_and_check_cache_node)
    workflow.add_node("ingest", ingest_document_node)
    workflow.add_node("mask", mask_pii_node)
    workflow.add_node("extract", extract_llm_node)
    workflow.add_node("unmask", unmask_pii_node)

    # 3. Define the precise execution routing
    workflow.set_entry_point("check_cache")

    # Routing logic: If cache hit, skip directly to end. Otherwise, run the pipeline.
    workflow.add_conditional_edges(
        "check_cache",
        lambda state: "end" if state.get("cache_hit") else "process",
        {"end": END, "process": "ingest"},
    )

    # Standard linear flow for processing
    workflow.add_edge("ingest", "mask")
    workflow.add_edge("mask", "extract")
    workflow.add_edge("extract", "unmask")
    workflow.add_edge("unmask", END)

    # 4. Compile the graph into an executable application
    app = workflow.compile()

    return app
