"""StateGraph assembly, SQLite checkpointing, and execution helpers for TripPilot."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from trippilot.config import DEFAULT_DB_PATH
from trippilot.graph.nodes import (
    audit_plan,
    finalize_itinerary,
    gather_grounding,
    human_approval,
    plan_itinerary,
    targeted_replan_search,
)
from trippilot.graph.routing import route_after_approval, route_after_audit
from trippilot.state import TripState


def build_workflow() -> StateGraph:
    """Construct the uncompiled TripPilot StateGraph with self-correction and HITL edges."""
    builder = StateGraph(TripState)

    builder.add_node("gather_grounding", gather_grounding)
    builder.add_node("plan_itinerary", plan_itinerary)
    builder.add_node("audit_plan", audit_plan)
    builder.add_node("targeted_replan_search", targeted_replan_search)
    builder.add_node("human_approval", human_approval)
    builder.add_node("finalize_itinerary", finalize_itinerary)

    builder.add_edge(START, "gather_grounding")
    builder.add_edge("gather_grounding", "plan_itinerary")
    builder.add_edge("plan_itinerary", "audit_plan")

    builder.add_conditional_edges(
        "audit_plan",
        route_after_audit,
        {
            "targeted_replan_search": "targeted_replan_search",
            "human_approval": "human_approval",
        },
    )
    builder.add_edge("targeted_replan_search", "plan_itinerary")

    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "plan_itinerary": "plan_itinerary",
            "finalize_itinerary": "finalize_itinerary",
        },
    )
    builder.add_edge("finalize_itinerary", END)
    return builder


def compile_graph(checkpointer: Any | None = None) -> Any:
    """Compile the TripPilot graph with the supplied checkpointer (defaults to InMemorySaver)."""
    cp = checkpointer if checkpointer is not None else InMemorySaver()
    return build_workflow().compile(checkpointer=cp)


@contextmanager
def open_sqlite_graph(db_path: str | Path = DEFAULT_DB_PATH) -> Iterator[Any]:
    """Open a thread-safe SQLite-backed checkpointer and yield a compiled LangGraph."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    path_str = str(db_path)
    if path_str != ":memory:":
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path_str, check_same_thread=False)
    try:
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        yield compile_graph(checkpointer=checkpointer)
    finally:
        conn.close()


def get_mermaid_diagram() -> str:
    """Return LangGraph's Mermaid graph definition for documentation and UI display."""
    return build_workflow().compile().get_graph().draw_mermaid()

