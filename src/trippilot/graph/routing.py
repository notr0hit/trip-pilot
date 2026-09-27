"""Conditional edge routing functions for the TripPilot StateGraph."""

from __future__ import annotations

from typing import Literal

from trippilot.config import DEFAULT_MAX_REVISIONS
from trippilot.state import TripState


def route_after_audit(
    state: TripState,
) -> Literal["targeted_replan_search", "human_approval"]:
    """Loop back to self-correct when audits fail and revision budget remains."""
    issues = state.get("validation_issues") or []
    revision_count = int(state.get("revision_count", 0))
    max_revisions = int(state.get("max_revisions", DEFAULT_MAX_REVISIONS))

    if issues and revision_count < max_revisions:
        return "targeted_replan_search"
    return "human_approval"


def route_after_approval(
    state: TripState,
) -> Literal["plan_itinerary", "finalize_itinerary"]:
    """Route to re-planning or finalization based on the human HITL decision."""
    decision = state.get("human_decision") or {}
    action = str(decision.get("action", "approve")).lower().strip()
    if action == "replan":
        return "plan_itinerary"
    return "finalize_itinerary"

