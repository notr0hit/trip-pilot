"""End-to-end LangGraph tests for conditional self-correction, cycle bounds, and SQLite HITL."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from langgraph.types import Command

from trippilot.graph.routing import route_after_approval, route_after_audit
from trippilot.graph.workflow import get_mermaid_diagram, open_sqlite_graph
from trippilot.state import (
    ActivityItem,
    DayPlan,
    DayWeather,
    GeoLocation,
    ItineraryDraft,
    SearchSnippet,
)


def _mock_grounding():
    geo = GeoLocation(name="Goa", country="India", latitude=15.49, longitude=73.82, timezone="Asia/Kolkata")
    forecast = [
        DayWeather(
            date="2026-11-27",
            weather_code=0,
            condition_label="Clear sky",
            temp_min_c=23.0,
            temp_max_c=31.0,
            precip_prob_max=10,
            is_rain_risk=False,
        ),
        DayWeather(
            date="2026-11-28",
            weather_code=65,
            condition_label="Heavy rain",
            temp_min_c=22.0,
            temp_max_c=27.0,
            precip_prob_max=85,
            is_rain_risk=True,
        ),
    ]
    snippets = [
        SearchSnippet(
            query="goa indoor",
            title="Goa State Museum",
            url="https://example.com/goa-museum",
            snippet="Indoor heritage exhibits.",
            category="indoor_backup",
        )
    ]
    return geo, forecast, snippets


class _FakeStructuredLLM:
    """Deterministic fake chat model whose drafts respond to revision_count & feedback."""

    def __init__(self, always_over_budget: bool = False) -> None:
        self.calls = 0
        self.always_over_budget = always_over_budget

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls += 1
        prompt_text = "\n".join(str(m.content) for m in messages)
        is_replan = "SELF-CORRECTION REQUIRED" in prompt_text
        has_human_feedback = "HUMAN-IN-THE-LOOP REVIEWER FEEDBACK" in prompt_text

        if self.always_over_budget or (not is_replan and not has_human_feedback):
            day2_act = ActivityItem(
                title="Kayaking Tour",
                time_slot="morning",
                category="outdoor",
                estimated_cost=9000.0,
                weather_safe_if_rain=False,
            )
            stay_cost = 6000.0
        else:
            day2_act = ActivityItem(
                title="Goa State Museum & Food Walk" if has_human_feedback else "Goa State Museum",
                time_slot="morning",
                category="indoor",
                estimated_cost=1500.0,
                weather_safe_if_rain=True,
                source_url="https://example.com/goa-museum",
            )
            stay_cost = 3500.0

        return ItineraryDraft(
            destination="Goa, India",
            currency="INR",
            travelers=2,
            days=[
                DayPlan(
                    day_number=1,
                    date="2026-11-27",
                    theme="Arrival",
                    activities=[
                        ActivityItem(
                            title="Fort Aguada Walk",
                            time_slot="afternoon",
                            category="outdoor",
                            estimated_cost=1500.0,
                            weather_safe_if_rain=False,
                        ),
                        ActivityItem(
                            title="Beachside Stay",
                            time_slot="lodging",
                            category="lodging",
                            estimated_cost=stay_cost,
                            weather_safe_if_rain=True,
                        ),
                    ],
                ),
                DayPlan(
                    day_number=2,
                    date="2026-11-28",
                    theme="Explore",
                    activities=[day2_act],
                ),
            ],
            rationale="Test plan",
        )


def test_routing_edges_unit() -> None:
    assert route_after_audit({"validation_issues": ["overrun"], "revision_count": 0, "max_revisions": 3}) == "targeted_replan_search"
    assert route_after_audit({"validation_issues": ["overrun"], "revision_count": 3, "max_revisions": 3}) == "human_approval"
    assert route_after_audit({"validation_issues": [], "revision_count": 0, "max_revisions": 3}) == "human_approval"

    assert route_after_approval({"human_decision": {"action": "approve"}}) == "finalize_itinerary"
    assert route_after_approval({"human_decision": {"action": "replan"}}) == "plan_itinerary"


def test_mermaid_diagram_contains_core_nodes() -> None:
    mermaid = get_mermaid_diagram()
    for node_name in [
        "gather_grounding",
        "plan_itinerary",
        "audit_plan",
        "targeted_replan_search",
        "human_approval",
        "finalize_itinerary",
    ]:
        assert node_name in mermaid


def test_graph_self_corrects_pauses_hitl_and_resumes_from_sqlite(tmp_path: Path) -> None:
    geo, forecast, snippets = _mock_grounding()
    fake_llm = _FakeStructuredLLM()
    db_file = tmp_path / "checkpoints.db"

    initial_state = {
        "destination": "Goa, India",
        "start_date": "2026-11-27",
        "num_days": 2,
        "total_budget": 10000.0,
        "currency": "INR",
        "travelers": 2,
        "preferences": ["heritage"],
        "pace": "balanced",
        "provider": "gemini",
        "max_revisions": 3,
        "execution_trace": [],
    }
    config = {"configurable": {"thread_id": "test-thread-1", "llm": fake_llm}}

    with (
        patch("trippilot.graph.nodes.geocode_destination", return_value=geo),
        patch("trippilot.graph.nodes.fetch_weather_forecast", return_value=(forecast, "live_forecast")),
        patch("trippilot.graph.nodes.gather_destination_snippets", return_value=(snippets, [])),
        patch("trippilot.graph.nodes.gather_replan_snippets", return_value=(snippets, [])),
    ):
        # Phase 1: Run until HITL interrupt
        with open_sqlite_graph(db_file) as graph:
            paused = graph.invoke(initial_state, config=config)
            assert "__interrupt__" in paused
            snap = graph.get_state(config)
            assert snap.values["revision_count"] == 1
            assert snap.values["validation_issues"] == []
            assert snap.values["budget_report"]["is_within_budget"] is True
            assert len(snap.values["revisions"]) == 2

        # Phase 2: Re-open SQLite in a fresh context (simulates Streamlit rerun) and resume with replan feedback
        with open_sqlite_graph(db_file) as graph:
            paused_again = graph.invoke(
                Command(resume={"action": "replan", "feedback": "Add food walk", "new_budget": 12000.0}),
                config=config,
            )
            assert "__interrupt__" in paused_again

            # Phase 3: Approve and finalize
            final_state = graph.invoke(Command(resume={"action": "approve"}), config=config)

    assert final_state["status"] == "finalized"
    assert "final_summary" in final_state
    assert final_state["budget_report"]["total_estimated"] <= 12000.0
    trace_statuses = [ev["status"] for ev in final_state["execution_trace"]]
    assert "loop_back" in trace_statuses
    assert "resumed" in trace_statuses


def test_graph_respects_max_revisions_cap(tmp_path: Path) -> None:
    geo, forecast, snippets = _mock_grounding()
    stuck_llm = _FakeStructuredLLM(always_over_budget=True)
    db_file = tmp_path / "cap.db"
    config = {"configurable": {"thread_id": "cap-thread", "llm": stuck_llm}}

    with (
        patch("trippilot.graph.nodes.geocode_destination", return_value=geo),
        patch("trippilot.graph.nodes.fetch_weather_forecast", return_value=(forecast, "live_forecast")),
        patch("trippilot.graph.nodes.gather_destination_snippets", return_value=(snippets, [])),
        patch("trippilot.graph.nodes.gather_replan_snippets", return_value=(snippets, [])),
        open_sqlite_graph(db_file) as graph,
    ):
        paused = graph.invoke(
            {
                "destination": "Goa, India",
                "start_date": "2026-11-27",
                "num_days": 2,
                "total_budget": 10000.0,
                "currency": "INR",
                "travelers": 2,
                "provider": "gemini",
                "max_revisions": 2,
                "execution_trace": [],
            },
            config=config,
        )
        assert "__interrupt__" in paused
        snap = graph.get_state(config)
        assert snap.values["revision_count"] == 2
        assert len(snap.values["validation_issues"]) >= 1
        assert len(snap.values["revisions"]) == 3


def test_demo_mode_runs_full_self_correction_without_api_key(tmp_path: Path) -> None:
    geo, forecast, snippets = _mock_grounding()
    db_file = tmp_path / "demo.db"
    config = {"configurable": {"thread_id": "demo-thread"}}

    with (
        patch("trippilot.graph.nodes.geocode_destination", return_value=geo),
        patch("trippilot.graph.nodes.fetch_weather_forecast", return_value=(forecast, "live_forecast")),
        patch("trippilot.graph.nodes.gather_destination_snippets", return_value=(snippets, [])),
        patch("trippilot.graph.nodes.gather_replan_snippets", return_value=(snippets, [])),
        open_sqlite_graph(db_file) as graph,
    ):
        paused = graph.invoke(
            {
                "destination": "Goa, India",
                "start_date": "2026-11-27",
                "num_days": 2,
                "total_budget": 18000.0,
                "currency": "INR",
                "travelers": 2,
                "provider": "demo",
                "max_revisions": 3,
                "execution_trace": [],
            },
            config=config,
        )
        assert "__interrupt__" in paused
        snap = graph.get_state(config)
        # Demo rev 0 intentionally fails weather + budget, then rev 1 fixes both
        assert snap.values["revision_count"] == 1
        assert snap.values["validation_issues"] == []
        final_state = graph.invoke(Command(resume={"action": "approve"}), config=config)
        assert final_state["status"] == "finalized"


def test_llm_planner_json_repair_fallback() -> None:
    from langchain_core.messages import AIMessage
    from trippilot.graph.nodes import _invoke_llm_planner

    valid_json = (
        '<think>planning...</think>{"destination":"Miami, USA","currency":"USD","travelers":1,'
        '"rationale":"Compact plan","days":[{"day_number":1,"date":"2026-11-27","theme":"Beach",'
        '"activities":[{"title":"South Beach Walk","time_slot":"morning","category":"outdoor",'
        '"estimated_cost":50.0,"duration_hours":2.0,"weather_safe_if_rain":false,"notes":"Walk","source_url":null}]}]}'
    )

    class _FlakyStructuredLLM:
        def __init__(self) -> None:
            self.raw_calls = 0

        def with_structured_output(self, schema, **kwargs):
            raise RuntimeError("tool schema unsupported")

        def invoke(self, messages):
            self.raw_calls += 1
            if self.raw_calls == 1:
                return AIMessage(content='{"destination": "Miami, USA", "days": [')
            return AIMessage(content=valid_json)

    flaky = _FlakyStructuredLLM()
    draft = _invoke_llm_planner(
        flaky,
        {
            "destination": "Miami, USA",
            "start_date": "2026-11-27",
            "num_days": 1,
            "total_budget": 1000.0,
            "currency": "USD",
            "travelers": 1,
            "weather_forecast": [],
            "search_notes": [],
        },
    )
    assert flaky.raw_calls == 2
    assert draft.destination == "Miami, USA"
    assert draft.days[0].activities[0].title == "South Beach Walk"

