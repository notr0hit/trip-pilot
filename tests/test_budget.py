"""Unit tests for the deterministic Python budget calculator."""

from __future__ import annotations

from trippilot.state import ActivityItem, DayPlan, ItineraryDraft
from trippilot.tools.budget import calculate_budget, calculate_trip_budget_tool


def _sample_draft(costs: list[float]) -> ItineraryDraft:
    return ItineraryDraft(
        destination="Goa, India",
        currency="INR",
        travelers=2,
        days=[
            DayPlan(
                day_number=1,
                date="2026-12-01",
                theme="Arrival",
                activities=[
                    ActivityItem(
                        title="Heritage Walk",
                        time_slot="morning",
                        category="outdoor",
                        estimated_cost=costs[0],
                    ),
                    ActivityItem(
                        title="Goan Dinner",
                        time_slot="evening",
                        category="dining",
                        estimated_cost=costs[1],
                    ),
                ],
            ),
            DayPlan(
                day_number=2,
                date="2026-12-02",
                theme="Explore",
                activities=[
                    ActivityItem(
                        title="Boutique Resort",
                        time_slot="lodging",
                        category="lodging",
                        estimated_cost=costs[2],
                    ),
                ],
            ),
        ],
    )


def test_calculate_budget_within_limit() -> None:
    draft = _sample_draft([1500.0, 2000.0, 6500.0])
    report = calculate_budget(draft, budget_limit=10000.0)

    assert report.total_estimated == 10000.0
    assert report.variance == 0.0
    assert report.overrun_pct == 0.0
    assert report.is_within_budget is True
    assert [d.amount for d in report.by_day] == [3500.0, 6500.0]
    assert report.top_cost_drivers[0].title == "Boutique Resort"


def test_calculate_budget_catches_overrun_and_shares() -> None:
    draft = _sample_draft([2000.0, 2500.0, 8000.0])
    report = calculate_budget(draft, budget_limit=10000.0)

    assert report.total_estimated == 12500.0
    assert report.variance == 2500.0
    assert report.overrun_pct == 25.0
    assert report.is_within_budget is False
    categories = {c.category: c for c in report.by_category}
    assert categories["lodging"].amount == 8000.0
    assert categories["lodging"].share_pct == 64.0


def test_calculate_trip_budget_tool_wrapper() -> None:
    draft = _sample_draft([1000.0, 1500.0, 4000.0])
    result = calculate_trip_budget_tool.invoke(
        {"itinerary_json": draft.model_dump(), "budget_limit": 7000.0}
    )
    assert result["total_estimated"] == 6500.0
    assert result["is_within_budget"] is True

