"""Deterministic Python budget calculator and LangChain tool wrapper."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_core.tools import tool

from trippilot.state import (
    BudgetReport,
    CategorySpend,
    CostDriver,
    DaySpend,
    ItineraryDraft,
)


def calculate_budget(draft: ItineraryDraft, budget_limit: float) -> BudgetReport:
    """Deterministically compute itinerary totals, category shares, and overruns."""
    category_totals: dict[str, float] = defaultdict(float)
    by_day: list[DaySpend] = []
    all_drivers: list[CostDriver] = []

    total_estimated = 0.0
    for day in draft.days:
        day_total = 0.0
        for item in day.activities:
            cost = round(float(item.estimated_cost), 2)
            day_total += cost
            category_totals[item.category] += cost
            all_drivers.append(
                CostDriver(
                    day_number=day.day_number,
                    title=item.title,
                    category=item.category,
                    estimated_cost=cost,
                )
            )
        day_total = round(day_total, 2)
        total_estimated += day_total
        by_day.append(
            DaySpend(day_number=day.day_number, date=day.date, amount=day_total)
        )

    total_estimated = round(total_estimated, 2)
    limit = round(float(budget_limit), 2)
    variance = round(total_estimated - limit, 2)
    overrun_pct = round((variance / limit) * 100.0, 1) if limit > 0 else 0.0
    is_within_budget = total_estimated <= limit

    by_category = [
        CategorySpend(
            category=cat,
            amount=round(amount, 2),
            share_pct=round((amount / total_estimated) * 100.0, 1)
            if total_estimated > 0
            else 0.0,
        )
        for cat, amount in sorted(
            category_totals.items(), key=lambda pair: pair[1], reverse=True
        )
    ]

    top_cost_drivers = sorted(
        all_drivers, key=lambda driver: driver.estimated_cost, reverse=True
    )[:5]

    return BudgetReport(
        currency=draft.currency,
        budget_limit=limit,
        total_estimated=total_estimated,
        variance=variance,
        overrun_pct=overrun_pct,
        is_within_budget=is_within_budget,
        by_day=by_day,
        by_category=by_category,
        top_cost_drivers=top_cost_drivers,
    )


@tool
def calculate_trip_budget_tool(itinerary_json: dict[str, Any], budget_limit: float) -> dict[str, Any]:
    """Deterministically calculate total spend, per-day breakdown, and budget overrun."""
    draft = ItineraryDraft.model_validate(itinerary_json)
    return calculate_budget(draft, budget_limit).model_dump()

