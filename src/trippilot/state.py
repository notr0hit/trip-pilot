"""Typed Pydantic contracts and LangGraph state definition for TripPilot."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


class GeoLocation(BaseModel):
    """Resolved destination metadata from Open-Meteo Geocoding."""

    name: str
    country: str = ""
    admin1: str = ""
    latitude: float
    longitude: float
    timezone: str = "auto"


class DayWeather(BaseModel):
    """Daily weather forecast or seasonal proxy for one trip day."""

    date: str
    weather_code: int
    condition_label: str
    temp_min_c: float
    temp_max_c: float
    precip_prob_max: int = Field(ge=0, le=100)
    precip_sum_mm: float = 0.0
    is_rain_risk: bool
    source_mode: Literal["live_forecast", "seasonal_proxy", "simulated"] = "live_forecast"


class SearchSnippet(BaseModel):
    """Web grounding snippet returned by DuckDuckGo."""

    query: str
    title: str
    url: str
    snippet: str
    category: Literal["general", "indoor_backup", "budget"] = "general"


class ActivityItem(BaseModel):
    """Single scheduled block in a day's itinerary.

    estimated_cost represents the total estimated spend for the entire travel party.
    """

    title: str
    time_slot: Literal["morning", "afternoon", "evening", "lodging"]
    category: Literal["outdoor", "indoor", "dining", "transit", "lodging"]
    estimated_cost: float = Field(ge=0.0)
    duration_hours: float = Field(gt=0.0, le=24.0, default=2.0)
    weather_safe_if_rain: bool = False
    notes: str = ""
    source_url: str | None = None


class DayPlan(BaseModel):
    """Planned itinerary for a single day."""

    day_number: int = Field(ge=1)
    date: str
    theme: str
    activities: list[ActivityItem] = Field(min_length=1)


class ItineraryDraft(BaseModel):
    """Structured multi-day itinerary produced by the planner node."""

    destination: str
    currency: str = "INR"
    travelers: int = Field(ge=1, default=2)
    days: list[DayPlan] = Field(min_length=1)
    rationale: str = ""


class CategorySpend(BaseModel):
    """Spend aggregation for a single activity category."""

    category: str
    amount: float
    share_pct: float


class DaySpend(BaseModel):
    """Spend aggregation for one trip day."""

    day_number: int
    date: str
    amount: float


class CostDriver(BaseModel):
    """High-cost line item surfaced for self-correction and user inspection."""

    day_number: int
    title: str
    category: str
    estimated_cost: float


class BudgetReport(BaseModel):
    """Deterministic Python budget audit output."""

    currency: str
    budget_limit: float
    total_estimated: float
    variance: float
    overrun_pct: float
    is_within_budget: bool
    by_day: list[DaySpend]
    by_category: list[CategorySpend]
    top_cost_drivers: list[CostDriver]


class WeatherConflict(BaseModel):
    """Detected mismatch between rain risk and a non-rain-safe outdoor activity."""

    day_number: int
    date: str
    condition_label: str
    precip_prob_max: int
    conflicting_activities: list[str]
    reason: str


class WeatherAudit(BaseModel):
    """Deterministic Python weather compatibility audit."""

    has_conflicts: bool
    conflicts: list[WeatherConflict]
    checked_days: int


class RevisionSnapshot(BaseModel):
    """Captured draft + audit state at a specific revision for diff inspection."""

    revision_number: int
    draft: ItineraryDraft
    budget_report: BudgetReport
    weather_audit: WeatherAudit
    issues_addressed: list[str] = Field(default_factory=list)


class ToolCallLog(BaseModel):
    """Structured record of a single tool invocation."""

    tool_name: str
    args_summary: str
    result_summary: str


class TraceEvent(BaseModel):
    """Structured node execution record for timeline inspection."""

    timestamp: str
    node: str
    status: Literal["ok", "warning", "loop_back", "interrupted", "resumed"] = "ok"
    summary: str
    tool_calls: list[ToolCallLog] = Field(default_factory=list)
    duration_ms: int = 0


class TripState(TypedDict, total=False):
    """LangGraph state dictionary persisted across checkpoints."""

    # Request inputs
    destination: str
    origin: str
    start_date: str
    num_days: int
    total_budget: float
    currency: str
    travelers: int
    preferences: list[str]
    pace: Literal["relaxed", "balanced", "packed"]
    provider: str
    model_name: str
    api_key: str
    rain_prob_threshold: int
    max_revisions: int
    simulate_rain_day: int | None

    # Grounding context
    geo_location: dict[str, Any]
    weather_forecast: list[dict[str, Any]]
    weather_mode: str
    search_notes: list[dict[str, Any]]

    # Planning & verification loop
    draft_itinerary: dict[str, Any]
    budget_report: dict[str, Any]
    weather_audit: dict[str, Any]
    validation_issues: list[str]
    revision_count: int
    revisions: list[dict[str, Any]]

    # HITL & finalization
    human_decision: dict[str, Any] | None
    final_summary: str
    packing_tips: list[str]
    status: str

    # Append-only execution trace
    execution_trace: Annotated[list[dict[str, Any]], operator.add]

