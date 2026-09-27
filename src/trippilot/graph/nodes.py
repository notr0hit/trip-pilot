"""LangGraph node implementations for TripPilot."""

import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from trippilot.config import (
    DEFAULT_MAX_REVISIONS,
    DEFAULT_RAIN_PROB_THRESHOLD,
    build_chat_model,
    format_currency,
    resolve_provider,
)
from trippilot.state import (
    ActivityItem,
    BudgetReport,
    DayPlan,
    DayWeather,
    GeoLocation,
    ItineraryDraft,
    RevisionSnapshot,
    SearchSnippet,
    ToolCallLog,
    TraceEvent,
    TripState,
    WeatherAudit,
)
from trippilot.tools.budget import calculate_budget
from trippilot.tools.search import (
    gather_destination_snippets,
    gather_replan_snippets,
)
from trippilot.tools.weather import (
    audit_weather,
    fetch_weather_forecast,
    geocode_destination,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


def _make_trace(
    node: str,
    status: str,
    summary: str,
    started_at: float,
    tool_calls: list[ToolCallLog] | None = None,
) -> dict[str, Any]:
    elapsed_ms = max(1, int((time.perf_counter() - started_at) * 1000))
    return TraceEvent(
        timestamp=_now_iso(),
        node=node,
        status=status,  # type: ignore[arg-type]
        summary=summary,
        tool_calls=tool_calls or [],
        duration_ms=elapsed_ms,
    ).model_dump()


def gather_grounding(state: TripState) -> dict[str, Any]:
    """Geocode destination, fetch Open-Meteo forecast, and run initial DuckDuckGo search."""
    t0 = time.perf_counter()
    destination = state["destination"]
    start_date = state["start_date"]
    num_days = int(state.get("num_days", 3))
    currency = state.get("currency", "INR")
    preferences = state.get("preferences") or []
    rain_threshold = int(state.get("rain_prob_threshold", DEFAULT_RAIN_PROB_THRESHOLD))
    simulate_rain_day = state.get("simulate_rain_day")

    geo = geocode_destination(destination)
    forecast, weather_mode = fetch_weather_forecast(
        geo=geo,
        start_date=start_date,
        num_days=num_days,
        rain_prob_threshold=rain_threshold,
        simulate_rain_day=simulate_rain_day,
    )
    snippets, warnings = gather_destination_snippets(destination, preferences, currency)

    rainy_days = [f"Day {i + 1} ({d.date})" for i, d in enumerate(forecast) if d.is_rain_risk]
    tool_logs = [
        ToolCallLog(
            tool_name="geocode_destination",
            args_summary=f"destination='{destination}'",
            result_summary=f"{geo.name}, {geo.country} ({geo.latitude:.2f}, {geo.longitude:.2f})",
        ),
        ToolCallLog(
            tool_name="fetch_weather_forecast",
            args_summary=f"days={num_days}, start={start_date}, mode={weather_mode}",
            result_summary=(
                f"{len(forecast)} days loaded; rain risk on: {', '.join(rainy_days) or 'none'}"
            ),
        ),
        ToolCallLog(
            tool_name="web_search_places",
            args_summary=f"3 queries for '{destination}'",
            result_summary=f"{len(snippets)} snippets"
            + (f" ({len(warnings)} warnings)" if warnings else ""),
        ),
    ]

    summary = (
        f"Grounded {geo.name}, {geo.country} ({weather_mode}) with {len(forecast)}d weather "
        f"and {len(snippets)} web snippets."
    )
    if warnings:
        summary += f" Search warnings: {'; '.join(warnings)}"

    return {
        "geo_location": geo.model_dump(),
        "weather_forecast": [d.model_dump() for d in forecast],
        "weather_mode": weather_mode,
        "search_notes": [s.model_dump() for s in snippets],
        "revision_count": 0,
        "revisions": [],
        "validation_issues": [],
        "human_decision": None,
        "status": "planning",
        "execution_trace": [
            _make_trace(
                node="gather_grounding",
                status="warning" if warnings else "ok",
                summary=summary,
                started_at=t0,
                tool_calls=tool_logs,
            )
        ],
    }


def _extract_json_block(raw_text: str) -> str:
    """Extract JSON payload from reasoning tags, markdown fences, or raw LLM output."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        return fenced.group(1)
    braced = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if braced:
        return braced.group(1)
    return cleaned


def _normalize_draft_dates(
    draft: ItineraryDraft,
    start_date: str,
    num_days: int,
    currency: str,
    travelers: int,
) -> ItineraryDraft:
    """Ensure sequential day numbers and ISO dates matching the user's request."""
    base = date.fromisoformat(start_date)
    normalized_days: list[DayPlan] = []
    for idx in range(num_days):
        expected_date = (base + timedelta(days=idx)).isoformat()
        if idx < len(draft.days):
            day = draft.days[idx]
            normalized_days.append(
                DayPlan(
                    day_number=idx + 1,
                    date=expected_date,
                    theme=day.theme,
                    activities=day.activities,
                )
            )
    if not normalized_days:
        normalized_days = draft.days
    return ItineraryDraft(
        destination=draft.destination,
        currency=currency,
        travelers=travelers,
        days=normalized_days,
        rationale=draft.rationale,
    )


def _build_planner_prompts(state: TripState) -> tuple[str, str]:
    destination = state["destination"]
    start_date = state["start_date"]
    num_days = int(state.get("num_days", 3))
    total_budget = float(state.get("total_budget", 25000))
    currency = state.get("currency", "INR")
    travelers = int(state.get("travelers", 2))
    preferences = state.get("preferences") or ["culture", "food"]
    pace = state.get("pace", "balanced")
    revision_count = int(state.get("revision_count", 0))
    issues = state.get("validation_issues") or []
    human_decision = state.get("human_decision") or {}
    forecast = state.get("weather_forecast") or []
    snippets = state.get("search_notes") or []

    weather_lines = []
    for idx, dw in enumerate(forecast, start=1):
        risk = "RAIN RISK - AVOID UNCOVERED OUTDOOR ACTIVITIES" if dw.get("is_rain_risk") else "Clear/Good for outdoor"
        weather_lines.append(
            f"- Day {idx} ({dw['date']}): {dw['condition_label']}, "
            f"{dw['temp_min_c']}°C–{dw['temp_max_c']}°C, "
            f"precip prob {dw['precip_prob_max']}% [{risk}]"
        )

    snippet_lines = [
        f"- [{s.get('category', 'general')}] {s['title']} ({s.get('url', '')}): {str(s['snippet'])[:180]}"
        for s in snippets[:6]
    ]

    system_prompt = (
        "You are TripPilot, an expert travel planner building realistic, weather-safe, "
        "strictly budget-compliant multi-day itineraries. "
        "CRITICAL RULES:\n"
        f"1. Total sum of ALL estimated_cost values across ALL days MUST be <= {total_budget} {currency} "
        f"(estimated_cost is the combined spend for {travelers} travelers).\n"
        "2. Every activity MUST have category in {'outdoor','indoor','dining','transit','lodging'} "
        "and time_slot in {'morning','afternoon','evening','lodging'}.\n"
        "3. On any day marked RAIN RISK, NEVER schedule outdoor activities with weather_safe_if_rain=false; "
        "schedule indoor or covered alternatives instead.\n"
        "4. Include realistic lodging per night (except optionally the final departure day) and meals.\n"
        "5. Cite relevant URLs from the web snippets in source_url when applicable.\n"
        "6. Keep notes concise (under 15 words each) and schedule 3-4 activities per day so JSON stays compact."
    )

    feedback_block = ""
    if issues:
        feedback_block += (
            f"\n\nSELF-CORRECTION REQUIRED (Revision #{revision_count}):\n"
            "Your previous draft FAILED deterministic verification for these exact reasons:\n"
            + "\n".join(f"- {issue}" for issue in issues)
            + "\nFix ALL listed issues completely while preserving quality on unaffected days."
        )
    if human_decision.get("feedback"):
        feedback_block += (
            "\n\nHUMAN-IN-THE-LOOP REVIEWER FEEDBACK:\n"
            f"- Request: {human_decision['feedback']}\n"
            "Honor this feedback while keeping the itinerary within budget and rain-safe."
        )

    user_prompt = (
        f"Plan a {num_days}-day trip to {destination} starting {start_date}.\n"
        f"Travelers: {travelers} | Pace: {pace} | Interests: {', '.join(preferences)}\n"
        f"Strict Total Budget Ceiling: {format_currency(total_budget, currency)} ({total_budget} {currency})\n\n"
        f"WEATHER FORECAST:\n" + "\n".join(weather_lines) + "\n\n"
        f"WEB GROUNDING SNIPPETS:\n" + ("\n".join(snippet_lines) if snippet_lines else "- None")
        + feedback_block
    )
    return system_prompt, user_prompt


def _raw_text_from_response(raw: Any) -> str:
    content = raw.content if hasattr(raw, "content") else str(raw)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _invoke_llm_planner(llm: Any, state: TripState) -> ItineraryDraft:
    system_prompt, user_prompt = _build_planner_prompts(state)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    for struct_kwargs in ({}, {"method": "json_mode"}):
        try:
            structured = llm.with_structured_output(ItineraryDraft, **struct_kwargs)
            result = structured.invoke(messages)
            if isinstance(result, ItineraryDraft):
                return result
            return ItineraryDraft.model_validate(result)
        except Exception:
            continue

    compact_template = (
        '{"destination":"string","currency":"USD","travelers":1,"rationale":"short summary",'
        '"days":[{"day_number":1,"date":"YYYY-MM-DD","theme":"short theme","activities":['
        '{"title":"string","time_slot":"morning|afternoon|evening|lodging",'
        '"category":"outdoor|indoor|dining|transit|lodging","estimated_cost":100.0,'
        '"duration_hours":2.0,"weather_safe_if_rain":true,"notes":"short 10-word note","source_url":null}]}]}'
    )
    repair_note = ""
    last_exc: Exception | None = None
    for _ in range(2):
        fallback_msg = HumanMessage(
            content=(
                user_prompt
                + "\n\nReturn ONLY a single complete valid JSON object (no markdown, no extra text) "
                f"matching this exact shape:\n{compact_template}"
                + repair_note
            )
        )
        raw = llm.invoke([SystemMessage(content=system_prompt), fallback_msg])
        extracted = _extract_json_block(_raw_text_from_response(raw))
        try:
            return ItineraryDraft.model_validate_json(extracted)
        except Exception as exc:
            last_exc = exc
            repair_note = (
                f"\n\nYour previous JSON failed validation ({exc}). "
                "Keep notes under 8 words and ensure every bracket and brace is closed."
            )

    assert last_exc is not None
    raise last_exc


def _pick_snippet_url(snippets: list[dict[str, Any]], category: str) -> str | None:
    for item in snippets:
        if item.get("category") == category and item.get("url"):
            return str(item["url"])
    for item in snippets:
        if item.get("url"):
            return str(item["url"])
    return None


def _generate_demo_draft(state: TripState) -> ItineraryDraft:
    """Deterministic zero-key demo planner that showcases realistic self-correction."""
    destination = state["destination"]
    city_short = destination.split(",")[0].strip() or destination
    start_date = state["start_date"]
    num_days = int(state.get("num_days", 3))
    total_budget = float(state.get("total_budget", 25000))
    currency = state.get("currency", "INR")
    travelers = int(state.get("travelers", 2))
    preferences = state.get("preferences") or ["culture", "local food"]
    revision_count = int(state.get("revision_count", 0))
    forecast = state.get("weather_forecast") or []
    snippets = state.get("search_notes") or []
    human_decision = state.get("human_decision") or {}
    human_feedback = str(human_decision.get("feedback") or "").strip()

    base = date.fromisoformat(start_date)
    general_url = _pick_snippet_url(snippets, "general")
    indoor_url = _pick_snippet_url(snippets, "indoor_backup")
    budget_url = _pick_snippet_url(snippets, "budget")

    # Initial pass intentionally overshoots ~14% and places an outdoor tour on rainy days
    # so the reviewer/recruiter sees the deterministic guardrails catch and fix real issues.
    is_initial = revision_count == 0 and not human_feedback
    target_total = total_budget * (1.14 if is_initial else 0.92)
    per_day_target = round(target_total / max(1, num_days), 2)

    days: list[DayPlan] = []
    for idx in range(num_days):
        day_num = idx + 1
        d_str = (base + timedelta(days=idx)).isoformat()
        dw = forecast[idx] if idx < len(forecast) else {}
        is_rain = bool(dw.get("is_rain_risk", False))

        lodging_cost = round(per_day_target * 0.36, 2) if day_num < num_days else 0.0
        morning_cost = round(per_day_target * 0.24, 2)
        afternoon_cost = round(per_day_target * 0.22, 2)
        evening_cost = round(per_day_target - lodging_cost - morning_cost - afternoon_cost, 2)

        pref_focus = preferences[idx % len(preferences)].title()
        activities: list[ActivityItem] = []

        if day_num == 1:
            activities.append(
                ActivityItem(
                    title=f"{city_short} Old Quarter Heritage Walk",
                    time_slot="morning",
                    category="outdoor" if not (is_rain and not is_initial) else "indoor",
                    estimated_cost=morning_cost,
                    duration_hours=2.5,
                    weather_safe_if_rain=not is_initial and is_rain,
                    notes="Orientation walk covering landmark architecture and local markets.",
                    source_url=general_url,
                )
            )
        elif is_rain and not is_initial:
            activities.append(
                ActivityItem(
                    title=f"{city_short} Art Museum & Artisan Craft Workshop",
                    time_slot="morning",
                    category="indoor",
                    estimated_cost=morning_cost,
                    duration_hours=3.0,
                    weather_safe_if_rain=True,
                    notes="Swapped to rain-safe indoor museum & hands-on workshop due to rain risk.",
                    source_url=indoor_url,
                )
            )
        else:
            activities.append(
                ActivityItem(
                    title=f"{city_short} Coastal/Ridge Kayaking & Nature Trail ({pref_focus})",
                    time_slot="morning",
                    category="outdoor",
                    estimated_cost=morning_cost,
                    duration_hours=3.0,
                    weather_safe_if_rain=False,
                    notes="Guided outdoor excursion highlighting natural scenery.",
                    source_url=general_url,
                )
            )

        if is_rain and not is_initial:
            activities.append(
                ActivityItem(
                    title=f"Covered Spice Bazaar & Heritage Tea Lounge",
                    time_slot="afternoon",
                    category="indoor",
                    estimated_cost=afternoon_cost,
                    duration_hours=2.5,
                    weather_safe_if_rain=True,
                    notes="Sheltered afternoon cultural tasting and indoor gallery visit.",
                    source_url=indoor_url or budget_url,
                )
            )
        else:
            activities.append(
                ActivityItem(
                    title=f"{city_short} {pref_focus} Discovery & Local Market"
                    + (" (Boutique Private Tour)" if is_initial else " (Self-Guided)"),
                    time_slot="afternoon",
                    category="indoor" if is_rain and not is_initial else "outdoor",
                    estimated_cost=afternoon_cost,
                    duration_hours=2.5,
                    weather_safe_if_rain=not is_initial,
                    notes="Curated afternoon exploration tailored to trip interests.",
                    source_url=budget_url or general_url,
                )
            )

        evening_title = (
            f"Local Street Food Crawl & Acoustic Cafe ({human_feedback[:28]})"
            if human_feedback and day_num == min(2, num_days)
            else f"{city_short} Regional Dinner & Night Market"
        )
        activities.append(
            ActivityItem(
                title=evening_title,
                time_slot="evening",
                category="dining",
                estimated_cost=evening_cost,
                duration_hours=2.0,
                weather_safe_if_rain=True,
                notes="Indoor regional dining featuring local specialties.",
                source_url=budget_url,
            )
        )

        if lodging_cost > 0:
            activities.append(
                ActivityItem(
                    title=(
                        f"{city_short} Boutique Resort Stay"
                        if is_initial
                        else f"{city_short} Central Value Guesthouse"
                    ),
                    time_slot="lodging",
                    category="lodging",
                    estimated_cost=lodging_cost,
                    duration_hours=12.0,
                    weather_safe_if_rain=True,
                    notes="Well-reviewed central stay balancing comfort and budget.",
                    source_url=budget_url,
                )
            )

        days.append(
            DayPlan(
                day_number=day_num,
                date=d_str,
                theme=f"{pref_focus} & {'Rain-Safe Indoors' if (is_rain and not is_initial) else 'Local Highlights'}",
                activities=activities,
            )
        )

    rationale = (
        "Initial draft prioritizing marquee experiences before deterministic guardrails."
        if is_initial
        else "Self-corrected draft swapping rainy outdoor blocks to indoor backups and tightening lodging/tour spend."
    )
    return ItineraryDraft(
        destination=destination,
        currency=currency,
        travelers=travelers,
        days=days,
        rationale=rationale,
    )


def plan_itinerary(
    state: TripState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Generate or revise a structured multi-day ItineraryDraft."""
    t0 = time.perf_counter()
    configurable = (config or {}).get("configurable") or {}
    injected_llm = configurable.get("llm")

    revision_count = int(state.get("revision_count", 0))
    provider = resolve_provider(
        preferred=state.get("provider", "auto"),
        api_key_override=state.get("api_key", ""),
    )

    if injected_llm is not None:
        draft = _invoke_llm_planner(injected_llm, state)
        used_label = "injected-llm"
    elif provider == "demo":
        draft = _generate_demo_draft(state)
        used_label = "demo-engine"
    else:
        llm = build_chat_model(
            provider=provider,
            model_name=state.get("model_name", ""),
            api_key=state.get("api_key", ""),
        )
        draft = _invoke_llm_planner(llm, state)
        used_label = f"{provider}:{state.get('model_name') or 'default'}"

    draft = _normalize_draft_dates(
        draft=draft,
        start_date=state["start_date"],
        num_days=int(state.get("num_days", 3)),
        currency=state.get("currency", "INR"),
        travelers=int(state.get("travelers", 2)),
    )

    total_acts = sum(len(d.activities) for d in draft.days)
    summary = (
        f"Draft revision #{revision_count} created via {used_label} "
        f"({len(draft.days)} days, {total_acts} activities)."
    )
    return {
        "draft_itinerary": draft.model_dump(),
        "status": "auditing",
        "execution_trace": [
            _make_trace(
                node="plan_itinerary",
                status="ok",
                summary=summary,
                started_at=t0,
            )
        ],
    }


def audit_plan(state: TripState) -> dict[str, Any]:
    """Run deterministic Python budget and weather conflict verification."""
    t0 = time.perf_counter()
    draft = ItineraryDraft.model_validate(state["draft_itinerary"])
    forecast = [DayWeather.model_validate(item) for item in state.get("weather_forecast") or []]
    budget_limit = float(state.get("total_budget", 25000))
    revision_count = int(state.get("revision_count", 0))
    max_revisions = int(state.get("max_revisions", DEFAULT_MAX_REVISIONS))

    budget_report: BudgetReport = calculate_budget(draft, budget_limit)
    weather_audit: WeatherAudit = audit_weather(draft, forecast)

    issues: list[str] = [conflict.reason for conflict in weather_audit.conflicts]
    if not budget_report.is_within_budget:
        drivers = ", ".join(
            f"{d.title} ({format_currency(d.estimated_cost, budget_report.currency)})"
            for d in budget_report.top_cost_drivers[:3]
        )
        issues.append(
            f"Budget overrun: estimated {format_currency(budget_report.total_estimated, budget_report.currency)} "
            f"exceeds limit {format_currency(budget_report.budget_limit, budget_report.currency)} "
            f"by {format_currency(budget_report.variance, budget_report.currency)} "
            f"(+{budget_report.overrun_pct}%). Top cost drivers: {drivers}."
        )

    existing_revisions = list(state.get("revisions") or [])
    snapshot = RevisionSnapshot(
        revision_number=revision_count,
        draft=draft,
        budget_report=budget_report,
        weather_audit=weather_audit,
        issues_addressed=list(state.get("validation_issues") or []),
    ).model_dump()
    existing_revisions.append(snapshot)

    will_loop = bool(issues) and revision_count < max_revisions
    trace_status = "loop_back" if will_loop else ("warning" if issues else "ok")
    if not issues:
        summary = (
            f"Audit PASSED (rev #{revision_count}): weather-safe across {weather_audit.checked_days} days; "
            f"total {format_currency(budget_report.total_estimated, budget_report.currency)} "
            f"<= {format_currency(budget_report.budget_limit, budget_report.currency)}."
        )
    elif will_loop:
        summary = (
            f"Audit FAILED (rev #{revision_count}): {len(issues)} issue(s) found. "
            f"Triggering self-correction loop."
        )
    else:
        summary = (
            f"Audit warning at max_revisions={max_revisions}: {len(issues)} unresolved issue(s); "
            "escalating to Human-in-the-Loop."
        )

    tool_logs = [
        ToolCallLog(
            tool_name="calculate_budget",
            args_summary=f"limit={budget_report.budget_limit}",
            result_summary=f"total={budget_report.total_estimated}, pass={budget_report.is_within_budget}",
        ),
        ToolCallLog(
            tool_name="audit_weather",
            args_summary=f"days={weather_audit.checked_days}",
            result_summary=f"conflicts={len(weather_audit.conflicts)}",
        ),
    ]

    return {
        "budget_report": budget_report.model_dump(),
        "weather_audit": weather_audit.model_dump(),
        "validation_issues": issues,
        "revisions": existing_revisions,
        "status": "self_correcting" if will_loop else "awaiting_approval",
        "execution_trace": [
            _make_trace(
                node="audit_plan",
                status=trace_status,
                summary=summary,
                started_at=t0,
                tool_calls=tool_logs,
            )
        ],
    }


def targeted_replan_search(state: TripState) -> dict[str, Any]:
    """Fetch fresh indoor or budget alternatives on DuckDuckGo before re-planning."""
    t0 = time.perf_counter()
    new_rev = int(state.get("revision_count", 0)) + 1
    destination = state["destination"]
    currency = state.get("currency", "INR")
    weather_audit = state.get("weather_audit") or {}
    budget_report = state.get("budget_report") or {}

    need_indoor = bool(weather_audit.get("has_conflicts", False))
    need_budget = not bool(budget_report.get("is_within_budget", True))

    new_snippets, warnings = gather_replan_snippets(
        destination=destination,
        need_indoor=need_indoor,
        need_budget=need_budget,
        currency=currency,
    )

    existing = list(state.get("search_notes") or [])
    seen = {item.get("url") or item.get("title") for item in existing}
    added = 0
    for snip in new_snippets:
        key = snip.url or snip.title
        if key not in seen:
            seen.add(key)
            existing.append(snip.model_dump())
            added += 1

    focus_parts = []
    if need_indoor:
        focus_parts.append("indoor rainy-day backups")
    if need_budget:
        focus_parts.append("budget-friendly alternatives")
    focus_str = " & ".join(focus_parts) or "itinerary refinements"

    return {
        "revision_count": new_rev,
        "search_notes": existing,
        "status": "self_correcting",
        "execution_trace": [
            _make_trace(
                node="targeted_replan_search",
                status="loop_back",
                summary=f"Loop #{new_rev}: searched {focus_str} (+{added} snippets).",
                started_at=t0,
                tool_calls=[
                    ToolCallLog(
                        tool_name="web_search_places",
                        args_summary=f"need_indoor={need_indoor}, need_budget={need_budget}",
                        result_summary=f"+{added} new snippets"
                        + (f" ({len(warnings)} warnings)" if warnings else ""),
                    )
                ],
            )
        ],
    }


def human_approval(state: TripState) -> dict[str, Any]:
    """Pause execution via LangGraph interrupt() for Human-in-the-Loop approval."""
    t0 = time.perf_counter()
    payload = {
        "question": "Review TripPilot's audited itinerary before finalizing.",
        "destination": state.get("destination"),
        "revision_count": state.get("revision_count", 0),
        "validation_issues": state.get("validation_issues") or [],
        "budget_report": state.get("budget_report"),
        "weather_audit": state.get("weather_audit"),
        "draft_itinerary": state.get("draft_itinerary"),
    }
    decision_raw = interrupt(payload)
    decision = (
        decision_raw
        if isinstance(decision_raw, dict)
        else {"action": "approve", "feedback": str(decision_raw or "")}
    )

    action = str(decision.get("action", "approve")).lower().strip()
    updated_budget = float(state.get("total_budget", 25000))
    if decision.get("new_budget") is not None:
        try:
            candidate = float(decision["new_budget"])
            if candidate > 0:
                updated_budget = candidate
        except (TypeError, ValueError):
            pass

    revision_count = int(state.get("revision_count", 0))
    max_revisions = int(state.get("max_revisions", DEFAULT_MAX_REVISIONS))
    if action == "replan":
        revision_count += 1
        max_revisions = max(max_revisions, revision_count + 1)

    feedback_note = str(decision.get("feedback") or "").strip()
    summary = f"HITL resumed with action='{action}'"
    if feedback_note:
        summary += f" (feedback: '{feedback_note[:80]}')"

    return {
        "human_decision": decision,
        "total_budget": updated_budget,
        "revision_count": revision_count,
        "max_revisions": max_revisions,
        "status": "self_correcting" if action == "replan" else "finalizing",
        "execution_trace": [
            _make_trace(
                node="human_approval",
                status="resumed",
                summary=summary,
                started_at=t0,
            )
        ],
    }


def finalize_itinerary(state: TripState) -> dict[str, Any]:
    """Produce final travel summary, weather-derived packing tips, and complete receipt."""
    t0 = time.perf_counter()
    draft = ItineraryDraft.model_validate(state["draft_itinerary"])
    budget = BudgetReport.model_validate(state["budget_report"])
    weather_audit = WeatherAudit.model_validate(state["weather_audit"])
    forecast = [DayWeather.model_validate(d) for d in state.get("weather_forecast") or []]
    revision_count = int(state.get("revision_count", 0))

    packing_tips: list[str] = []
    if any(d.is_rain_risk for d in forecast):
        packing_tips.append("Pack a compact umbrella, waterproof shell, and quick-dry footwear for rainy days.")
    if any(d.temp_max_c >= 30.0 for d in forecast):
        packing_tips.append("Highs reach 30°C+: bring breathable cotton/linen layers, sunscreen, and a refillable bottle.")
    if any(d.temp_min_c <= 14.0 for d in forecast):
        packing_tips.append("Cooler evenings (<=14°C): pack a light sweater or packable jacket.")
    if not packing_tips:
        packing_tips.append("Mild temperatures expected; comfortable walking shoes and layered casual wear will cover all days.")

    budget_status = (
        f"Within budget ({format_currency(abs(budget.variance), budget.currency)} remaining)"
        if budget.is_within_budget
        else f"Over budget by {format_currency(budget.variance, budget.currency)}"
    )
    weather_status = (
        "All scheduled activities are rain-compatible"
        if not weather_audit.has_conflicts
        else f"{len(weather_audit.conflicts)} weather warning(s) noted"
    )

    final_summary = (
        f"### {draft.destination} — {len(draft.days)}-Day Approved Itinerary\n"
        f"- **Travelers:** {draft.travelers} | **Self-Correction Cycles:** {revision_count}\n"
        f"- **Budget:** {format_currency(budget.total_estimated, budget.currency)} / "
        f"{format_currency(budget.budget_limit, budget.currency)} ({budget_status})\n"
        f"- **Weather Readiness:** {weather_status}"
    )

    return {
        "final_summary": final_summary,
        "packing_tips": packing_tips,
        "status": "finalized",
        "execution_trace": [
            _make_trace(
                node="finalize_itinerary",
                status="ok",
                summary=f"Finalized {len(draft.days)}-day itinerary after {revision_count} revision(s).",
                started_at=t0,
            )
        ],
    }

