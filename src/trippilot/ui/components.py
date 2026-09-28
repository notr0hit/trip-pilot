"""Visual and editorial Streamlit rendering components for TripPilot."""

from __future__ import annotations

import html
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from trippilot.config import DEFAULT_RAIN_PROB_THRESHOLD, format_currency

CATEGORY_META: dict[str, tuple[str, str, str]] = {
    "outdoor": ("Outdoor", "🌿", "#0D9488"),
    "indoor": ("Indoor", "🏛️", "#6366F1"),
    "dining": ("Dining", "🍽️", "#F59E0B"),
    "transit": ("Transit", "🚕", "#64748B"),
    "lodging": ("Lodging", "🏨", "#EC4899"),
}

WEATHER_SOURCE_LABELS = {
    "live_forecast": "Live Open-Meteo Forecast",
    "seasonal_proxy": "Seasonal Proxy (Prior Year)",
    "simulated": "Simulated Rain Demo + Live Forecast",
}


def _esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _render_html(raw: str) -> None:
    """Strip leading indentation so CommonMark never turns HTML blocks into <pre><code>."""
    cleaned = "\n".join(line.strip() for line in raw.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def _weather_icon(code: int, is_rain: bool) -> str:
    if code in {95, 96, 99}:
        return "⛈️"
    if code in {71, 73, 75, 77, 85, 86}:
        return "❄️"
    if is_rain or code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}:
        return "🌧️"
    if code in {45, 48}:
        return "🌫️"
    if code == 3:
        return "☁️"
    if code == 2:
        return "⛅"
    return "☀️"


def inject_custom_styles() -> None:
    """Inject responsive, theme-adaptive CSS for TripPilot's cards and timelines."""
    _render_html(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }
        .tp-hero {
            background: linear-gradient(135deg, rgba(13, 148, 136, 0.14), rgba(59, 130, 246, 0.12));
            border: 1px solid rgba(13, 148, 136, 0.28);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.1rem;
        }
        .tp-hero-title {
            font-size: 1.85rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            margin: 0 0 0.35rem 0;
        }
        .tp-hero-sub {
            font-size: 0.96rem;
            opacity: 0.85;
            margin: 0 0 0.75rem 0;
            line-height: 1.45;
        }
        .tp-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .tp-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.24rem 0.65rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.14);
            border: 1px solid rgba(148, 163, 184, 0.25);
        }
        .tp-stepper {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.6rem;
            margin: 0.75rem 0 1.1rem 0;
        }
        .tp-step {
            border-radius: 12px;
            padding: 0.65rem 0.75rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: rgba(148, 163, 184, 0.06);
            font-size: 0.83rem;
        }
        .tp-step-done {
            border-color: rgba(16, 185, 129, 0.42);
            background: rgba(16, 185, 129, 0.10);
        }
        .tp-step-active {
            border-color: rgba(14, 165, 233, 0.60);
            background: rgba(14, 165, 233, 0.13);
            box-shadow: 0 0 0 1px rgba(14, 165, 233, 0.22);
        }
        .tp-step-label {
            font-weight: 700;
            display: block;
            margin-top: 0.15rem;
        }
        .tp-overview-card {
            border-radius: 14px;
            padding: 1rem 1.25rem;
            border: 1px solid rgba(148, 163, 184, 0.24);
            background: rgba(148, 163, 184, 0.06);
            margin-bottom: 1rem;
        }
        .tp-overview-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .tp-overview-text {
            font-size: 0.93rem;
            line-height: 1.55;
            opacity: 0.9;
            margin: 0;
        }
        .tp-weather-card {
            border-radius: 14px;
            padding: 0.9rem 1rem;
            border: 1px solid rgba(148, 163, 184, 0.24);
            background: rgba(148, 163, 184, 0.06);
            margin-bottom: 0.6rem;
        }
        .tp-weather-rain {
            border-color: rgba(59, 130, 246, 0.45);
            background: linear-gradient(160deg, rgba(59, 130, 246, 0.12), rgba(99, 102, 241, 0.07));
        }
        .tp-weather-clear {
            border-color: rgba(245, 158, 11, 0.38);
            background: linear-gradient(160deg, rgba(245, 158, 11, 0.10), rgba(16, 185, 129, 0.06));
        }
        .tp-day-header {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 0.5rem;
            padding: 0.85rem 1.1rem;
            border-radius: 14px 14px 0 0;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: linear-gradient(90deg, rgba(13, 148, 136, 0.12), rgba(59, 130, 246, 0.08));
            margin-top: 1rem;
        }
        .tp-day-body {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-top: none;
            border-radius: 0 0 14px 14px;
            padding: 0.65rem 1rem 0.85rem 1rem;
            background: rgba(148, 163, 184, 0.03);
            margin-bottom: 0.85rem;
        }
        .tp-act-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            padding: 0.75rem 0.35rem;
            border-bottom: 1px dashed rgba(148, 163, 184, 0.22);
        }
        .tp-act-row:last-child {
            border-bottom: none;
        }
        .tp-slot-pill {
            display: inline-block;
            font-size: 0.70rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 0.16rem 0.5rem;
            border-radius: 6px;
            background: rgba(148, 163, 184, 0.16);
            margin-right: 0.4rem;
        }
        .tp-cat-pill {
            display: inline-block;
            font-size: 0.73rem;
            font-weight: 600;
            padding: 0.15rem 0.52rem;
            border-radius: 999px;
            margin-left: 0.35rem;
        }
        .tp-cost-chip {
            font-weight: 700;
            font-size: 0.92rem;
            white-space: nowrap;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.28);
        }
        .tp-trace-card {
            border-radius: 12px;
            padding: 0.75rem 0.95rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: rgba(148, 163, 184, 0.05);
            margin-bottom: 0.6rem;
        }
        .tp-diff-box {
            border-radius: 12px;
            padding: 0.85rem 1rem;
            border: 1px solid rgba(148, 163, 184, 0.24);
            background: rgba(148, 163, 184, 0.05);
        }
        .tp-tip-card {
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
            border: 1px solid rgba(14, 165, 233, 0.28);
            background: rgba(14, 165, 233, 0.08);
            font-size: 0.88rem;
            line-height: 1.45;
            margin-bottom: 0.5rem;
        }
        @media (max-width: 768px) {
            .tp-stepper {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """
    )


def render_hero_header() -> None:
    """Render TripPilot's top editorial hero header."""
    _render_html(
        """
        <div class="tp-hero">
            <div class="tp-hero-title">🧭 TripPilot — Self-Correcting Weather &amp; Budget Travel Agent</div>
            <p class="tp-hero-sub">
                A stateful <strong>LangGraph</strong> planner that grounds itineraries in live
                <strong>Open-Meteo</strong> forecasts and <strong>DuckDuckGo</strong> research, enforces
                deterministic Python weather &amp; budget guardrails, auto-loops on conflicts, and pauses for
                SQLite-persisted <strong>Human-in-the-Loop</strong> approval.
            </p>
            <div class="tp-chip-row">
                <span class="tp-chip">🌦️ Open-Meteo (No Key)</span>
                <span class="tp-chip">🔎 DuckDuckGo Grounding</span>
                <span class="tp-chip">🧮 Deterministic Budget Audit</span>
                <span class="tp-chip">🔁 Cyclic Self-Correction</span>
                <span class="tp-chip">⏸️ LangGraph HITL + SQLite</span>
            </div>
        </div>
        """
    )


def render_pipeline_stepper(status: str, revision_count: int) -> None:
    """Render a 5-stage connected visual progress stepper."""
    stages = [
        ("Stage 1", "🌦️ Weather & Web"),
        ("Stage 2", "📝 Draft Plan"),
        ("Stage 3", f"🔁 Audit & Loop ({revision_count})"),
        ("Stage 4", "⏸️ Human Approval"),
        ("Stage 5", "✨ Finalized"),
    ]
    order_map = {
        "idle": -1,
        "planning": 1,
        "auditing": 2,
        "self_correcting": 2,
        "awaiting_approval": 3,
        "finalizing": 4,
        "finalized": 4,
    }
    active_idx = order_map.get(status, 0)

    cards: list[str] = []
    for idx, (kicker, title) in enumerate(stages):
        is_done = idx < active_idx or (status == "finalized" and idx == active_idx)
        is_active = idx == active_idx and not is_done
        cls = "tp-step tp-step-done" if is_done else ("tp-step tp-step-active" if is_active else "tp-step")
        state_txt = "Completed" if is_done else ("In Progress" if is_active else "Pending")
        cards.append(
            f'<div class="{cls}">'
            f'<span style="font-size:0.72rem;opacity:0.72;">{_esc(kicker)} · {_esc(state_txt)}</span>'
            f'<span class="tp-step-label">{_esc(title)}</span>'
            f"</div>"
        )
    _render_html(f'<div class="tp-stepper">{"".join(cards)}</div>')


def render_trip_overview_banner(state_values: dict[str, Any]) -> None:
    """Render a crisp narrative executive brief summarizing destination, weather, and budget."""
    draft = state_values.get("draft_itinerary") or {}
    geo = state_values.get("geo_location") or {}
    budget = state_values.get("budget_report") or {}
    forecast = state_values.get("weather_forecast") or []
    revisions = state_values.get("revisions") or []

    dest_label = draft.get("destination") or state_values.get("destination", "Your Trip")
    place_parts = [p for p in [geo.get("name"), geo.get("admin1"), geo.get("country")] if p]
    geo_str = ", ".join(dict.fromkeys(place_parts))
    coords = (
        f" ({geo['latitude']:.2f}°, {geo['longitude']:.2f}°)"
        if "latitude" in geo and "longitude" in geo
        else ""
    )

    num_days = len(draft.get("days") or forecast)
    travelers = draft.get("travelers") or state_values.get("travelers", 2)
    currency = budget.get("currency") or state_values.get("currency", "INR")
    total_est = float(budget.get("total_estimated", 0))
    limit = float(budget.get("budget_limit", 0))
    variance = float(budget.get("variance", 0))
    within = bool(budget.get("is_within_budget", True))

    rainy_days = [f"Day {i + 1}" for i, d in enumerate(forecast) if d.get("is_rain_risk")]
    weather_sentence = (
        f"Rain risk detected on <strong>{', '.join(rainy_days)}</strong> — indoor backups are prioritized."
        if rainy_days
        else "Clear-to-mild conditions across all days — ideal for outdoor exploration."
    )
    budget_sentence = (
        f"Estimated spend is <strong>{format_currency(total_est, currency)}</strong> against a "
        f"<strong>{format_currency(limit, currency)}</strong> ceiling "
        f"(<strong>{format_currency(abs(variance), currency)}</strong> cushion remaining)."
        if within
        else f"Estimated spend is <strong>{format_currency(total_est, currency)}</strong>, exceeding "
        f"the <strong>{format_currency(limit, currency)}</strong> ceiling by "
        f"<strong>{format_currency(variance, currency)}</strong>."
    )

    rev_count = int(state_values.get("revision_count", 0))
    correction_sentence = ""
    if rev_count > 0 and len(revisions) >= 2:
        first_total = float(revisions[0]["budget_report"]["total_estimated"])
        saved = first_total - total_est
        saved_note = f" and trimmed <strong>{format_currency(saved, currency)}</strong>" if saved > 0 else ""
        correction_sentence = (
            f" TripPilot self-corrected across <strong>{rev_count} cycle(s)</strong>{saved_note}."
        )

    _render_html(
        f"""
        <div class="tp-overview-card">
            <div class="tp-overview-title">✨ Executive Trip Brief — {_esc(dest_label)}</div>
            <p class="tp-overview-text">
                Planning <strong>{num_days} days</strong> for <strong>{travelers} travelers</strong>
                {f"in <strong>{_esc(geo_str)}</strong>{_esc(coords)}" if geo_str else ""}.
                {weather_sentence} {budget_sentence}{correction_sentence}
            </p>
        </div>
        """
    )


def render_weather_strip(
    forecast: list[dict[str, Any]],
    weather_mode: str,
    rain_threshold: int = DEFAULT_RAIN_PROB_THRESHOLD,
) -> None:
    """Render styled weather cards plus spaced Altair temperature & precipitation charts."""
    if not forecast:
        return

    st.caption(f"Weather Source: **{WEATHER_SOURCE_LABELS.get(weather_mode, weather_mode)}**")
    cols = st.columns(min(len(forecast), 5))
    for idx, day in enumerate(forecast):
        col = cols[idx % len(cols)]
        rain_risk = bool(day.get("is_rain_risk"))
        card_cls = "tp-weather-card tp-weather-rain" if rain_risk else "tp-weather-card tp-weather-clear"
        icon = _weather_icon(int(day.get("weather_code", 0)), rain_risk)
        badge = "Rain Risk" if rain_risk else "Outdoor OK"
        badge_bg = "rgba(59, 130, 246, 0.22)" if rain_risk else "rgba(16, 185, 129, 0.20)"
        prob = int(day.get("precip_prob_max", 0))

        with col:
            _render_html(
                f"""
                <div class="{card_cls}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <strong>Day {idx + 1}</strong>
                        <span style="font-size:0.74rem;padding:0.15rem 0.5rem;border-radius:999px;background:{badge_bg};font-weight:600;">{_esc(badge)}</span>
                    </div>
                    <div style="font-size:0.78rem;opacity:0.75;margin-bottom:0.4rem;">{_esc(day['date'])}</div>
                    <div style="font-size:1.05rem;font-weight:650;margin-bottom:0.35rem;">{icon} {_esc(day['condition_label'])}</div>
                    <div style="font-size:0.86rem;">🌡️ <strong>{day['temp_min_c']}°</strong> – <strong>{day['temp_max_c']}°C</strong></div>
                    <div style="font-size:0.84rem;margin-top:0.2rem;">💧 Precip: <strong>{prob}%</strong></div>
                </div>
                """
            )

    _render_html('<div style="height:1rem;"></div>')

    df_weather = pd.DataFrame(
        [
            {
                "Day": f"Day {i + 1}",
                "Date": d["date"],
                "Precip %": int(d.get("precip_prob_max", 0)),
                "High (°C)": float(d.get("temp_max_c", 0)),
                "Low (°C)": float(d.get("temp_min_c", 0)),
                "Status": "Rain Risk" if d.get("is_rain_risk") else "Outdoor OK",
            }
            for i, d in enumerate(forecast)
        ]
    )

    left, right = st.columns(2)
    with left:
        bars = (
            alt.Chart(df_weather)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=34)
            .encode(
                x=alt.X("Day:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y(
                    "Precip %:Q",
                    scale=alt.Scale(domain=[0, 100]),
                    title="Precipitation (%)",
                ),
                color=alt.Color(
                    "Status:N",
                    scale=alt.Scale(domain=["Outdoor OK", "Rain Risk"], range=["#10B981", "#3B82F6"]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=["Day", "Date", "Precip %", "Status"],
            )
        )
        threshold_rule = (
            alt.Chart(pd.DataFrame({"y": [rain_threshold]}))
            .mark_rule(color="#EF4444", strokeDash=[6, 4], strokeWidth=2)
            .encode(y="y:Q")
        )
        rain_chart = (bars + threshold_rule).properties(
            height=240,
            padding={"top": 12, "left": 6, "right": 8, "bottom": 4},
            title="Rain Risk vs. Threshold",
        )
        st.altair_chart(rain_chart, width="stretch")

    with right:
        band = (
            alt.Chart(df_weather)
            .mark_area(opacity=0.18, color="#0EA5E9")
            .encode(
                x=alt.X("Day:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Low (°C):Q", scale=alt.Scale(zero=False), title="Temperature (°C)"),
                y2="High (°C):Q",
            )
        )
        temp_long = df_weather.melt(
            id_vars=["Day", "Date"],
            value_vars=["High (°C)", "Low (°C)"],
            var_name="Metric",
            value_name="Temp (°C)",
        )
        lines = (
            alt.Chart(temp_long)
            .mark_line(point=alt.OverlayMarkDef(size=75), strokeWidth=3)
            .encode(
                x=alt.X("Day:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Temp (°C):Q", scale=alt.Scale(zero=False), title="Temperature (°C)"),
                color=alt.Color(
                    "Metric:N",
                    scale=alt.Scale(domain=["High (°C)", "Low (°C)"], range=["#F59E0B", "#0EA5E9"]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=["Day", "Date", "Metric", "Temp (°C)"],
            )
        )
        temp_chart = (band + lines).properties(
            height=240,
            padding={"top": 12, "left": 6, "right": 8, "bottom": 4},
            title="Daily Temperature Range",
        )
        st.altair_chart(temp_chart, width="stretch")


def render_budget_panel(budget_report: dict[str, Any]) -> None:
    """Render deterministic budget KPIs, Altair charts, and top cost drivers."""
    if not budget_report:
        return

    currency = budget_report.get("currency", "INR")
    limit = float(budget_report.get("budget_limit", 0))
    total = float(budget_report.get("total_estimated", 0))
    variance = float(budget_report.get("variance", 0))
    within = bool(budget_report.get("is_within_budget", True))

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Total", format_currency(total, currency))
    c2.metric("Budget Ceiling", format_currency(limit, currency))
    delta_label = (
        f"{format_currency(abs(variance), currency)} under"
        if within
        else f"+{format_currency(variance, currency)} over"
    )
    c3.metric(
        "Audit Verdict",
        "Within Budget ✅" if within else "Over Budget ⚠️",
        delta=delta_label,
        delta_color="normal" if within else "inverse",
    )

    ratio = min(1.0, total / limit) if limit > 0 else 0.0
    pct = (total / limit * 100) if limit > 0 else 0.0
    st.progress(
        ratio,
        text=f"Budget Utilization: {pct:.1f}% ({format_currency(total, currency)} / {format_currency(limit, currency)})",
    )
    _render_html('<div style="height:0.65rem;"></div>')

    by_cat = budget_report.get("by_category") or []
    by_day = budget_report.get("by_day") or []

    chart_l, chart_r = st.columns(2)
    with chart_l:
        if by_cat:
            df_cat = pd.DataFrame(
                [
                    {
                        "Category": row["category"].title(),
                        "Amount": float(row["amount"]),
                        "Formatted": format_currency(row["amount"], currency),
                        "Share %": row["share_pct"],
                    }
                    for row in by_cat
                ]
            )
            donut = (
                alt.Chart(df_cat)
                .mark_arc(innerRadius=54, outerRadius=92, cornerRadius=4)
                .encode(
                    theta=alt.Theta("Amount:Q"),
                    color=alt.Color(
                        "Category:N",
                        scale=alt.Scale(
                            range=["#0D9488", "#6366F1", "#F59E0B", "#EC4899", "#64748B"]
                        ),
                        legend=alt.Legend(orient="right", title="Category"),
                    ),
                    tooltip=["Category", "Formatted", "Share %"],
                )
                .properties(
                    height=250,
                    padding={"top": 10, "left": 6, "right": 8, "bottom": 4},
                    title="Spend Breakdown by Category",
                )
            )
            st.altair_chart(donut, width="stretch")

    with chart_r:
        if by_day:
            df_day = pd.DataFrame(
                [
                    {
                        "Day": f"Day {row['day_number']}",
                        "Date": row["date"],
                        "Amount": float(row["amount"]),
                        "Formatted": format_currency(row["amount"], currency),
                    }
                    for row in by_day
                ]
            )
            avg_budget_day = limit / max(1, len(by_day))
            day_bars = (
                alt.Chart(df_day)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=36, color="#0D9488")
                .encode(
                    x=alt.X("Day:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Amount:Q", title=f"Daily Spend ({currency})"),
                    tooltip=["Day", "Date", "Formatted"],
                )
            )
            pace_rule = (
                alt.Chart(pd.DataFrame({"y": [avg_budget_day]}))
                .mark_rule(color="#F59E0B", strokeDash=[5, 4], strokeWidth=2)
                .encode(y="y:Q")
            )
            st.altair_chart(
                (day_bars + pace_rule).properties(
                    height=250,
                    padding={"top": 10, "left": 6, "right": 8, "bottom": 4},
                    title="Daily Spend vs. Target Daily Pace",
                ),
                width="stretch",
            )

    drivers = budget_report.get("top_cost_drivers") or []
    if drivers:
        st.markdown("**Top Cost Drivers**")
        rows = [
            {
                "Day": f"Day {row['day_number']}",
                "Activity": row["title"],
                "Category": row["category"].title(),
                "Cost": format_currency(row["estimated_cost"], currency),
            }
            for row in drivers
        ]
        st.dataframe(rows, hide_index=True, width="stretch")


def render_itinerary_days(draft: dict[str, Any], forecast: list[dict[str, Any]]) -> None:
    """Render editorial day cards and structured activity timeline rows."""
    if not draft:
        return
    currency = draft.get("currency", "INR")
    if draft.get("rationale"):
        st.info(f"**Planner Rationale:** {draft['rationale']}")

    forecast_by_date = {d["date"]: d for d in forecast or []}
    for day in draft.get("days") or []:
        dw = forecast_by_date.get(day["date"], {})
        is_rain = bool(dw.get("is_rain_risk"))
        w_icon = _weather_icon(int(dw.get("weather_code", 0)), is_rain)
        weather_pill = (
            f"{w_icon} {_esc(dw.get('condition_label', 'Rain'))} ({dw.get('precip_prob_max', 0)}%)"
            if is_rain
            else f"{w_icon} {_esc(dw.get('condition_label', 'Clear'))}"
        )
        day_total = sum(float(a.get("estimated_cost", 0)) for a in day.get("activities") or [])

        rows_html: list[str] = []
        for act in day.get("activities") or []:
            cat_key = str(act.get("category", "indoor")).lower()
            cat_label, cat_icon, cat_color = CATEGORY_META.get(
                cat_key, (cat_key.title(), "📍", "#64748B")
            )
            rain_safe = (
                '<span style="font-size:0.73rem;margin-left:0.4rem;opacity:0.85;">☔ Rain-safe</span>'
                if act.get("weather_safe_if_rain")
                else ""
            )
            src_html = (
                f' · <a href="{_esc(act["source_url"])}" target="_blank" '
                f'style="text-decoration:none;font-weight:600;">Verify Source ↗</a>'
                if act.get("source_url")
                else ""
            )
            notes = _esc(act.get("notes", ""))
            dur = act.get("duration_hours", 2)
            cost_str = format_currency(float(act.get("estimated_cost", 0)), currency)

            rows_html.append(
                '<div class="tp-act-row">'
                '<div style="flex:1;min-width:0;">'
                "<div>"
                f'<span class="tp-slot-pill">{_esc(act.get("time_slot", ""))}</span>'
                f'<strong style="font-size:0.98rem;">{_esc(act.get("title", ""))}</strong>'
                f'<span class="tp-cat-pill" style="background:{cat_color}22;color:{cat_color};border:1px solid {cat_color}55;">'
                f"{cat_icon} {_esc(cat_label)}</span>"
                f"{rain_safe}"
                "</div>"
                '<div style="font-size:0.86rem;opacity:0.82;margin-top:0.28rem;line-height:1.45;">'
                f'{notes} <span style="opacity:0.75;">({dur}h)</span>{src_html}'
                "</div>"
                "</div>"
                f'<div class="tp-cost-chip">{_esc(cost_str)}</div>'
                "</div>"
            )

        day_card_html = (
            '<div class="tp-day-header">'
            "<div>"
            f'<span style="font-size:1.05rem;font-weight:750;">Day {int(day["day_number"])} · {_esc(day["theme"])}</span>'
            f'<span style="font-size:0.84rem;opacity:0.76;margin-left:0.55rem;">{_esc(day["date"])}</span>'
            "</div>"
            '<div style="display:flex;align-items:center;gap:0.6rem;font-size:0.85rem;">'
            f"<span>{weather_pill}</span>"
            f'<span class="tp-cost-chip">{_esc(format_currency(day_total, currency))}</span>'
            "</div>"
            "</div>"
            f'<div class="tp-day-body">{"".join(rows_html)}</div>'
        )
        _render_html(day_card_html)


def render_revision_diff(revisions: list[dict[str, Any]]) -> None:
    """Show Before vs. After self-correction chart and side-by-side activity diff."""
    if not revisions:
        st.info("Run the planner to inspect revision snapshots.")
        return

    if len(revisions) == 1:
        st.success("Passed all weather and budget audits on Revision #0 (no re-plan loop needed).")
        return

    first = revisions[0]
    latest = revisions[-1]
    currency = first["budget_report"].get("currency", "INR")
    limit = float(latest["budget_report"]["budget_limit"])

    st.markdown(f"### Self-Correction Impact (`Rev #0` → `Rev #{latest['revision_number']}`)")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Total Spend",
        format_currency(latest["budget_report"]["total_estimated"], currency),
        delta=format_currency(
            latest["budget_report"]["total_estimated"] - first["budget_report"]["total_estimated"],
            currency,
        ),
        delta_color="inverse",
    )
    c2.metric(
        "Weather Conflicts",
        len(latest["weather_audit"]["conflicts"]),
        delta=len(latest["weather_audit"]["conflicts"]) - len(first["weather_audit"]["conflicts"]),
        delta_color="inverse",
    )
    c3.metric("Re-plan Iterations", latest["revision_number"])

    df_rev = pd.DataFrame(
        [
            {
                "Revision": f"Rev #{rev['revision_number']}",
                "Total Spend": float(rev["budget_report"]["total_estimated"]),
                "Formatted": format_currency(rev["budget_report"]["total_estimated"], currency),
                "Status": "Pass" if rev["budget_report"]["is_within_budget"] else "Overrun",
            }
            for rev in revisions
        ]
    )
    rev_bars = (
        alt.Chart(df_rev)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=48)
        .encode(
            x=alt.X("Revision:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Total Spend:Q", title=f"Estimated Spend ({currency})"),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(domain=["Overrun", "Pass"], range=["#EF4444", "#10B981"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["Revision", "Formatted", "Status"],
        )
    )
    ceiling_rule = (
        alt.Chart(pd.DataFrame({"y": [limit]}))
        .mark_rule(color="#F59E0B", strokeDash=[6, 4], strokeWidth=2)
        .encode(y="y:Q")
    )
    st.altair_chart(
        (rev_bars + ceiling_rule).properties(
            height=225,
            padding={"top": 12, "left": 6, "right": 8, "bottom": 4},
            title="Self-Correction Spend Convergence vs. Budget Ceiling",
        ),
        width="stretch",
    )

    latest_titles = {
        (d["day_number"], a["title"])
        for d in latest["draft"]["days"]
        for a in d["activities"]
    }
    first_titles = {
        (d["day_number"], a["title"])
        for d in first["draft"]["days"]
        for a in d["activities"]
    }

    left, right = st.columns(2)
    with left:
        lines = ["<div class='tp-diff-box'><strong>Before (Revision #0)</strong><hr style='margin:0.45rem 0;opacity:0.2;'/>"]
        for day in first["draft"]["days"]:
            lines.append(f"<div style='font-weight:650;margin-top:0.45rem;'>Day {int(day['day_number'])}</div>")
            for act in day["activities"]:
                removed = (day["day_number"], act["title"]) not in latest_titles
                tag = " <span style='color:#EF4444;font-weight:700;'>[Replaced]</span>" if removed else ""
                lines.append(
                    f"<div style='font-size:0.86rem;margin:0.2rem 0;'>• {_esc(act['title'])} "
                    f"(<code>{_esc(act['category'])}</code>, {_esc(format_currency(act['estimated_cost'], currency))}){tag}</div>"
                )
        lines.append("</div>")
        _render_html("".join(lines))

    with right:
        lines = [
            f"<div class='tp-diff-box'><strong>After (Revision #{int(latest['revision_number'])})</strong>"
            "<hr style='margin:0.45rem 0;opacity:0.2;'/>"
        ]
        for day in latest["draft"]["days"]:
            lines.append(f"<div style='font-weight:650;margin-top:0.45rem;'>Day {int(day['day_number'])}</div>")
            for act in day["activities"]:
                changed = (day["day_number"], act["title"]) not in first_titles
                tag = " <span style='color:#10B981;font-weight:700;'>[Swapped In]</span>" if changed else ""
                lines.append(
                    f"<div style='font-size:0.86rem;margin:0.2rem 0;'>• {_esc(act['title'])} "
                    f"(<code>{_esc(act['category'])}</code>, {_esc(format_currency(act['estimated_cost'], currency))}){tag}</div>"
                )
        lines.append("</div>")
        _render_html("".join(lines))


def render_execution_trace(trace: list[dict[str, Any]], thread_id: str) -> None:
    """Display styled LangGraph node execution cards and tool invocations."""
    st.caption(f"SQLite Checkpoint Thread ID: `{thread_id}`")
    if not trace:
        st.info("No execution events yet.")
        return

    status_badge = {
        "ok": ("✅ OK", "#10B981"),
        "warning": ("⚠️ Warning", "#F59E0B"),
        "loop_back": ("🔁 Self-Correct Loop", "#3B82F6"),
        "interrupted": ("⏸️ Interrupted", "#8B5CF6"),
        "resumed": ("▶️ HITL Resumed", "#0D9488"),
    }
    for idx, ev in enumerate(trace, start=1):
        label, color = status_badge.get(ev.get("status", "ok"), ("• Step", "#64748B"))
        tools_html = ""
        for tc in ev.get("tool_calls") or []:
            tools_html += (
                f"<div style='font-size:0.80rem;opacity:0.85;margin-top:0.25rem;'>"
                f"🔧 <code>{_esc(tc['tool_name'])}</code> ({_esc(tc['args_summary'])}) → {_esc(tc['result_summary'])}"
                f"</div>"
            )
        _render_html(
            f"""
            <div class="tp-trace-card">
                <div style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;">
                    <div>
                        <strong>{idx}. <code>{_esc(ev['node'])}</code></strong>
                        <span style="font-size:0.74rem;font-weight:700;padding:0.14rem 0.5rem;border-radius:999px;background:{color}22;color:{color};border:1px solid {color}55;margin-left:0.4rem;">
                            {_esc(label)}
                        </span>
                    </div>
                    <span style="font-size:0.76rem;opacity:0.72;">{int(ev.get('duration_ms', 0))} ms · {_esc(ev.get('timestamp', ''))}</span>
                </div>
                <div style="font-size:0.87rem;margin-top:0.35rem;line-height:1.45;">{_esc(ev.get('summary', ''))}</div>
                {tools_html}
            </div>
            """
        )


def render_architecture_svg(trace: list[dict[str, Any]] | None = None) -> None:
    """Render a crisp inline SVG diagram of TripPilot's cyclic LangGraph state machine."""
    visited = {ev.get("node") for ev in (trace or [])}
    looped = any(ev.get("status") == "loop_back" for ev in (trace or []))

    def box_style(node: str, accent: str) -> tuple[str, str]:
        if node in visited:
            return f"{accent}26", accent
        return "rgba(148,163,184,0.10)", "#64748B"

    g_fill, g_stroke = box_style("gather_grounding", "#0D9488")
    p_fill, p_stroke = box_style("plan_itinerary", "#0EA5E9")
    a_fill, a_stroke = box_style("audit_plan", "#F59E0B")
    r_fill, r_stroke = box_style("targeted_replan_search", "#3B82F6")
    h_fill, h_stroke = box_style("human_approval", "#8B5CF6")
    f_fill, f_stroke = box_style("finalize_itinerary", "#10B981")
    loop_stroke = "#3B82F6" if looped else "#94A3B8"

    svg = f"""
    <svg viewBox="0 0 440 490" width="100%" style="max-width:440px;font-family:system-ui,-apple-system,sans-serif;">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#94A3B8" />
        </marker>
        <marker id="arrow-loop" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="{loop_stroke}" />
        </marker>
      </defs>
      <line x1="150" y1="36" x2="150" y2="56" stroke="#94A3B8" stroke-width="2" marker-end="url(#arrow)" />
      <line x1="150" y1="104" x2="150" y2="128" stroke="#94A3B8" stroke-width="2" marker-end="url(#arrow)" />
      <line x1="150" y1="176" x2="150" y2="202" stroke="#94A3B8" stroke-width="2" marker-end="url(#arrow)" />
      <line x1="150" y1="252" x2="150" y2="292" stroke="#94A3B8" stroke-width="2" marker-end="url(#arrow)" />
      <line x1="150" y1="344" x2="150" y2="384" stroke="#94A3B8" stroke-width="2" marker-end="url(#arrow)" />
      <line x1="150" y1="432" x2="150" y2="452" stroke="#94A3B8" stroke-width="2" marker-end="url(#arrow)" />
      <path d="M 248 227 L 336 227 L 336 184" fill="none" stroke="{loop_stroke}" stroke-width="2" stroke-dasharray="5,4" marker-end="url(#arrow-loop)" />
      <path d="M 336 134 L 336 152 L 248 152" fill="none" stroke="{loop_stroke}" stroke-width="2" stroke-dasharray="5,4" marker-end="url(#arrow-loop)" />
      <text x="292" y="219" font-size="10" fill="{loop_stroke}" text-anchor="middle" font-weight="600">Rain / Overrun</text>
      <path d="M 52 318 L 16 318 L 16 152 L 50 152" fill="none" stroke="#8B5CF6" stroke-width="1.8" stroke-dasharray="4,4" marker-end="url(#arrow)" />
      <text x="20" y="240" font-size="10" fill="#8B5CF6" transform="rotate(-90 20 240)" text-anchor="middle" font-weight="600">HITL Edit</text>
      <rect x="108" y="8" width="84" height="28" rx="14" fill="rgba(148,163,184,0.18)" stroke="#94A3B8" stroke-width="1.5"/>
      <text x="150" y="26" font-size="11" fill="currentColor" text-anchor="middle" font-weight="700">START</text>
      <rect x="52" y="58" width="196" height="46" rx="10" fill="{g_fill}" stroke="{g_stroke}" stroke-width="2"/>
      <text x="150" y="78" font-size="12" fill="currentColor" text-anchor="middle" font-weight="700">gather_grounding</text>
      <text x="150" y="94" font-size="10" fill="currentColor" opacity="0.75" text-anchor="middle">Open-Meteo + DuckDuckGo</text>
      <rect x="52" y="130" width="196" height="46" rx="10" fill="{p_fill}" stroke="{p_stroke}" stroke-width="2"/>
      <text x="150" y="150" font-size="12" fill="currentColor" text-anchor="middle" font-weight="700">plan_itinerary</text>
      <text x="150" y="166" font-size="10" fill="currentColor" opacity="0.75" text-anchor="middle">Structured Draft Generator</text>
      <rect x="52" y="204" width="196" height="48" rx="10" fill="{a_fill}" stroke="{a_stroke}" stroke-width="2"/>
      <text x="150" y="224" font-size="12" fill="currentColor" text-anchor="middle" font-weight="700">audit_plan</text>
      <text x="150" y="240" font-size="10" fill="currentColor" opacity="0.75" text-anchor="middle">Deterministic Budget &amp; Rain</text>
      <rect x="260" y="134" width="156" height="48" rx="10" fill="{r_fill}" stroke="{r_stroke}" stroke-width="2"/>
      <text x="338" y="154" font-size="11" fill="currentColor" text-anchor="middle" font-weight="700">targeted_replan</text>
      <text x="338" y="170" font-size="10" fill="currentColor" opacity="0.75" text-anchor="middle">Indoor / Budget Search</text>
      <rect x="52" y="294" width="196" height="50" rx="10" fill="{h_fill}" stroke="{h_stroke}" stroke-width="2"/>
      <text x="150" y="315" font-size="12" fill="currentColor" text-anchor="middle" font-weight="700">human_approval</text>
      <text x="150" y="331" font-size="10" fill="currentColor" opacity="0.75" text-anchor="middle">interrupt() + SQLite Pause</text>
      <rect x="52" y="386" width="196" height="46" rx="10" fill="{f_fill}" stroke="{f_stroke}" stroke-width="2"/>
      <text x="150" y="406" font-size="12" fill="currentColor" text-anchor="middle" font-weight="700">finalize_itinerary</text>
      <text x="150" y="422" font-size="10" fill="currentColor" opacity="0.75" text-anchor="middle">Summary + Packing Tips</text>
      <rect x="108" y="454" width="84" height="28" rx="14" fill="rgba(148,163,184,0.18)" stroke="#94A3B8" stroke-width="1.5"/>
      <text x="150" y="472" font-size="11" fill="currentColor" text-anchor="middle" font-weight="700">END</text>
    </svg>
    """
    _render_html(svg)


def render_packing_tips(tips: list[str]) -> None:
    """Render weather-aware packing readiness cards."""
    if not tips:
        return
    st.markdown("**🎒 Weather-Aware Packing & Readiness**")
    cols = st.columns(len(tips))
    for col, tip in zip(cols, tips):
        with col:
            _render_html(f"<div class='tp-tip-card'>✓ {_esc(tip)}</div>")


def build_markdown_export(state_values: dict[str, Any]) -> str:
    """Format the finalized itinerary, budget report, and packing tips as Markdown."""
    draft = state_values.get("draft_itinerary") or {}
    budget = state_values.get("budget_report") or {}
    packing = state_values.get("packing_tips") or []
    currency = draft.get("currency", "INR")

    lines = [
        f"# TripPilot Itinerary: {draft.get('destination', 'Trip')}",
        "",
        state_values.get("final_summary", ""),
        "",
        "## Daily Schedule",
    ]
    for day in draft.get("days") or []:
        lines.append(f"### Day {day['day_number']} ({day['date']}) — {day['theme']}")
        for act in day.get("activities") or []:
            cost = format_currency(act.get("estimated_cost", 0), currency)
            src = f" ([Source]({act['source_url']}))" if act.get("source_url") else ""
            lines.append(
                f"- **{act['time_slot'].title()}**: {act['title']} "
                f"({act['category']}, {cost}){src} — {act.get('notes', '')}"
            )
        lines.append("")

    if budget:
        lines.append("## Deterministic Budget Summary")
        lines.append(
            f"- Total Estimated: {format_currency(budget.get('total_estimated', 0), currency)} "
            f"/ Ceiling: {format_currency(budget.get('budget_limit', 0), currency)}"
        )
        lines.append("")

    if packing:
        lines.append("## Weather Packing Tips")
        for tip in packing:
            lines.append(f"- {tip}")

    return "\n".join(lines)
