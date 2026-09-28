"""Streamlit entrypoint for TripPilot — Self-Correcting Weather & Budget Travel Agent."""

from __future__ import annotations

import json
import uuid
from datetime import date, timedelta
from typing import Any

import streamlit as st
from langgraph.types import Command

from trippilot.config import (
    DEFAULT_DB_PATH,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_MAX_REVISIONS,
    DEFAULT_RAIN_PROB_THRESHOLD,
    resolve_provider,
)
from trippilot.graph.workflow import get_mermaid_diagram, open_sqlite_graph
from trippilot.ui.components import (
    build_markdown_export,
    inject_custom_styles,
    render_architecture_svg,
    render_budget_panel,
    render_execution_trace,
    render_hero_header,
    render_itinerary_days,
    render_packing_tips,
    render_pipeline_stepper,
    render_revision_diff,
    render_trip_overview_banner,
    render_weather_strip,
)
from trippilot.ui.presets import get_presets

st.set_page_config(
    page_title="TripPilot — Self-Correcting Travel Agent",
    page_icon="🧭",
    layout="wide",
)


def _init_session_defaults() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"trip-{uuid.uuid4().hex[:8]}"
    if "form_values" not in st.session_state:
        first_preset = next(iter(get_presets().values()))
        st.session_state.form_values = dict(first_preset)


def _apply_preset(preset: dict[str, Any]) -> None:
    st.session_state.form_values = dict(preset)
    st.session_state.thread_id = f"trip-{uuid.uuid4().hex[:8]}"


def _load_graph_snapshot(thread_id: str) -> tuple[dict[str, Any], list[Any]]:
    config = {"configurable": {"thread_id": thread_id}}
    with open_sqlite_graph(DEFAULT_DB_PATH) as graph:
        snap = graph.get_state(config)
        values = dict(snap.values) if snap and snap.values else {}
        interrupts: list[Any] = []
        if snap and snap.tasks:
            for task in snap.tasks:
                if getattr(task, "interrupts", None):
                    interrupts.extend(task.interrupts)
        return values, interrupts


def main() -> None:
    inject_custom_styles()
    _init_session_defaults()
    presets = get_presets()

    render_hero_header()

    with st.sidebar:
        st.header("1-Click Demo Presets")
        for name, cfg in presets.items():
            if st.button(name, width="stretch", help=cfg["blurb"]):
                _apply_preset(cfg)
                st.rerun()

        st.divider()
        st.header("LLM & Guardrail Settings")
        provider_choice = st.selectbox(
            "LLM Provider",
            options=["auto", "demo", "gemini", "groq"],
            format_func=lambda p: {
                "auto": "Auto (Gemini → Groq → Demo)",
                "demo": "Demo Mode (₹0 / No API Key)",
                "gemini": "Google Gemini (Free Tier)",
                "groq": "Groq (Free Tier)",
            }[p],
        )
        effective_provider = resolve_provider(provider_choice)
        model_default = (
            DEFAULT_GEMINI_MODEL
            if (provider_choice == "gemini" or (provider_choice == "auto" and effective_provider == "gemini"))
            else DEFAULT_GROQ_MODEL
        )
        api_key_input = ""
        model_input = model_default
        if provider_choice in {"gemini", "groq"}:
            api_key_input = st.text_input(
                f"{provider_choice.title()} API Key (optional if in .env)",
                type="password",
            )
            model_input = st.text_input(
                "Model Name",
                value=model_default,
                key=f"model_name_{provider_choice}",
            )
            resolved_now = resolve_provider(provider_choice, api_key_input)
            if resolved_now == "demo":
                st.warning("No API key found; falling back to Demo Mode.")
        else:
            st.caption(f"Resolved provider: `{effective_provider}`")

        rain_threshold = st.slider(
            "Rain Risk Threshold (%)",
            min_value=20,
            max_value=90,
            value=DEFAULT_RAIN_PROB_THRESHOLD,
            step=5,
        )
        max_revisions = st.slider(
            "Max Self-Correction Loops",
            min_value=1,
            max_value=4,
            value=DEFAULT_MAX_REVISIONS,
        )
        simulate_rain_toggle = st.checkbox(
            "Demo: Force Rainstorm on Day 2",
            value=st.session_state.form_values.get("simulate_rain_day") == 2,
            help="Injects an 85% heavy rainstorm on Day 2 to guarantee a visible weather self-correction loop.",
        )

        st.divider()
        st.caption(f"Active Thread: `{st.session_state.thread_id}`")
        if st.button("Start Fresh Thread", width="stretch"):
            st.session_state.thread_id = f"trip-{uuid.uuid4().hex[:8]}"
            st.rerun()

    fv = st.session_state.form_values
    with st.form("trip_planner_form"):
        c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.0, 1.0])
        destination = c1.text_input("Destination", value=fv.get("destination", "Panjim, India"))
        start_val = date.fromisoformat(fv.get("start_date", (date.today() + timedelta(days=1)).isoformat()))
        start_date_val = c2.date_input("Start Date", value=start_val)
        num_days = int(c3.number_input("Days", min_value=1, max_value=7, value=int(fv.get("num_days", 3))))
        travelers = int(c4.number_input("Travelers", min_value=1, max_value=12, value=int(fv.get("travelers", 2))))

        c5, c6, c7, c8 = st.columns([1.0, 1.4, 1.2, 2.4])
        currencies = ["INR", "USD", "EUR", "GBP", "JPY"]
        cur_idx = currencies.index(fv.get("currency", "INR")) if fv.get("currency", "INR") in currencies else 0
        currency = c5.selectbox("Currency", options=currencies, index=cur_idx)
        total_budget = float(
            c6.number_input(
                "Total Budget",
                min_value=100.0,
                step=500.0 if currency == "INR" else 50.0,
                value=float(fv.get("total_budget", 18000.0)),
            )
        )
        paces = ["relaxed", "balanced", "packed"]
        pace_idx = paces.index(fv.get("pace", "balanced")) if fv.get("pace", "balanced") in paces else 1
        pace = c7.selectbox("Pace", options=paces, index=pace_idx)
        prefs_text = c8.text_input(
            "Interests (comma-separated)",
            value=", ".join(fv.get("preferences", ["beaches", "heritage", "local food"])),
        )

        submitted = st.form_submit_button("🚀 Plan & Verify Trip", type="primary", width="stretch")

    if submitted:
        new_thread = f"trip-{uuid.uuid4().hex[:8]}"
        st.session_state.thread_id = new_thread
        prefs = [p.strip() for p in prefs_text.split(",") if p.strip()] or ["sightseeing", "food"]
        simulate_day = 2 if (simulate_rain_toggle and num_days >= 2) else (1 if simulate_rain_toggle else None)

        initial_state = {
            "destination": destination.strip(),
            "start_date": start_date_val.isoformat(),
            "num_days": num_days,
            "total_budget": total_budget,
            "currency": currency,
            "travelers": travelers,
            "preferences": prefs,
            "pace": pace,
            "provider": provider_choice,
            "model_name": model_input,
            "api_key": api_key_input,
            "rain_prob_threshold": rain_threshold,
            "max_revisions": max_revisions,
            "simulate_rain_day": simulate_day,
            "execution_trace": [],
        }

        config = {"configurable": {"thread_id": new_thread}}
        with st.spinner("Running LangGraph agent (Open-Meteo + DuckDuckGo + Deterministic Audits)..."):
            try:
                with open_sqlite_graph(DEFAULT_DB_PATH) as graph:
                    graph.invoke(initial_state, config=config)
            except Exception as exc:
                st.error(f"Trip planning failed: {exc}")
                return

    state_values, pending_interrupts = _load_graph_snapshot(st.session_state.thread_id)
    if not state_values:
        st.info("Select a 1-click preset on the left or configure your trip above and click **Plan & Verify Trip**.")
        col_arch, col_note = st.columns([1.1, 1.4])
        with col_arch:
            render_architecture_svg()
        with col_note:
            st.markdown(
                """
                ### How TripPilot Works
                1. **Grounds Context:** Resolves coordinates & daily forecasts via **Open-Meteo** and local attractions/pricing via **DuckDuckGo**.
                2. **Drafts & Audits:** Generates a structured multi-day itinerary and verifies it with **deterministic Python math & rain checks**.
                3. **Self-Corrects Automatically:** Loops back to find indoor backups or cheaper stays whenever rain conflicts or budget overruns appear.
                4. **Pauses for You (HITL):** Suspends via LangGraph `interrupt()` in SQLite so you can approve, tweak budget, or request edits before finalizing.
                """
            )
            with st.expander("Mermaid Graph Source"):
                st.code(get_mermaid_diagram(), language="mermaid")
        return

    status = state_values.get("status", "planning")
    if pending_interrupts:
        status = "awaiting_approval"
    revision_count = int(state_values.get("revision_count", 0))

    render_pipeline_stepper(status, revision_count)
    render_trip_overview_banner(state_values)

    # HITL Interrupt Card
    if pending_interrupts:
        st.warning(
            "⏸️ **Paused at Human-in-the-Loop (HITL) Checkpoint** — "
            "LangGraph suspended execution via `interrupt()` and saved state to SQLite. "
            "Review the visual plan below to approve or request changes."
        )
        issues = state_values.get("validation_issues") or []
        if issues:
            st.error(
                "Unresolved warnings after max self-correction loops:\n"
                + "\n".join(f"- {item}" for item in issues)
            )
        elif revision_count > 0:
            st.success(
                f"Self-correction succeeded after {revision_count} loop(s): "
                "all weather and budget checks now pass!"
            )

        with st.form("hitl_decision_form"):
            hc1, hc2 = st.columns([2.5, 1.5])
            feedback = hc1.text_input(
                "Optional Revision Instructions (if requesting changes)",
                placeholder="e.g., Swap Day 2 evening for a street food tour",
            )
            new_budget = hc2.number_input(
                f"Adjust Budget ({state_values.get('currency', 'INR')})",
                min_value=100.0,
                value=float(state_values.get("total_budget", 18000.0)),
                step=500.0 if state_values.get("currency", "INR") == "INR" else 50.0,
            )
            b1, b2 = st.columns(2)
            approve_clicked = b1.form_submit_button("✅ Approve & Finalize", type="primary", width="stretch")
            replan_clicked = b2.form_submit_button("🔁 Request Changes & Re-verify", width="stretch")

        if approve_clicked or replan_clicked:
            decision_payload = {
                "action": "approve" if approve_clicked else "replan",
                "feedback": feedback.strip(),
                "new_budget": float(new_budget),
            }
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            with st.spinner("Resuming LangGraph execution from SQLite checkpoint..."):
                with open_sqlite_graph(DEFAULT_DB_PATH) as graph:
                    graph.invoke(Command(resume=decision_payload), config=config)
            st.rerun()

    elif status == "finalized":
        st.success("✅ **Trip Finalized!**")
        render_packing_tips(state_values.get("packing_tips") or [])
        st.write("")
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "⬇️ Download Itinerary (Markdown)",
            data=build_markdown_export(state_values),
            file_name=f"trippilot_{state_values.get('destination', 'trip').lower().replace(' ', '_')}.md",
            mime="text/markdown",
            width="stretch",
        )
        dl2.download_button(
            "⬇️ Download Execution Trace (JSON)",
            data=json.dumps(state_values, indent=2),
            file_name=f"trippilot_trace_{st.session_state.thread_id}.json",
            mime="application/json",
            width="stretch",
        )

    tab_plan, tab_diff, tab_trace = st.tabs(
        ["🗓️ Itinerary, Weather & Budget", "🔁 Self-Correction Diff", "🧠 Agent Trace & Graph"]
    )

    with tab_plan:
        st.subheader("🌦️ Weather Forecast & Rain Risk")
        render_weather_strip(
            state_values.get("weather_forecast") or [],
            state_values.get("weather_mode", "live_forecast"),
            rain_threshold=int(state_values.get("rain_prob_threshold", rain_threshold)),
        )
        st.divider()
        st.subheader("💰 Deterministic Budget Breakdown")
        render_budget_panel(state_values.get("budget_report") or {})
        st.divider()
        st.subheader("🗓️ Day-by-Day Editorial Itinerary")
        render_itinerary_days(
            state_values.get("draft_itinerary") or {},
            state_values.get("weather_forecast") or [],
        )

    with tab_diff:
        render_revision_diff(state_values.get("revisions") or [])

    with tab_trace:
        t1, t2 = st.columns([1.35, 1.0])
        with t1:
            st.subheader("Node & Tool Execution Timeline")
            render_execution_trace(
                state_values.get("execution_trace") or [],
                st.session_state.thread_id,
            )
        with t2:
            st.subheader("LangGraph Topology")
            render_architecture_svg(state_values.get("execution_trace") or [])
            with st.expander("Mermaid Graph Source"):
                st.code(get_mermaid_diagram(), language="mermaid")


if __name__ == "__main__":
    main()
