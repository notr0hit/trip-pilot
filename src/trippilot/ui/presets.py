"""Curated 1-click recruiter and portfolio demo presets."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any


def get_presets() -> dict[str, dict[str, Any]]:
    """Return preset trip configurations with near-term dates inside Open-Meteo's 16-day window."""
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    next_week = (date.today() + timedelta(days=3)).isoformat()

    return {
        "Goa Monsoon & Budget Stress-Test": {
            "destination": "Panjim, India",
            "start_date": tomorrow,
            "num_days": 3,
            "total_budget": 18000.0,
            "currency": "INR",
            "travelers": 2,
            "pace": "balanced",
            "preferences": ["beaches", "heritage", "local food"],
            "simulate_rain_day": 2,
            "blurb": "Triggers Day 2 rain swap + budget trim to showcase self-correction.",
        },
        "Jaipur Royal Budget Weekend": {
            "destination": "Jaipur, India",
            "start_date": tomorrow,
            "num_days": 3,
            "total_budget": 15000.0,
            "currency": "INR",
            "travelers": 2,
            "pace": "balanced",
            "preferences": ["forts", "bazaars", "street food"],
            "simulate_rain_day": None,
            "blurb": "Live Open-Meteo forecast with tight ₹15,000 budget ceiling.",
        },
        "Kyoto Rain-Ready Culture": {
            "destination": "Kyoto, Japan",
            "start_date": next_week,
            "num_days": 4,
            "total_budget": 900.0,
            "currency": "USD",
            "travelers": 2,
            "pace": "relaxed",
            "preferences": ["temples", "tea houses", "museums"],
            "simulate_rain_day": 2,
            "blurb": "Multi-day international itinerary with indoor rainy-day backup.",
        },
        "London Museums & Markets": {
            "destination": "London, UK",
            "start_date": next_week,
            "num_days": 3,
            "total_budget": 650.0,
            "currency": "GBP",
            "travelers": 2,
            "pace": "packed",
            "preferences": ["museums", "markets", "history"],
            "simulate_rain_day": None,
            "blurb": "Live UK forecast with museum-rich fallback options.",
        },
    }

