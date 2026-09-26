"""Open-Meteo geocoding, weather retrieval, and deterministic weather auditing."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import httpx
from langchain_core.tools import tool

from trippilot.config import DEFAULT_RAIN_PROB_THRESHOLD
from trippilot.state import (
    DayWeather,
    GeoLocation,
    ItineraryDraft,
    WeatherAudit,
    WeatherConflict,
)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

WMO_CODE_MAP: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

PRECIP_WMO_CODES: set[int] = {
    51, 53, 55, 56, 57,
    61, 63, 65, 66, 67,
    71, 73, 75, 77,
    80, 81, 82, 85, 86,
    95, 96, 99,
}


def decode_wmo_code(code: int) -> str:
    """Convert a WMO weather code into a human-readable condition label."""
    return WMO_CODE_MAP.get(int(code), f"Weather code {code}")


def _shift_year(d: date, years: int) -> date:
    """Safely shift a date by whole years (handling Feb 29)."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)


def geocode_destination(
    destination: str,
    client: httpx.Client | None = None,
) -> GeoLocation:
    """Resolve a destination string into coordinates via Open-Meteo Geocoding."""
    query = destination.strip()
    if not query:
        raise ValueError("Destination must not be empty.")

    candidates = [query]
    country_hint = ""
    city_lower = query.lower()
    if "," in query:
        city_part, hint_part = [part.strip() for part in query.split(",", 1)]
        country_hint = hint_part.lower()
        city_lower = city_part.lower()
        if city_part and city_part != query:
            candidates.append(city_part)
    if city_lower in {"goa", "panaji"} and (not country_hint or "india" in country_hint):
        candidates.append("Panjim")

    owns_client = client is None
    http = client or httpx.Client(timeout=15.0)
    try:
        for candidate in candidates:
            resp = http.get(
                GEOCODING_URL,
                params={"name": candidate, "count": 10, "language": "en", "format": "json"},
            )
            resp.raise_for_status()
            payload = resp.json()
            results = payload.get("results") or []
            if results:
                if country_hint:
                    hinted = [
                        r
                        for r in results
                        if country_hint in str(r.get("country", "")).lower()
                        or country_hint in str(r.get("admin1", "")).lower()
                    ]
                    if hinted:
                        results = hinted
                    elif candidate != candidates[-1]:
                        continue
                best = max(results, key=lambda r: int(r.get("population") or 0))
                if (
                    city_lower in {"goa", "panaji"}
                    and candidate != "Panjim"
                    and str(best.get("admin1", "")).lower() != "goa"
                ):
                    continue
                return GeoLocation(
                    name=best.get("name", candidate),
                    country=best.get("country", ""),
                    admin1=best.get("admin1", ""),
                    latitude=float(best["latitude"]),
                    longitude=float(best["longitude"]),
                    timezone=best.get("timezone", "auto"),
                )
        raise ValueError(f"Could not geocode destination '{destination}' via Open-Meteo.")
    finally:
        if owns_client:
            http.close()


def _build_day_weather(
    day_str: str,
    code: int,
    temp_min: float,
    temp_max: float,
    precip_prob: int,
    precip_sum: float,
    rain_prob_threshold: int,
    source_mode: str,
) -> DayWeather:
    is_rain = (precip_prob >= rain_prob_threshold) or (code in PRECIP_WMO_CODES)
    return DayWeather(
        date=day_str,
        weather_code=code,
        condition_label=decode_wmo_code(code),
        temp_min_c=round(float(temp_min), 1),
        temp_max_c=round(float(temp_max), 1),
        precip_prob_max=int(max(0, min(100, precip_prob))),
        precip_sum_mm=round(float(precip_sum), 1),
        is_rain_risk=is_rain,
        source_mode=source_mode,  # type: ignore[arg-type]
    )


def fetch_weather_forecast(
    geo: GeoLocation,
    start_date: str,
    num_days: int,
    rain_prob_threshold: int = DEFAULT_RAIN_PROB_THRESHOLD,
    simulate_rain_day: int | None = None,
    client: httpx.Client | None = None,
    today_override: date | None = None,
) -> tuple[list[DayWeather], str]:
    """Fetch Open-Meteo daily forecast, falling back to prior-year archive beyond 16 days."""
    start = date.fromisoformat(start_date)
    end = start + timedelta(days=num_days - 1)
    today = today_override or date.today()
    max_forecast_day = today + timedelta(days=15)

    use_live_forecast = (start >= today - timedelta(days=1)) and (end <= max_forecast_day)

    owns_client = client is None
    http = client or httpx.Client(timeout=15.0)
    try:
        days_weather: list[DayWeather] = []
        if use_live_forecast:
            params = {
                "latitude": geo.latitude,
                "longitude": geo.longitude,
                "daily": [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_max",
                    "precipitation_sum",
                ],
                "timezone": geo.timezone or "auto",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            }
            resp = http.get(FORECAST_URL, params=params)
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            times = daily.get("time", [])
            for idx, d_str in enumerate(times):
                code = int((daily.get("weather_code") or [0])[idx] or 0)
                t_max = float((daily.get("temperature_2m_max") or [28.0])[idx] or 28.0)
                t_min = float((daily.get("temperature_2m_min") or [20.0])[idx] or 20.0)
                prob = int((daily.get("precipitation_probability_max") or [0])[idx] or 0)
                p_sum = float((daily.get("precipitation_sum") or [0.0])[idx] or 0.0)
                days_weather.append(
                    _build_day_weather(
                        d_str, code, t_min, t_max, prob, p_sum,
                        rain_prob_threshold, "live_forecast",
                    )
                )
            weather_mode = "live_forecast"
        else:
            proxy_start = _shift_year(start, -1)
            proxy_end = _shift_year(end, -1)
            params = {
                "latitude": geo.latitude,
                "longitude": geo.longitude,
                "daily": [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                ],
                "timezone": geo.timezone or "auto",
                "start_date": proxy_start.isoformat(),
                "end_date": proxy_end.isoformat(),
            }
            resp = http.get(ARCHIVE_URL, params=params)
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            times = daily.get("time", [])
            for idx, _ in enumerate(times):
                target_day = (start + timedelta(days=idx)).isoformat()
                code = int((daily.get("weather_code") or [0])[idx] or 0)
                t_max = float((daily.get("temperature_2m_max") or [28.0])[idx] or 28.0)
                t_min = float((daily.get("temperature_2m_min") or [20.0])[idx] or 20.0)
                p_sum = float((daily.get("precipitation_sum") or [0.0])[idx] or 0.0)
                # Archive API omits forecast probability; infer deterministically from precip/WMO code
                if p_sum >= 5.0:
                    inferred_prob = 80
                elif p_sum >= 1.5 or code in PRECIP_WMO_CODES:
                    inferred_prob = 60
                else:
                    inferred_prob = 15
                days_weather.append(
                    _build_day_weather(
                        target_day, code, t_min, t_max, inferred_prob, p_sum,
                        rain_prob_threshold, "seasonal_proxy",
                    )
                )
            weather_mode = "seasonal_proxy"

        if simulate_rain_day is not None and 1 <= simulate_rain_day <= len(days_weather):
            target = days_weather[simulate_rain_day - 1]
            days_weather[simulate_rain_day - 1] = DayWeather(
                date=target.date,
                weather_code=65,
                condition_label="Heavy rain (Simulated Demo)",
                temp_min_c=target.temp_min_c,
                temp_max_c=round(max(target.temp_min_c + 2.0, target.temp_max_c - 3.0), 1),
                precip_prob_max=85,
                precip_sum_mm=18.5,
                is_rain_risk=True,
                source_mode="simulated",
            )
            weather_mode = "simulated"

        return days_weather, weather_mode
    finally:
        if owns_client:
            http.close()


def audit_weather(draft: ItineraryDraft, forecast: list[DayWeather]) -> WeatherAudit:
    """Deterministically flag outdoor non-rain-safe activities scheduled on rainy days."""
    by_date = {day_w.date: day_w for day_w in forecast}
    conflicts: list[WeatherConflict] = []

    for day_plan in draft.days:
        weather = by_date.get(day_plan.date)
        if weather is None and 0 <= day_plan.day_number - 1 < len(forecast):
            weather = forecast[day_plan.day_number - 1]
        if weather is None or not weather.is_rain_risk:
            continue

        bad_items = [
            act.title
            for act in day_plan.activities
            if act.category == "outdoor" and not act.weather_safe_if_rain
        ]
        if bad_items:
            conflicts.append(
                WeatherConflict(
                    day_number=day_plan.day_number,
                    date=day_plan.date,
                    condition_label=weather.condition_label,
                    precip_prob_max=weather.precip_prob_max,
                    conflicting_activities=bad_items,
                    reason=(
                        f"Day {day_plan.day_number} ({day_plan.date}) forecasts "
                        f"{weather.condition_label} ({weather.precip_prob_max}% precip prob) "
                        f"with outdoor activities: {', '.join(bad_items)}"
                    ),
                )
            )

    return WeatherAudit(
        has_conflicts=bool(conflicts),
        conflicts=conflicts,
        checked_days=len(draft.days),
    )


@tool
def get_weather_forecast_tool(
    destination: str,
    start_date: str,
    num_days: int,
) -> dict[str, Any]:
    """Geocode destination and return Open-Meteo daily forecast for the requested dates."""
    geo = geocode_destination(destination)
    days, mode = fetch_weather_forecast(geo, start_date, num_days)
    return {
        "geo_location": geo.model_dump(),
        "weather_mode": mode,
        "forecast": [d.model_dump() for d in days],
    }

