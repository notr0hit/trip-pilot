"""Unit tests for Open-Meteo geocoding, forecasting, archive fallback, and weather audit."""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from trippilot.state import ActivityItem, DayPlan, DayWeather, GeoLocation, ItineraryDraft
from trippilot.tools.weather import (
    audit_weather,
    decode_wmo_code,
    fetch_weather_forecast,
    geocode_destination,
)


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_decode_wmo_code() -> None:
    assert decode_wmo_code(0) == "Clear sky"
    assert decode_wmo_code(65) == "Heavy rain"
    assert "999" in decode_wmo_code(999)


def test_geocode_destination_with_comma_fallback() -> None:
    seen_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.params.get("name", "")
        seen_queries.append(name)
        if "," in name:
            return httpx.Response(200, json={"results": []})
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Goa",
                        "country": "India",
                        "admin1": "Goa",
                        "latitude": 15.49,
                        "longitude": 73.82,
                        "timezone": "Asia/Kolkata",
                    }
                ]
            },
        )

    with _mock_client(handler) as client:
        geo = geocode_destination("Goa, India", client=client)

    assert seen_queries == ["Goa, India", "Goa"]
    assert geo.name == "Goa"
    assert geo.country == "India"
    assert geo.latitude == pytest.approx(15.49)


def test_geocode_destination_not_found_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    with _mock_client(handler) as client:
        with pytest.raises(ValueError, match="Could not geocode"):
            geocode_destination("Atlantis", client=client)


def test_fetch_weather_live_and_simulated_override() -> None:
    geo = GeoLocation(name="Goa", country="India", latitude=15.49, longitude=73.82, timezone="Asia/Kolkata")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "api.open-meteo.com/v1/forecast" in str(request.url)
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": ["2026-11-27", "2026-11-28", "2026-11-29"],
                    "weather_code": [0, 1, 61],
                    "temperature_2m_max": [31.0, 30.5, 28.0],
                    "temperature_2m_min": [23.0, 22.5, 22.0],
                    "precipitation_probability_max": [10, 15, 70],
                    "precipitation_sum": [0.0, 0.0, 8.4],
                }
            },
        )

    with _mock_client(handler) as client:
        days, mode = fetch_weather_forecast(
            geo=geo,
            start_date="2026-11-27",
            num_days=3,
            rain_prob_threshold=50,
            simulate_rain_day=2,
            client=client,
            today_override=date(2026, 11, 26),
        )

    assert mode == "simulated"
    assert days[0].is_rain_risk is False
    assert days[1].is_rain_risk is True
    assert days[1].source_mode == "simulated"
    assert days[2].is_rain_risk is True


def test_fetch_weather_far_future_uses_archive_proxy() -> None:
    geo = GeoLocation(name="Kyoto", country="Japan", latitude=35.01, longitude=135.76, timezone="Asia/Tokyo")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "archive-api.open-meteo.com/v1/archive" in str(request.url)
        assert request.url.params["start_date"] == "2026-06-10"
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": ["2026-06-10", "2026-06-11"],
                    "weather_code": [0, 63],
                    "temperature_2m_max": [27.0, 24.0],
                    "temperature_2m_min": [18.0, 17.0],
                    "precipitation_sum": [0.0, 12.0],
                }
            },
        )

    with _mock_client(handler) as client:
        days, mode = fetch_weather_forecast(
            geo=geo,
            start_date="2027-06-10",
            num_days=2,
            client=client,
            today_override=date(2026, 11, 26),
        )

    assert mode == "seasonal_proxy"
    assert [d.date for d in days] == ["2027-06-10", "2027-06-11"]
    assert days[0].is_rain_risk is False
    assert days[1].is_rain_risk is True


def test_audit_weather_detects_unsafe_outdoor_only() -> None:
    forecast = [
        DayWeather(
            date="2026-11-27",
            weather_code=0,
            condition_label="Clear sky",
            temp_min_c=22.0,
            temp_max_c=30.0,
            precip_prob_max=10,
            is_rain_risk=False,
        ),
        DayWeather(
            date="2026-11-28",
            weather_code=65,
            condition_label="Heavy rain",
            temp_min_c=21.0,
            temp_max_c=26.0,
            precip_prob_max=85,
            is_rain_risk=True,
        ),
    ]
    draft = ItineraryDraft(
        destination="Goa",
        currency="INR",
        travelers=2,
        days=[
            DayPlan(
                day_number=1,
                date="2026-11-27",
                theme="Beach",
                activities=[
                    ActivityItem(
                        title="Beach Walk",
                        time_slot="morning",
                        category="outdoor",
                        estimated_cost=500,
                        weather_safe_if_rain=False,
                    )
                ],
            ),
            DayPlan(
                day_number=2,
                date="2026-11-28",
                theme="Rainy Day",
                activities=[
                    ActivityItem(
                        title="Kayaking",
                        time_slot="morning",
                        category="outdoor",
                        estimated_cost=2000,
                        weather_safe_if_rain=False,
                    ),
                    ActivityItem(
                        title="Art Museum",
                        time_slot="afternoon",
                        category="indoor",
                        estimated_cost=600,
                        weather_safe_if_rain=True,
                    ),
                ],
            ),
        ],
    )

    audit = audit_weather(draft, forecast)
    assert audit.has_conflicts is True
    assert len(audit.conflicts) == 1
    assert audit.conflicts[0].day_number == 2
    assert audit.conflicts[0].conflicting_activities == ["Kayaking"]

