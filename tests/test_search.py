"""Unit tests for DuckDuckGo search helpers and error isolation."""

from __future__ import annotations

from unittest.mock import patch

from trippilot.tools.search import (
    gather_destination_snippets,
    gather_replan_snippets,
    search_travel_web,
)


class _FakeDDGS:
    def __init__(self, results=None, error: Exception | None = None) -> None:
        self._results = results or []
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query, region="wt-wt", safesearch="moderate", max_results=4):
        if self._error:
            raise self._error
        return self._results[:max_results]


def test_search_travel_web_normalizes_results() -> None:
    fake_rows = [
        {"title": "Goa Museum", "href": "https://example.com/museum", "body": "Great indoor gallery."},
        {"title": "Spice Farm", "href": "https://example.com/spice", "body": "Covered tour."},
    ]
    with patch("trippilot.tools.search.DDGS", return_value=_FakeDDGS(fake_rows)):
        results = search_travel_web("indoor attractions Goa", max_results=2, category="indoor_backup")

    assert len(results) == 2
    assert results[0].url == "https://example.com/museum"
    assert results[0].category == "indoor_backup"


def test_gather_destination_snippets_deduplicates_and_handles_errors() -> None:
    with patch("trippilot.tools.search.DDGS", return_value=_FakeDDGS(error=RuntimeError("rate limited"))):
        snippets, warnings = gather_destination_snippets("Goa", ["beaches"], "INR")

    assert snippets == []
    assert len(warnings) == 3
    assert "rate limited" in warnings[0]


def test_gather_replan_snippets_targets_requested_modes() -> None:
    fake_rows = [
        {"title": "Indoor Cafe", "href": "https://example.com/cafe", "body": "Cozy rainy-day spot."}
    ]
    with patch("trippilot.tools.search.DDGS", return_value=_FakeDDGS(fake_rows)):
        snippets, warnings = gather_replan_snippets("Goa", need_indoor=True, need_budget=False, currency="INR")

    assert warnings == []
    assert len(snippets) == 1
    assert snippets[0].category == "indoor_backup"

