"""DuckDuckGo (ddgs) web search helpers and LangChain tool wrapper."""

from __future__ import annotations

from typing import Literal

from ddgs import DDGS
from langchain_core.tools import tool

from trippilot.state import SearchSnippet


def search_travel_web(
    query: str,
    max_results: int = 4,
    category: Literal["general", "indoor_backup", "budget"] = "general",
) -> list[SearchSnippet]:
    """Run a DuckDuckGo text search and return normalized SearchSnippet items."""
    clean_query = query.strip()
    if not clean_query:
        return []

    snippets: list[SearchSnippet] = []
    with DDGS() as ddgs:
        raw_results = list(
            ddgs.text(
                clean_query,
                region="us-en",
                safesearch="moderate",
                max_results=max_results,
            )
        )
    for item in raw_results:
        title = str(item.get("title") or "").strip()
        href = str(item.get("href") or item.get("link") or "").strip()
        body = str(item.get("body") or item.get("snippet") or "").strip()
        if title or body:
            snippets.append(
                SearchSnippet(
                    query=clean_query,
                    title=title or "Web result",
                    url=href,
                    snippet=body[:320],
                    category=category,
                )
            )
    return snippets


def _run_safe_queries(
    queries: list[tuple[str, Literal["general", "indoor_backup", "budget"]]],
    max_per_query: int = 3,
) -> tuple[list[SearchSnippet], list[str]]:
    """Execute multiple search queries with non-fatal error isolation."""
    collected: list[SearchSnippet] = []
    warnings: list[str] = []
    seen_urls: set[str] = set()

    for query, category in queries:
        try:
            hits = search_travel_web(query, max_results=max_per_query, category=category)
            for hit in hits:
                dedup_key = hit.url or hit.title
                if dedup_key not in seen_urls:
                    seen_urls.add(dedup_key)
                    collected.append(hit)
        except Exception as exc:
            warnings.append(f"DuckDuckGo search degraded for '{query}': {exc}")

    return collected, warnings


def gather_destination_snippets(
    destination: str,
    preferences: list[str],
    currency: str,
) -> tuple[list[SearchSnippet], list[str]]:
    """Gather initial attraction, rainy-day, and budget cost snippets for a destination."""
    pref_text = " ".join(preferences[:3]) if preferences else "sightseeing local food"
    queries: list[tuple[str, Literal["general", "indoor_backup", "budget"]]] = [
        (f"best things to do in {destination} {pref_text}", "general"),
        (f"best indoor attractions museums cafes {destination} rainy day", "indoor_backup"),
        (f"{destination} average daily travel budget hotel food cost {currency}", "budget"),
    ]
    return _run_safe_queries(queries, max_per_query=3)


def gather_replan_snippets(
    destination: str,
    need_indoor: bool,
    need_budget: bool,
    currency: str,
) -> tuple[list[SearchSnippet], list[str]]:
    """Fetch targeted indoor or budget-saving alternatives when self-correction loops."""
    queries: list[tuple[str, Literal["general", "indoor_backup", "budget"]]] = []
    if need_indoor:
        queries.append(
            (f"top indoor activities museums workshops markets in {destination}", "indoor_backup")
        )
    if need_budget:
        queries.append(
            (f"free cheap things to do budget stays street food {destination} {currency}", "budget")
        )
    if not queries:
        return [], []
    return _run_safe_queries(queries, max_per_query=3)


@tool
def web_search_places(query: str, max_results: int = 4) -> list[dict[str, str]]:
    """Search DuckDuckGo for travel attractions, indoor alternatives, and cost estimates."""
    try:
        return [s.model_dump() for s in search_travel_web(query, max_results=max_results)]
    except Exception as exc:
        return [{"query": query, "title": "Search error", "url": "", "snippet": str(exc)}]

