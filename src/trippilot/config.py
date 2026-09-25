"""Configuration, secrets resolution, and LLM provider factory for TripPilot."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_RAIN_PROB_THRESHOLD = 50
DEFAULT_MAX_REVISIONS = 3
DEFAULT_DB_PATH = "trippilot_checkpoints.db"
DEFAULT_GEMINI_MODEL = os.getenv("TRIPPILOT_GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_GROQ_MODEL = os.getenv("TRIPPILOT_GROQ_MODEL", "openai/gpt-oss-120b")

CURRENCY_SYMBOLS: dict[str, str] = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
}


def format_currency(amount: float, currency: str = "INR") -> str:
    """Format a monetary amount with its currency symbol."""
    symbol = CURRENCY_SYMBOLS.get(currency.upper(), f"{currency.upper()} ")
    if currency.upper() in {"INR", "JPY"}:
        return f"{symbol}{amount:,.0f}"
    return f"{symbol}{amount:,.2f}"


def get_secret(key: str, default: str = "") -> str:
    """Resolve a secret from environment variables or Streamlit secrets."""
    env_val = os.getenv(key)
    if env_val:
        return env_val.strip()
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default


def resolve_provider(preferred: str = "auto", api_key_override: str = "") -> str:
    """Pick an active provider ('gemini', 'groq', or 'demo') based on preference and keys."""
    pref = (preferred or "auto").lower().strip()
    if pref == "demo":
        return "demo"
    if pref == "gemini":
        if api_key_override or get_secret("GOOGLE_API_KEY"):
            return "gemini"
        return "demo"
    if pref == "groq":
        if api_key_override or get_secret("GROQ_API_KEY"):
            return "groq"
        return "demo"

    # Auto mode: prefer Gemini, then Groq, then Demo
    if get_secret("GOOGLE_API_KEY"):
        return "gemini"
    if get_secret("GROQ_API_KEY"):
        return "groq"
    return "demo"


def build_chat_model(
    provider: str,
    model_name: str = "",
    api_key: str = "",
    temperature: float = 0.2,
) -> Any:
    """Instantiate a LangChain chat model for Gemini or Groq."""
    norm = provider.lower().strip()
    if norm == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = api_key.strip() or get_secret("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY is required for the Gemini provider.")
        return ChatGoogleGenerativeAI(
            model=model_name.strip() or DEFAULT_GEMINI_MODEL,
            google_api_key=key,
            temperature=temperature,
            max_output_tokens=8192,
        )

    if norm == "groq":
        from langchain_groq import ChatGroq

        key = api_key.strip() or get_secret("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is required for the Groq provider.")
        selected_model = model_name.strip() or DEFAULT_GROQ_MODEL
        groq_kwargs: dict[str, Any] = {
            "model": selected_model,
            "groq_api_key": key,
            "temperature": temperature,
            "max_tokens": 3000,
        }
        if any(tag in selected_model.lower() for tag in ("gpt-oss", "qwen")):
            groq_kwargs["reasoning_effort"] = "low"
        return ChatGroq(**groq_kwargs)

    raise ValueError(f"Unsupported LLM provider for live calls: {provider}")

