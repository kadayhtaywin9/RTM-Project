"""Cache helpers that also work when modules are used outside Streamlit."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def cache_data(*, ttl: int | None = None, max_entries: int = 16) -> Callable[[F], F]:
    """Use Streamlit's data cache when available and a no-op elsewhere."""
    try:
        import streamlit as st

        return st.cache_data(ttl=ttl, max_entries=max_entries, show_spinner=False)  # type: ignore[return-value]
    except ImportError:
        def decorator(func: F) -> F:
            return func

        return decorator


def cache_resource(*, max_entries: int = 8) -> Callable[[F], F]:
    """Use Streamlit's resource cache for models and API clients."""
    try:
        import streamlit as st

        return st.cache_resource(max_entries=max_entries, show_spinner=False)  # type: ignore[return-value]
    except ImportError:
        def decorator(func: F) -> F:
            return func

        return decorator
