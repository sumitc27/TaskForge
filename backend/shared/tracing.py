"""Langfuse tracing setup."""
from __future__ import annotations

import functools
import os
from typing import Any, Callable, Optional

_LANGFUSE_AVAILABLE = False
try:
    from langfuse import Langfuse, observe as _lf_observe  # type: ignore

    _LANGFUSE_AVAILABLE = True
except Exception:
    Langfuse = None  # type: ignore
    _lf_observe = None  # type: ignore


_client: Optional["Langfuse"] = None


def tracing_enabled() -> bool:
    return (
        _LANGFUSE_AVAILABLE
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
    )


def get_client() -> Optional["Langfuse"]:
    global _client
    if not tracing_enabled():
        return None
    if _client is None:
        _client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    return _client


def observe(*, name: Optional[str] = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        if tracing_enabled() and _lf_observe is not None:
            return _lf_observe(name=name or func.__name__)(func)

        @functools.wraps(func)
        def passthrough(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return passthrough

    return decorator


def flush() -> None:
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
