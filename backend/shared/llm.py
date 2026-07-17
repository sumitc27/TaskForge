"""Single LLM entrypoint."""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Iterator, Optional

import litellm
from litellm import completion
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from .tracing import observe


PRIMARY_MODEL    = os.getenv("PRIMARY_MODEL",    "groq/openai/gpt-oss-120b")
FALLBACK_MODEL   = os.getenv("FALLBACK_MODEL",   "gemini/gemini-3.5-flash")
FALLBACK_MODEL_2 = os.getenv("FALLBACK_MODEL_2", "gemini/gemini-3.1-flash-lite")

_RETRYABLE = (
    RateLimitError,
    ServiceUnavailableError,
    InternalServerError,
    APIConnectionError,
    Timeout,
)

litellm.drop_params = True
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

_log = logging.getLogger("llm.router")


def _backoff_sleep(attempt: int, base: float = 1.0, cap: float = 60.0) -> None:
    delay = min(cap, base * (2 ** attempt))
    time.sleep(random.uniform(0, delay))


def _call(model: str, messages: list[dict], stream: bool = False, **kwargs: Any):
    return completion(model=model, messages=messages, stream=stream, **kwargs)


def _attempt_model(
    model: str,
    messages: list[dict],
    *,
    max_retries: int,
    stream: bool,
    **kwargs: Any,
):
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return _call(model, messages, stream=stream, **kwargs)
        except _RETRYABLE as err:
            last_err = err
            if attempt < max_retries:
                _backoff_sleep(attempt)
            continue
    assert last_err is not None
    raise last_err


def _model_chain(
    primary: str,
    fallback_model: Optional[str],
) -> list[str]:
    seen: set[str] = set()
    chain: list[str] = []
    for m in [primary, fallback_model, FALLBACK_MODEL_2]:
        if m and m not in seen:
            seen.add(m)
            chain.append(m)
    return chain


@observe(name="llm-chat")
def chat(
    messages: list[dict],
    *,
    model: str = PRIMARY_MODEL,
    fallback_model: Optional[str] = FALLBACK_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    max_retries: int = 4,
    metadata: Optional[dict] = None,
    **kwargs: Any,
) -> str:
    if metadata:
        kwargs.setdefault("metadata", metadata)

    chain = _model_chain(model, fallback_model)
    last_err: Optional[Exception] = None
    for i, m in enumerate(chain):
        if i > 0:
            cooldown = random.uniform(2, 6)
            _log.warning("provider %s exhausted — waiting %.1fs then trying %s", chain[i - 1], cooldown, m)
            time.sleep(cooldown)
        try:
            resp = _attempt_model(
                m, messages, max_retries=max_retries, stream=False,
                temperature=temperature, max_tokens=max_tokens, **kwargs,
            )
            if i > 0:
                _log.info("succeeded on fallback provider %s", m)
            return resp.choices[0].message.content or ""
        except _RETRYABLE as err:
            last_err = err
            _log.warning("provider %s failed after %d retries: %s", m, max_retries, str(err)[:120])
            continue

    assert last_err is not None
    raise last_err


@observe(name="llm-chat-stream")
def chat_stream(
    messages: list[dict],
    *,
    model: str = PRIMARY_MODEL,
    fallback_model: Optional[str] = FALLBACK_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    max_retries: int = 2,
    metadata: Optional[dict] = None,
    **kwargs: Any,
) -> Iterator[str]:
    if metadata:
        kwargs.setdefault("metadata", metadata)

    emitted = 0

    def _iter(m: str):
        nonlocal emitted
        resp = _attempt_model(
            m, messages, max_retries=max_retries, stream=True,
            temperature=temperature, max_tokens=max_tokens, **kwargs,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                emitted += 1
                yield delta

    chain = _model_chain(model, fallback_model)
    last_err: Optional[Exception] = None
    for i, m in enumerate(chain):
        if emitted:
            break
        if i > 0:
            cooldown = random.uniform(2, 6)
            _log.warning("stream: provider %s exhausted — waiting %.1fs then trying %s",
                         chain[i - 1], cooldown, m)
            time.sleep(cooldown)
        try:
            yield from _iter(m)
            return
        except _RETRYABLE as err:
            if emitted:
                raise
            last_err = err
            _log.warning("stream: provider %s failed: %s", m, str(err)[:120])
            continue

    if last_err is not None:
        raise last_err


def health_check() -> dict:
    status = {}
    for label, m in (
        ("primary",    PRIMARY_MODEL),
        ("fallback",   FALLBACK_MODEL),
        ("fallback_2", FALLBACK_MODEL_2),
    ):
        try:
            chat(
                [{"role": "user", "content": "ping"}],
                model=m,
                fallback_model=None,
                max_tokens=1,
                max_retries=0,
            )
            status[label] = {"model": m, "ok": True}
        except Exception as err:
            status[label] = {"model": m, "ok": False, "error": str(err)[:200]}
    return status
