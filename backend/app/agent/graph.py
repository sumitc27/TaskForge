"""The TaskForge agent as a LangGraph state graph."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..config import get_settings
from ..llm_compat import chat, chat_stream
from .prompts import critic_user, plan_user, reason_user
from .state import AgentState
from .tools_registry import Tool, build_registry

logger = logging.getLogger("taskforge.agent")

_registry: dict[str, Tool] | None = None
_reg_lock = asyncio.Lock()
_WRITE_TOOLS: set[str] = set()


async def get_registry() -> dict[str, Tool]:
    global _registry
    async with _reg_lock:
        if _registry is None:
            _registry = await build_registry()
            _WRITE_TOOLS.update(n for n, t in _registry.items() if t.write)
    return _registry


def _emit(etype: str, **data) -> None:
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None
    if writer is not None:
        writer({"type": etype, "data": data})


async def _llm(messages: list[dict], *, max_tokens: int = 900, temperature: float = 0.2) -> str:
    s = get_settings()
    t0 = time.monotonic()
    _emit("status", phase="waiting_provider")
    result = await asyncio.to_thread(
        lambda: chat(
            messages,
            model=s.primary_model,
            fallback_model=s.fallback_model,
            temperature=temperature, max_tokens=max_tokens,
            metadata={"trace_name": "taskforge-agent"},
        )
    )
    elapsed = time.monotonic() - t0
    logger.debug("[llm] tokens≈%d elapsed=%.2fs", len(result) // 4, elapsed)
    if elapsed > 10:
        logger.warning("[llm] slow call: %.1fs (likely backoff/fallback)", elapsed)
    return result


def parse_json(text: str) -> dict:
    if not text:
        return {}
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return {}


_PLACEHOLDER_RE = re.compile(
    r"\[[A-Za-z][A-Za-z]*(?:\s+[A-Za-z][A-Za-z]*){1,5}\]"
    r"|\{\{.{1,40}?\}\}"
    r"|<[A-Za-z][A-Za-z]*(?:\s+[A-Za-z][A-Za-z]*){1,5}>"
)


def _find_placeholder(args: dict) -> str | None:
    for v in args.values():
        if isinstance(v, str):
            m = _PLACEHOLDER_RE.search(v)
            if m:
                return m.group(0)
    return None


async def plan_node(state: AgentState) -> dict:
    registry = await get_registry()
    logger.info("[plan] task=%r tools=%s", state["task"][:80], list(registry))
    _emit("status", phase="planning")
    t0 = time.monotonic()
    plan = (await _llm(plan_user(state["task"], registry), max_tokens=300)).strip()
    logger.info("[plan] done in %.2fs", time.monotonic() - t0)
    _emit("plan", text=plan)
    return {
        "plan": plan, "scratchpad": [], "step": 0, "critiques": 0,
        "pending": None, "final_answer": None, "forced_finish": False,
        "status": "running",
    }


async def reason_node(state: AgentState) -> dict:
    registry = await get_registry()
    step = state["step"]
    max_steps = state["max_steps"]
    logger.info("[reason] step=%d/%d", step, max_steps)
    _emit("status", phase="reasoning", step=step, max_steps=max_steps)

    if step >= max_steps:
        logger.warning("[reason] step limit reached at %d", step)
        _emit("thought", text="Step limit reached — composing the final answer.")
        final = await _force_final(state)
        return {"final_answer": final, "forced_finish": True, "pending": None}

    t0 = time.monotonic()
    messages = reason_user(state, registry)
    raw = await _llm(messages, max_tokens=900)
    decision = parse_json(raw)
    if not decision:
        logger.warning("[reason] step=%d: bad JSON from LLM, retrying", step)
        retry = messages + [
            {"role": "assistant", "content": raw[:500]},
            {"role": "user", "content": "Your response was not valid JSON. Reply "
             "again with ONLY the JSON object — either an action or a final_answer."},
        ]
        raw = await _llm(retry, max_tokens=900)
        decision = parse_json(raw)
    logger.info("[reason] step=%d decision=%s elapsed=%.2fs",
                step, "action" if "action" in decision else "finish", time.monotonic() - t0)
    thought = (decision.get("thought") or "").strip()

    if "action" in decision and isinstance(decision["action"], dict):
        action = decision["action"]
        tool = action.get("tool", "")
        args = action.get("args", {}) or {}

        bad_value = _find_placeholder(args)
        if bad_value:
            logger.warning("[reason] step=%d: unresolved placeholder in args: %r", step, bad_value)
            nudge = messages + [
                {"role": "assistant", "content": raw[:500]},
                {"role": "user", "content": (
                    f"Your action's arguments contain what looks like an unresolved "
                    f"template placeholder ({bad_value!r}) instead of real content. "
                    "Never write placeholder text into a tool call. Use the actual "
                    "value from the HISTORY above, or call a read tool first if you "
                    "don't have the real value yet."
                )},
            ]
            raw2 = await _llm(nudge, max_tokens=900)
            decision2 = parse_json(raw2)
            fixed = decision2.get("action")
            if isinstance(fixed, dict) and fixed.get("tool"):
                tool = fixed.get("tool", tool)
                args = fixed.get("args", args) or args
                thought = (decision2.get("thought") or thought).strip()

        logger.info("[reason] step=%d → tool=%s args=%s", step, tool, json.dumps(args)[:120])
        _emit("thought", text=thought or f"Using {tool}.", step=step, max_steps=max_steps)
        return {"pending": {"tool": tool, "args": args, "thought": thought}}

    scratchpad = state.get("scratchpad", [])
    used_a_tool = any(
        s.get("tool") and s.get("tool") != "_critique" for s in scratchpad
    )
    just_revised = bool(scratchpad) and scratchpad[-1].get("tool") == "_critique"

    if not used_a_tool and not state.get("forced_finish"):
        logger.warning("[reason] step=%d: premature finish guard triggered", step)
        nudge = messages + [
            {"role": "assistant", "content": raw[:500]},
            {"role": "user", "content": "You have not called any tool yet, so you "
             "have no evidence to answer from. Do NOT finish and do NOT claim you "
             "already searched. Respond with ONLY a JSON action that calls a tool "
             "(e.g. web_search) to gather what the task needs."},
        ]
        raw2 = await _llm(nudge, max_tokens=900)
        forced = parse_json(raw2)
        act = forced.get("action")
        if isinstance(act, dict) and act.get("tool"):
            th = (forced.get("thought") or "").strip()
            logger.info("[reason] premature-finish guard → forced tool=%s", act["tool"])
            _emit("thought", text=th or f"Using {act['tool']}.", step=step, max_steps=max_steps)
            return {"pending": {"tool": act["tool"], "args": act.get("args", {}) or {},
                                "thought": th}}

    elif just_revised and not state.get("forced_finish"):
        logger.warning("[reason] step=%d: critic mandate ignored, forcing compliance", step)
        feedback = scratchpad[-1].get("observation", "")
        nudge = messages + [
            {"role": "assistant", "content": raw[:500]},
            {"role": "user", "content": (
                "You did not comply with the critic's required action:\n"
                f"{feedback}\n\n"
                "Do NOT finish, and do NOT claim to have already called a tool "
                "you have not called — check the HISTORY above for what "
                "actually happened. Respond with ONLY a JSON action that calls "
                "the tool the feedback requires."
            )},
        ]
        raw2 = await _llm(nudge, max_tokens=900)
        forced = parse_json(raw2)
        act = forced.get("action")
        if isinstance(act, dict) and act.get("tool"):
            th = (forced.get("thought") or "").strip()
            logger.info("[reason] critic-mandate guard → forced tool=%s", act["tool"])
            _emit("thought", text=th or f"Using {act['tool']}.", step=step, max_steps=max_steps)
            return {"pending": {"tool": act["tool"], "args": act.get("args", {}) or {},
                                "thought": th}}

    final = (decision.get("final_answer") or "").strip()
    if not final:
        logger.warning("[reason] step=%d: empty final_answer, forcing synthesis", step)
        final = (await _force_final(state)).strip()
    if not final:
        final = ("I couldn't gather enough information to answer that. Try rephrasing "
                 "the task, or check that web access is available.")
    logger.info("[reason] step=%d → finishing (answer_len=%d)", step, len(final))
    _emit("thought", text=thought or "Finishing.", step=step, max_steps=max_steps)
    return {"final_answer": final, "pending": None}


def _collect_sources(state: AgentState) -> list[dict]:
    scratch = state.get("scratchpad", [])
    titles: dict[str, str] = {}
    for s in scratch:
        if s.get("tool") == "web_search" and not s.get("observation", "").startswith("ERROR"):
            lines = s["observation"].splitlines()
            for i, line in enumerate(lines):
                url = line.strip()
                if url.startswith("http") and i > 0:
                    titles[url] = lines[i - 1].split(". ", 1)[-1].strip()

    sources: list[dict] = []
    seen_urls: set[str] = set()
    for s in scratch:
        tool, obs = s.get("tool"), s.get("observation", "")
        if obs.startswith("ERROR"):
            continue
        if tool == "read_url":
            url = s.get("args", {}).get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                host = url.split("//", 1)[-1].split("/", 1)[0]
                sources.append({"kind": "page", "url": url, "title": titles.get(url) or host})
        elif tool == "web_search":
            query = s.get("args", {}).get("query")
            if query:
                sources.append({"kind": "search", "query": query})
    return sources


async def _force_final(state: AgentState) -> str:
    from .prompts import _format_scratchpad
    msgs = [
        {"role": "system", "content": "You are out of tool steps. Using ONLY the "
         "history, write the best, complete final answer to the task. Output just "
         "the answer, formatted as clean Markdown when it aids clarity."},
        {"role": "user", "content": f"TASK: {state['task']}\n\nHISTORY:\n"
         f"{_format_scratchpad(state.get('scratchpad', []))}"},
    ]
    return (await _llm(msgs, max_tokens=800)).strip()


async def act_node(state: AgentState) -> dict:
    registry = await get_registry()
    pending = state["pending"] or {}
    tool, args = pending.get("tool", ""), pending.get("args", {})
    step_num = state["step"]
    logger.info("[act] step=%d tool=%s args=%s", step_num, tool, json.dumps(args)[:120])
    _emit("action", tool=tool, args=args, step=step_num, max_steps=state["max_steps"])
    _emit("status", phase="executing", tool=tool, step=step_num, max_steps=state["max_steps"])
    t = registry.get(tool)
    t0 = time.monotonic()
    if t is None:
        obs = f"ERROR: unknown tool '{tool}'. Available: {', '.join(registry)}"
        logger.error("[act] step=%d: unknown tool '%s'", step_num, tool)
    else:
        obs = await t.run(args)
        elapsed = time.monotonic() - t0
        if obs.startswith("ERROR"):
            logger.warning("[act] step=%d tool=%s ERROR in %.2fs: %s", step_num, tool, elapsed, obs[:120])
        else:
            logger.info("[act] step=%d tool=%s ok in %.2fs result_len=%d", step_num, tool, elapsed, len(obs))
    _emit("observation", tool=tool, result=obs, step=step_num, max_steps=state["max_steps"])
    entry = {"thought": pending.get("thought", ""), "tool": tool, "args": args, "observation": obs}
    return {
        "scratchpad": state["scratchpad"] + [entry],
        "step": step_num + 1,
        "pending": None,
    }


async def approval_node(state: AgentState) -> dict:
    pending = state["pending"] or {}
    tool_name = pending.get("tool")
    registry = await get_registry()
    tool_def = registry.get(tool_name)
    decision = interrupt({
        "tool": tool_name,
        "args": pending.get("args"),
        "thought": pending.get("thought"),
        "step": state["step"],
        "max_steps": state["max_steps"],
        "args_hint": tool_def.args_hint if tool_def else None,
        "schema": tool_def.schema if tool_def else None,
    })

    approved = decision.get("approved") if isinstance(decision, dict) else bool(decision)
    if approved:
        args = (decision.get("edited_args") if isinstance(decision, dict) else None) or pending.get("args", {})
        return {"pending": {**pending, "args": args, "_approved": True}}

    _emit("observation", tool=pending.get("tool"), result="Human REJECTED this action.")
    step = {
        "thought": pending.get("thought", ""), "tool": pending.get("tool", ""),
        "args": pending.get("args", {}),
        "observation": "Human REJECTED this action. Do not retry it; choose another approach or finish.",
    }
    return {"pending": None, "scratchpad": state["scratchpad"] + [step]}


async def critic_node(state: AgentState) -> dict:
    if state.get("forced_finish") or state["critiques"] >= state["max_critiques"]:
        logger.info("[critic] skipped (forced=%s critiques=%d)", state.get("forced_finish"), state["critiques"])
        return {"status": "done"}

    logger.info("[critic] evaluating answer (len=%d)", len(state.get("final_answer") or ""))
    t0 = time.monotonic()
    raw = await _llm(critic_user(state), max_tokens=300)
    verdict = parse_json(raw)
    v = (verdict.get("verdict") or "accept").lower()
    reason = (verdict.get("reason") or "").strip()
    logger.info("[critic] verdict=%s reason=%r elapsed=%.2fs", v, reason[:80], time.monotonic() - t0)
    _emit("critique", verdict=v, reason=reason)

    if v == "revise":
        fb = (verdict.get("feedback") or verdict.get("reason") or "Answer is incomplete.").strip()
        logger.info("[critic] revise feedback: %s", fb[:120])
        crit = {"thought": "", "tool": "_critique", "args": {}, "observation": fb}
        return {
            "critiques": state["critiques"] + 1,
            "final_answer": None,
            "scratchpad": state["scratchpad"] + [crit],
        }

    return {"status": "done"}


async def finalize_node(state: AgentState) -> dict:
    _emit("status", phase="finalizing")
    draft = state.get("final_answer") or ""
    sources = _collect_sources(state)
    messages = [
        {"role": "system", "content": (
            "Present the following answer to the user as clean Markdown. Keep "
            "every fact and claim exactly as given — do not add, remove, or "
            "change information. Improve formatting only (headings, lists, "
            "bold) where it aids clarity. Output only the presented answer."
        )},
        {"role": "user", "content": f"TASK: {state['task']}\n\nDRAFT ANSWER:\n{draft}"},
    ]

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()
    s = get_settings()

    def produce() -> None:
        try:
            for tok in chat_stream(
                messages, model=s.primary_model, fallback_model=s.fallback_model,
                temperature=0.2, max_tokens=900,
                metadata={"trace_name": "taskforge-finalize"},
            ):
                loop.call_soon_threadsafe(queue.put_nowait, tok)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    threading.Thread(target=produce, daemon=True).start()

    chunks: list[str] = []
    failed = False
    while True:
        item = await queue.get()
        if item is _DONE:
            break
        if isinstance(item, Exception):
            logger.warning("[finalize] stream failed, falling back to draft: %s", item)
            failed = True
            break
        chunks.append(item)
        _emit("final_token", text=item)

    full = "".join(chunks).strip() if (not failed and chunks) else draft
    _emit("final", text=full, sources=sources)
    return {"final_answer": full, "status": "done"}


def route_after_reason(state: AgentState) -> str:
    if state.get("final_answer") is not None:
        return "critic"
    pending = state.get("pending") or {}
    if pending.get("tool") in _WRITE_TOOLS:
        return "approval"
    return "act"


def route_after_approval(state: AgentState) -> str:
    pending = state.get("pending") or {}
    return "act" if pending.get("_approved") else "reason"


def route_after_critic(state: AgentState) -> str:
    return "reason" if state.get("final_answer") is None else "finalize"


def build_graph(checkpointer: BaseCheckpointSaver):
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("reason", reason_node)
    g.add_node("act", act_node)
    g.add_node("approval", approval_node)
    g.add_node("critic", critic_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "reason")
    g.add_conditional_edges("reason", route_after_reason, ["act", "approval", "critic"])
    g.add_conditional_edges("approval", route_after_approval, ["act", "reason"])
    g.add_edge("act", "reason")
    g.add_conditional_edges("critic", route_after_critic, ["reason", "finalize"])
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)


_graph = None
_agent_conn: aiosqlite.Connection | None = None
_graph_lock = asyncio.Lock()


async def get_graph():
    global _graph, _agent_conn
    async with _graph_lock:
        if _graph is None:
            _agent_conn = await aiosqlite.connect(get_settings().agent_db_path, timeout=10)
            await _agent_conn.execute("PRAGMA journal_mode=WAL")
            await _agent_conn.execute("PRAGMA busy_timeout=10000")
            saver = AsyncSqliteSaver(_agent_conn)
            await saver.setup()
            _graph = build_graph(saver)
    return _graph


async def close_graph() -> None:
    global _graph, _agent_conn
    if _agent_conn is not None:
        await _agent_conn.close()
    _graph = None
    _agent_conn = None
