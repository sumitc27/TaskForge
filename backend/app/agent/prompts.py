"""Prompt builders for the agent's plan / reason / critic steps."""
from __future__ import annotations

import json

from .tools_registry import Tool


def tools_block(registry: dict[str, Tool]) -> str:
    lines = []
    for t in registry.values():
        tag = " (WRITE — needs human approval)" if t.write else ""
        lines.append(f"- {t.name}{tag}: {t.description} args={t.args_hint}")
    return "\n".join(lines)


PLAN_SYSTEM = """You are TaskForge, an autonomous research-and-action agent.
Given a task, write a SHORT numbered plan (2-4 steps) describing how you'll use
the available tools to accomplish it. Be concrete. Output ONLY the plan text."""


def plan_user(task: str, registry: dict[str, Tool]) -> list[dict]:
    return [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": f"TASK: {task}\n\nAVAILABLE TOOLS:\n{tools_block(registry)}\n\nWrite the plan."},
    ]


REASON_SYSTEM = """You are TaskForge, an autonomous agent that works in a loop:
think, then either call ONE tool or give the final answer.

Respond with STRICT JSON and nothing else. Two allowed shapes:

To use a tool:
{{"thought": "why this step", "action": {{"tool": "<tool_name>", "args": {{...}}}}}}

To finish:
{{"thought": "why you're done", "final_answer": "the complete answer to the task"}}

Rules:
- Use ONLY tools from the list. Match the args shape exactly.
- If the task needs current/external information and you have NOT yet gathered it
  (no observations in the history below), you MUST call a tool — start with
  web_search. Do NOT finish, and do NOT claim you searched, until a tool result
  actually appears in the history.
- Typical flow: web_search to find sources → read_url on the best result(s) →
  then finish. Use the real URLs returned by web_search; never invent URLs.
- WRITE tools change real state and require human approval — only use them when
  the task clearly asks you to take that action.
- CRITICAL — WRITE ACTIONS: If the task asks you to create, add, or modify data
  (e.g. "add a task", "create a task", "mark as done"), you MUST call that WRITE
  tool via the action field. Do NOT write "I added a task" or "Task created:" in
  a final_answer unless that tool's observation already appears in the HISTORY
  above. Describing an action is NOT the same as performing it.
- The web_search results include SNIPPETS that often already contain the answer.
  If read_url fails or is blocked (e.g. HTTP 403), do NOT give up: either try a
  different URL from the results, or answer directly from the search snippets.
- Don't repeat an identical tool call. If a tool errors, adapt (different query/URL)
  rather than giving up. Never end by only saying a tool failed.
- Only FINISH once the history contains enough evidence; then write a complete,
  well-structured answer that directly addresses the task and cite your sources.
- Format the final_answer as clean Markdown (headings, bullet lists, bold) when
  it aids clarity; keep simple answers as plain prose."""


def reason_user(state: dict, registry: dict[str, Tool]) -> list[dict]:
    scratch = _format_scratchpad(state.get("scratchpad", []))
    remaining = state["max_steps"] - state["step"]
    finish_nudge = ""
    if remaining <= 1:
        finish_nudge = (
            "\n\nYou are almost out of steps — unless a final action is required, "
            "FINISH now with your best answer from what you've gathered."
        )

    # If the most recent scratchpad entry is a critic revision, surface it as an
    # unmissable mandate so even a weaker fallback model cannot ignore it.
    critic_mandate = ""
    scratchpad = state.get("scratchpad", [])
    if scratchpad and scratchpad[-1].get("tool") == "_critique":
        fb = scratchpad[-1].get("observation", "")
        critic_mandate = (
            f"\n\n⚠️  MANDATORY — CRITIC REQUIRED ACTION:\n"
            f"{fb}\n"
            "You MUST resolve this before you are allowed to output a final_answer. "
            "If the feedback says to call a tool, call that tool NOW as your next action."
        )

    content = (
        f"TASK: {state['task']}\n\n"
        f"PLAN:\n{state.get('plan','(none)')}\n\n"
        f"AVAILABLE TOOLS:\n{tools_block(registry)}\n\n"
        f"HISTORY SO FAR:\n{scratch or '(nothing yet)'}\n\n"
        f"Steps used: {state['step']}/{state['max_steps']}."
        f"{finish_nudge}"
        f"{critic_mandate}\n\n"
        "Respond with the next JSON action or the final answer."
    )
    return [
        {"role": "system", "content": REASON_SYSTEM},
        {"role": "user", "content": content},
    ]


CRITIC_SYSTEM = """You are a strict critic reviewing whether an agent's answer
fully and correctly addresses the task. Respond with STRICT JSON:
{"verdict": "accept" | "revise", "reason": "one sentence", "feedback": "if revise, what's missing or wrong"}
Accept if the answer is complete, relevant, and supported by the work done.
Ask to revise only for a real, fixable gap — not for polish.

WRITE-ACTION CHECK (most important): If the task asked the agent to create, add,
or modify data (e.g. "add a task", "create a task") but the WORK DONE section
contains NO call to a WRITE tool (add_task, complete_task), then the agent only
described the action without performing it. That is ALWAYS a "revise" — feedback
must say: "You described adding a task but never called add_task. Call the tool now." """


def critic_user(state: dict) -> list[dict]:
    scratch = _format_scratchpad(state.get("scratchpad", []))
    # Summarise which WRITE tools were actually called so the critic can check.
    write_tools_used = [
        s["tool"] for s in state.get("scratchpad", [])
        if s.get("tool") and s["tool"] not in ("_critique",)
        and not s.get("observation", "").startswith("ERROR")
    ]
    write_note = (
        f"WRITE tools actually called in this session: {write_tools_used or 'none'}"
    )
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": (
            f"TASK: {state['task']}\n\n"
            f"WORK DONE:\n{scratch or '(none)'}\n\n"
            f"{write_note}\n\n"
            f"PROPOSED FINAL ANSWER:\n{state.get('final_answer','')}\n\n"
            "Judge it."
        )},
    ]


def _format_scratchpad(scratch: list[dict]) -> str:
    out = []
    for i, s in enumerate(scratch, start=1):
        if s.get("tool") == "_critique":
            out.append(f"[Critic feedback] {s.get('observation','')}")
            continue
        args = json.dumps(s.get("args", {}), ensure_ascii=False)
        obs = s.get("observation", "")
        if len(obs) > 1200:
            obs = obs[:1200] + "…[truncated]"
        out.append(
            f"Step {i}: thought={s.get('thought','')}\n"
            f"  action={s.get('tool','')}({args})\n  observation={obs}"
        )
    return "\n".join(out)
