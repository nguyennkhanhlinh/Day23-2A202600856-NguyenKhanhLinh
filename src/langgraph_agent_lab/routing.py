"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.
"""

from __future__ import annotations

from .state import AgentState, Route

# Map a classified route value → the name of the next node in the graph.
_CLASSIFY_TO_NODE = {
    Route.SIMPLE.value: "answer",
    Route.TOOL.value: "tool",
    Route.MISSING_INFO.value: "clarify",
    Route.RISKY.value: "risky_action",
    Route.ERROR.value: "retry",
}


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node (default → answer)."""
    return _CLASSIFY_TO_NODE.get(state.get("route", ""), "answer")


def route_after_evaluate(state: AgentState) -> str:
    """The retry-loop gate: retry on a bad tool result, otherwise answer."""
    if state.get("evaluation_result") == "needs_retry":
        return "retry"
    return "answer"


def route_after_retry(state: AgentState) -> str:
    """Bounded retry: try the tool again until attempts run out, then dead-letter."""
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    if attempt < max_attempts:
        return "tool"
    return "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Proceed to the tool if approved, otherwise ask the user for an alternative."""
    approval = state.get("approval") or {}
    if approval.get("approved"):
        return "tool"
    return "clarify"
