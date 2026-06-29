"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit tests
that check schema/metrics can run even if students are still debugging graph wiring.
"""

from __future__ import annotations

from typing import Any

from .state import AgentState


def build_graph(checkpointer: Any | None = None) -> Any:
    """Build and compile the LangGraph support-ticket workflow.

    Topology:
        START → intake → classify → [route_after_classify]
          simple       → answer → finalize → END
          tool         → tool → evaluate → [route_after_evaluate]
                                              success     → answer → finalize → END
                                              needs_retry → retry → [route_after_retry]
                                                                      attempt<max → tool (loop)
                                                                      else        → dead_letter
                                                                                    → finalize → END
          missing_info → clarify → finalize → END
          risky        → risky_action → approval → [route_after_approval]
                                                      approved → tool → evaluate → ...
                                                      rejected → clarify → finalize → END
          error        → retry → [route_after_retry] → ...
    """
    from langgraph.graph import END, START, StateGraph

    from . import nodes
    from .routing import (
        route_after_approval,
        route_after_classify,
        route_after_evaluate,
        route_after_retry,
    )

    builder = StateGraph(AgentState)

    # 1) Register all 11 nodes. Node names here are what routing functions return.
    builder.add_node("intake", nodes.intake_node)
    builder.add_node("classify", nodes.classify_node)
    builder.add_node("tool", nodes.tool_node)
    builder.add_node("evaluate", nodes.evaluate_node)
    builder.add_node("answer", nodes.answer_node)
    builder.add_node("clarify", nodes.ask_clarification_node)
    builder.add_node("risky_action", nodes.risky_action_node)
    builder.add_node("approval", nodes.approval_node)
    builder.add_node("retry", nodes.retry_or_fallback_node)
    builder.add_node("dead_letter", nodes.dead_letter_node)
    builder.add_node("finalize", nodes.finalize_node)

    # 2) Fixed edges.
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "classify")
    builder.add_edge("tool", "evaluate")
    builder.add_edge("risky_action", "approval")
    builder.add_edge("answer", "finalize")
    builder.add_edge("clarify", "finalize")
    builder.add_edge("dead_letter", "finalize")
    builder.add_edge("finalize", END)

    # 3) Conditional edges. The second arg is the router; the third maps the
    #    router's returned string to an actual node name.
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "answer": "answer",
            "tool": "tool",
            "clarify": "clarify",
            "risky_action": "risky_action",
            "retry": "retry",
        },
    )
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"retry": "retry", "answer": "answer"},
    )
    builder.add_conditional_edges(
        "retry",
        route_after_retry,
        {"tool": "tool", "dead_letter": "dead_letter"},
    )
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"tool": "tool", "clarify": "clarify"},
    )

    # 4) Compile (checkpointer enables persistence / thread-scoped state).
    return builder.compile(checkpointer=checkpointer)
