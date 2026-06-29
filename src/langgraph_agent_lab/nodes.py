"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, Route, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Structured-output schema for LLM classification ─────────────────
class Classification(BaseModel):
    """What we force the LLM to emit so classification is reliable, not free text."""

    route: Route = Field(description="One of: risky, tool, missing_info, error, simple")
    reason: str = Field(default="", description="Short justification for the chosen route")


CLASSIFY_SYSTEM_PROMPT = """You are an intent classifier for a customer-support agent.
Classify the user's ticket into EXACTLY ONE route. Apply this priority order strictly
(if multiple could match, pick the highest in this list):

1. risky        — actions with side effects: refunds, deletions, cancellations,
                  sending emails, account changes. Anything that mutates data or
                  contacts the customer.
2. tool         — pure information lookups: order status, tracking, search.
3. missing_info — vague/incomplete requests lacking the context needed to act
                  (e.g. "fix it", "can you help?") with no concrete subject.
4. error        — system/technical failures: timeouts, crashes, service
                  unavailable, "cannot recover".
5. simple       — general questions answerable directly without tools or actions
                  (e.g. "how do I reset my password?").

Return the route and a one-line reason."""


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output."""
    query = state.get("query", "")
    llm = get_llm()
    classifier = llm.with_structured_output(Classification)
    result: Classification = classifier.invoke(
        [
            ("system", CLASSIFY_SYSTEM_PROMPT),
            ("human", query),
        ]
    )
    route = result.route.value
    risk_level = "high" if route == Route.RISKY.value else "low"
    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classify:{route}"],
        "events": [
            make_event("classify", "completed", f"route={route}", reason=result.reason)
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call, simulating transient failures for error scenarios.

    For the error route we fail on the first two attempts so the retry loop is
    exercised, then succeed (unless max_attempts is reached first → dead letter).
    """
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    query = state.get("query", "")

    if route == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient tool failure on attempt {attempt} for: {query[:40]}"
        event = make_event("tool", "failed", result, attempt=attempt)
    else:
        result = f"TOOL_OK: retrieved data for '{query[:40]}' (attempt {attempt})"
        event = make_event("tool", "completed", "tool call succeeded", attempt=attempt)

    return {
        "tool_results": [result],
        "messages": [f"tool:attempt={attempt}"],
        "events": [event],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate the latest tool result — the retry-loop gate.

    Heuristic: a result containing "ERROR" needs a retry; anything else is a success.
    (Acceptable for base score; an LLM-as-judge here would earn bonus points.)
    """
    tool_results = state.get("tool_results", []) or []
    latest = tool_results[-1] if tool_results else ""
    needs_retry = "ERROR" in latest.upper()
    evaluation_result = "needs_retry" if needs_retry else "success"
    return {
        "evaluation_result": evaluation_result,
        "messages": [f"evaluate:{evaluation_result}"],
        "events": [
            make_event("evaluate", "completed", f"evaluation={evaluation_result}")
        ],
    }


ANSWER_SYSTEM_PROMPT = """You are a helpful customer-support agent. Write a concise,
friendly final answer to the user's ticket. Ground your answer ONLY in the context
provided (tool results and approval decision). Do not invent order numbers, refunds,
or facts that are not in the context. If a risky action was approved, confirm it was
carried out. Keep it to a few sentences."""


def answer_node(state: AgentState) -> dict:
    """Generate a final grounded response using an LLM."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", []) or []
    approval = state.get("approval") or {}

    context_parts = [f"User ticket: {query}"]
    if tool_results:
        context_parts.append("Tool results:\n" + "\n".join(tool_results))
    if approval:
        context_parts.append(
            f"Approval decision: approved={approval.get('approved')} "
            f"by {approval.get('reviewer')} — {approval.get('comment')}"
        )
    context = "\n\n".join(context_parts)

    llm = get_llm()
    response = llm.invoke(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            ("human", context),
        ]
    )
    answer = response.content if hasattr(response, "content") else str(response)
    return {
        "final_answer": answer,
        "messages": ["answer:generated"],
        "events": [make_event("answer", "completed", "final answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Uses the LLM to generate a specific clarification question for the vague query.
    """
    query = state.get("query", "")
    llm = get_llm()
    response = llm.invoke(
        [
            (
                "system",
                "The user's support request is too vague to act on. Ask ONE concise, "
                "specific clarifying question to get the information you need. "
                "Return only the question.",
            ),
            ("human", query),
        ]
    )
    question = response.content if hasattr(response, "content") else str(response)
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": ["clarify:asked"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    proposed = (
        f"Proposed action requires approval (side effects detected): {query[:120]}"
    )
    return {
        "proposed_action": proposed,
        "messages": ["risky:proposed"],
        "events": [make_event("risky_action", "completed", proposed)],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default: mock approval (approved=True) so tests/CI run offline.
    Extension: if LANGGRAPH_INTERRUPT=true, pause for a real human decision via interrupt().
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        decision = interrupt(
            {
                "proposed_action": state.get("proposed_action", ""),
                "query": state.get("query", ""),
            }
        )
        approval = {
            "approved": bool(decision.get("approved", False)),
            "reviewer": decision.get("reviewer", "human"),
            "comment": decision.get("comment", ""),
        }
    else:
        approval = {
            "approved": True,
            "reviewer": "mock-reviewer",
            "comment": "auto-approved (mock mode)",
        }

    return {
        "approval": approval,
        "messages": [f"approval:approved={approval['approved']}"],
        "events": [
            make_event(
                "approval",
                "completed",
                f"approval decision approved={approval['approved']}",
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt: increment the attempt counter and log the failure."""
    attempt = state.get("attempt", 0) + 1
    return {
        "attempt": attempt,
        "errors": [f"transient failure, retry attempt {attempt}"],
        "messages": [f"retry:attempt={attempt}"],
        "events": [make_event("retry", "completed", f"retry attempt {attempt}")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded (retry → fallback → dead letter)."""
    attempt = state.get("attempt", 0)
    answer = (
        "We were unable to complete your request after multiple attempts. "
        "It has been escalated to our support team for manual follow-up."
    )
    return {
        # Note: we keep `route` as classified (e.g. "error") so metrics can compare
        # actual_route vs expected_route. Dead-letter is terminal handling, not a route.
        "final_answer": answer,
        "messages": ["dead_letter:escalated"],
        "events": [
            make_event(
                "dead_letter", "completed", "max retries exceeded", attempts=attempt
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "messages": ["finalize:done"],
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
