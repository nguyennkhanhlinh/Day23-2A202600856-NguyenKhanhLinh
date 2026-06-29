# Day 08 Lab Report

## 1. Team / student

- Name: Nguyen Khanh Linh
- Repo/commit: local working tree
- Date: generated from latest scenario metrics

## 2. Metrics summary

| Metric | Value |
|---|---:|
| Total scenarios | 7 |
| Success rate | 100.00% |
| Average nodes visited | 6.43 |
| Total retries | 3 |
| Total interrupts / approvals | 2 |
| Resume success | no |

## 3. Architecture

The workflow is a LangGraph `StateGraph` for support-ticket orchestration. Every run starts at
`intake`, then `classify` chooses one of five routes: `simple`, `tool`, `missing_info`, `risky`,
or `error`. Simple requests go directly to `answer`; tool requests call `tool`, then `evaluate`,
then either answer or retry. Risky requests go through `risky_action` and `approval` before any
side-effecting tool step. Missing information goes to `clarify`. All terminal paths pass through
`finalize` before `END`, which gives the run a consistent audit event.

The retry loop is bounded by `attempt` and `max_attempts`. When a tool result is unsatisfactory,
`retry` increments the counter and routes back to `tool` until the limit is reached, then routes to
`dead_letter` for escalation.

## 4. State schema

| Field | Reducer | Why |
|---|---|---|
| `thread_id` | overwrite | Stable id for checkpointer and state history per scenario. |
| `scenario_id` | overwrite | Connects final state back to metrics. |
| `query` | overwrite | Original user request. |
| `route` | overwrite | Latest classified route. |
| `risk_level` | overwrite | Marks risky requests for audit. |
| `attempt` | overwrite | Current retry count. |
| `max_attempts` | overwrite | Scenario-specific retry bound. |
| `evaluation_result` | overwrite | Retry gate result: `success` or `needs_retry`. |
| `pending_question` | overwrite | Clarification question for vague requests. |
| `proposed_action` | overwrite | Action prepared for human approval. |
| `approval` | overwrite | Human/mock approval decision. |
| `final_answer` | overwrite | Final user-facing response. |
| `messages` | append | Lightweight trace of node decisions. |
| `tool_results` | append | Preserves each tool attempt result. |
| `errors` | append | Records transient failures and retry causes. |
| `events` | append | Structured audit trail used by metrics. |

## 5. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | yes | 0 | 0 |
| S02_tool | tool | tool | yes | 0 | 0 |
| S03_missing | missing_info | missing_info | yes | 0 | 0 |
| S04_risky | risky | risky | yes | 0 | 1 |
| S05_error | error | error | yes | 2 | 0 |
| S06_delete | risky | risky | yes | 0 | 1 |
| S07_dead_letter | error | error | yes | 1 | 0 |

## 6. Failure analysis

1. Retry or tool failure: transient tool failures are detected by `evaluate`. The graph retries only
   while `attempt < max_attempts`, preventing infinite loops. If the limit is reached,
   the request is
   sent to `dead_letter` with a clear escalation answer.
2. Risky action without approval: refund, delete, cancellation, email, and similar side-effecting
   requests are routed to `risky_action` and `approval`. The tool path is only entered after an
   approved decision; rejected decisions go to clarification instead.

Latest failed scenario details:

- All sample scenarios succeeded in the latest metrics run.

## 7. Persistence / recovery evidence

The CLI builds the graph with `build_checkpointer()` and invokes each scenario using a configurable
`thread_id`, so memory checkpointing can track each run independently. SQLite/Postgres
persistence is
kept behind the same adapter and can be enabled by changing the checkpointer configuration after the
extension dependency is installed.

## 8. Extension work

Implemented core graph orchestration, structured LLM classification, grounded LLM answering, mock
HITL approval, bounded retry handling, dead-letter escalation, metrics generation, and
markdown report
rendering.

## 9. Improvement plan

With one more day, the first production improvements would be persistent SQLite checkpointing,
LLM-as-judge evaluation for tool outputs, richer approval UI, and replaying state history
for demo and
debugging.
