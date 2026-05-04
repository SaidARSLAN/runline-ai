"""Multi-path agent graph — classify → (retrieve+answer | quick | reject).

This is the heart of runline_ai. The graph is compiled once at module import
and exposed as the `graph` symbol.
"""

from typing import Annotated, Literal, NotRequired, Required, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from paperdex.models import Source
from pydantic import BaseModel, Field

from runline_ai import llm, retriever


class Classification(BaseModel):
    """Output schema for the classifier agent."""

    category: Literal["manufacturing", "general", "unsafe"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    machine_id: str | None = Field(
        default=None,
        description="Machine identifier mentioned in the question, e.g. P-203",
    )
    error_code: str | None = Field(
        default=None,
        description="Error code mentioned in the question, e.g. E4521",
    )


class DiagnosisItem(BaseModel):
    """A single possible root cause with likelihood and reasoning."""

    cause: str = Field(description="Possible root cause, one short phrase")
    likelihood: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated probability that this cause explains the issue",
    )
    reasoning: str = Field(description="Why this cause is plausible, one sentence")


class DiagnosisList(BaseModel):
    """Output schema for the diagnoser agent — ranked list of possible causes."""

    items: list[DiagnosisItem] = Field(
        min_length=1,
        max_length=5,
        description="Causes ordered from most to least likely",
    )


class SolutionStep(BaseModel):
    """A single step in a solution plan."""

    order: int = Field(ge=1, description="1-based step order")
    action: str = Field(description="What the operator should do")
    safety_critical: bool = Field(
        description=(
            "True if this step requires lockout-tagout, PPE, or other "
            "safety acknowledgment before proceeding"
        )
    )


class SolutionPlan(BaseModel):
    """Output schema for the solution planner agent."""

    steps: list[SolutionStep] = Field(min_length=1, max_length=10)
    estimated_minutes: int = Field(
        ge=1,
        description="Rough total time estimate for executing all steps",
    )


class SafetyVerdict(BaseModel):
    """Output schema for the safety verifier agent."""

    approved: bool = Field(description="True if the solution can be safely executed")
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-blocking safety reminders the operator should follow",
    )
    blocking_issue: str | None = Field(
        default=None,
        description=(
            "If approved is False, describes the safety issue that blocks execution; otherwise None"
        ),
    )


_classifier_llm = llm.with_structured_output(Classification)
_diagnoser_llm = llm.with_structured_output(DiagnosisList)
_solver_llm = llm.with_structured_output(SolutionPlan)
_verifier_llm = llm.with_structured_output(SafetyVerdict)


class AgentState(TypedDict):
    question: Required[str]

    # Conversation history — populated across turns when a thread_id is used.
    # The add_messages reducer means new messages APPEND to the list rather
    # than overwriting it. The endpoint pushes a HumanMessage on each turn,
    # and the final-answer node pushes an AIMessage with the response.
    messages: NotRequired[Annotated[list[BaseMessage], add_messages]]

    # Each agent fills its own field
    classification: NotRequired[Classification]
    sources: NotRequired[list[Source]]
    diagnosis: NotRequired[DiagnosisList]
    solution: NotRequired[SolutionPlan]
    safety: NotRequired[SafetyVerdict]

    # Final formatted output that the HTTP endpoint returns
    final_answer: NotRequired[str]


def _classify(state: AgentState) -> dict:
    # Use full conversation history if available (multi-turn), otherwise
    # fall back to the single question (single-turn callers / first turn).
    history = state.get("messages") or [HumanMessage(content=state["question"])]
    result = _classifier_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a classifier for a manufacturing operator chatbot. "
                    "Classify the latest user message, taking earlier messages "
                    "in the conversation into account when interpreting "
                    "follow-up references like 'it', 'that', or 'again'."
                )
            ),
            *history,
        ]
    )
    return {"classification": result}


def _diagnose(state: AgentState) -> dict:
    sources = state.get("sources", [])
    context = "\n\n".join(f"[Source {i}] {s.text}" for i, s in enumerate(sources, start=1))
    cls = state.get("classification")
    machine_info = ""
    if cls and (cls.machine_id or cls.error_code):
        machine_info = (
            f"Machine: {cls.machine_id or 'unknown'}, Error: {cls.error_code or 'unknown'}\n\n"
        )

    result = _diagnoser_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a diagnostic expert for manufacturing equipment. "
                    "Given the question and supporting context, list 1 to 5 "
                    "possible root causes ordered from most to least likely. "
                    "Each cause must include a probability and a one-sentence "
                    "reasoning."
                )
            ),
            HumanMessage(
                content=(
                    f"{machine_info}"
                    f"Question: {state['question']}\n\n"
                    f"Context from documentation:\n{context}\n\n"
                    f"What are the possible causes?"
                )
            ),
        ]
    )
    return {"diagnosis": result}


def _retrieve(state: AgentState) -> dict:
    sources = retriever.retrieve(state["question"], top_k=3)
    return {"sources": sources}


def _solve(state: AgentState) -> dict:
    diagnosis = state.get("diagnosis")
    if diagnosis is None or not diagnosis.items:
        # Should not happen if graph is wired correctly, but defensive guard
        return {
            "solution": SolutionPlan(
                steps=[
                    SolutionStep(
                        order=1,
                        action="Diagnosis missing — escalate to engineer",
                        safety_critical=True,
                    )
                ],
                estimated_minutes=1,
            )
        }

    top_cause = diagnosis.items[0]
    sources = state.get("sources", [])
    context = "\n\n".join(f"[Source {i}] {s.text}" for i, s in enumerate(sources, start=1))

    result = _solver_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a solution planner for manufacturing operators. "
                    "Given the most likely root cause and supporting documentation, "
                    "produce step-by-step instructions. Mark each step as "
                    "safety_critical if it requires lockout-tagout, PPE, or "
                    "any risk acknowledgment before proceeding."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {state['question']}\n\n"
                    f"Most likely cause: {top_cause.cause}\n"
                    f"Reasoning: {top_cause.reasoning}\n\n"
                    f"Supporting context:\n{context}\n\n"
                    f"Provide a step-by-step solution."
                )
            ),
        ]
    )
    return {"solution": result}


def _verify(state: AgentState) -> dict:
    solution = state.get("solution")
    if solution is None:
        return {
            "safety": SafetyVerdict(
                approved=False,
                blocking_issue="Solution missing — cannot verify safety",
            )
        }

    steps_text = "\n".join(
        f"{step.order}. {step.action} (safety_critical={step.safety_critical})"
        for step in solution.steps
    )

    result = _verifier_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a safety verifier for manufacturing procedures. "
                    "Review the proposed solution steps. Approve only if every "
                    "step is safe to execute. If any step puts the operator at "
                    "risk without proper safeguards (LOTO, PPE, zero-energy "
                    "verification), reject and explain the blocking issue. "
                    "Add non-blocking warnings as reminders."
                )
            ),
            HumanMessage(content=f"Steps:\n{steps_text}"),
        ]
    )
    return {"safety": result}


def _format_final(state: AgentState) -> dict:
    """Combine all agent outputs into a single markdown answer."""
    cls = state.get("classification")
    diagnosis = state.get("diagnosis")
    solution = state.get("solution")
    safety = state.get("safety")

    parts: list[str] = []

    if cls:
        parts.append(f"**Category:** {cls.category} (confidence: {cls.confidence:.0%})")

    if diagnosis:
        parts.append("\n## Possible Causes\n")
        for i, item in enumerate(diagnosis.items, start=1):
            parts.append(f"{i}. **{item.cause}** ({item.likelihood:.0%}) — {item.reasoning}")

    if solution:
        parts.append("\n## Solution Steps\n")
        for step in solution.steps:
            marker = "⚠️ " if step.safety_critical else ""
            parts.append(f"{step.order}. {marker}{step.action}")
        parts.append(f"\n_Estimated time: {solution.estimated_minutes} minutes_")

    if safety:
        parts.append("\n## Safety Check\n")
        if safety.approved:
            parts.append("✅ Approved")
            if safety.warnings:
                parts.append("**Warnings:**")
                parts.extend(f"- {w}" for w in safety.warnings)
        else:
            parts.append(f"❌ BLOCKED: {safety.blocking_issue}")

    final = "\n".join(parts)
    return {"final_answer": final, "messages": [AIMessage(content=final)]}


def _blocked(state: AgentState) -> dict:
    """Safety verifier rejected the solution — return a refusal message."""
    safety = state.get("safety")
    issue = (
        safety.blocking_issue if safety and safety.blocking_issue else "safety verification failed"
    )
    final = (
        f"❌ Solution blocked by safety verification.\n\n"
        f"**Reason:** {issue}\n\n"
        f"Please consult an engineer before proceeding."
    )
    return {"final_answer": final, "messages": [AIMessage(content=final)]}


def _safety_route(state: AgentState) -> str:
    """After verify, route to format if safe, blocked otherwise."""
    safety = state.get("safety")
    if safety is not None and safety.approved:
        return "format"
    return "blocked"


def _quick_answer(state: AgentState) -> dict:
    history = state.get("messages") or [HumanMessage(content=state["question"])]
    response = llm.invoke([SystemMessage(content="Give a brief, one-paragraph answer."), *history])
    final = str(response.content)
    return {"final_answer": final, "messages": [AIMessage(content=final)]}


def _reject(state: AgentState) -> dict:
    cls = state.get("classification")
    reason = cls.reasoning if cls else "unsafe content"
    final = f"Refused: {reason}"
    return {"final_answer": final, "messages": [AIMessage(content=final)]}


def _route(state: AgentState) -> str:
    cls = state.get("classification")
    if cls is None:
        return "quick_answer"
    if cls.category == "manufacturing":
        return "retrieve"
    if cls.category == "unsafe":
        return "reject"
    return "quick_answer"


def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("classify", _classify)
    builder.add_node("retrieve", _retrieve)
    builder.add_node("diagnose", _diagnose)
    builder.add_node("solve", _solve)
    builder.add_node("verify", _verify)
    builder.add_node("format", _format_final)
    builder.add_node("blocked", _blocked)
    builder.add_node("quick_answer", _quick_answer)
    builder.add_node("reject", _reject)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        _route,
        {
            "retrieve": "retrieve",
            "quick_answer": "quick_answer",
            "reject": "reject",
        },
    )

    # Manufacturing path — 5-agent chain
    builder.add_edge("retrieve", "diagnose")
    builder.add_edge("diagnose", "solve")
    builder.add_edge("solve", "verify")
    builder.add_conditional_edges(
        "verify",
        _safety_route,
        {"format": "format", "blocked": "blocked"},
    )
    builder.add_edge("format", END)
    builder.add_edge("blocked", END)

    # Other paths
    builder.add_edge("quick_answer", END)
    builder.add_edge("reject", END)

    # MemorySaver persists state per thread_id (in-process, lost on restart).
    # Production deployments would swap this for SqliteSaver or PostgresSaver.
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


graph = _build_graph()
