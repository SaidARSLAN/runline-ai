"""Multi-path agent graph — classify → (retrieve+answer | quick | reject).

This is the heart of runline_ai. The graph is compiled once at module import
and exposed as the `graph` symbol.
"""

from typing import Literal, NotRequired, Required, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from paperdex.models import Source
from pydantic import BaseModel, Field
from runline_ai import llm, retriever


class Classification(BaseModel):
    category: Literal["manufacturing", "general", "unsafe"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


_classifier_llm = llm.with_structured_output(Classification)


class AgentState(TypedDict):
    question: Required[str]
    classification: NotRequired[Classification]
    sources: NotRequired[list[Source]]
    answer: NotRequired[str]


def _classify(state: AgentState) -> dict:
    result = _classifier_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a classifier for a manufacturing operator chatbot. "
                    "Classify into one of three categories."
                )
            ),
            HumanMessage(content=state["question"]),
        ]
    )
    return {"classification": result}


def _retrieve(state: AgentState) -> dict:
    sources = retriever.retrieve(state["question"], top_k=3)
    return {"sources": sources}


def _contextual_answer(state: AgentState) -> dict:
    sources = state.get("sources", [])
    context = "\n\n".join(f"[Source {i}] {s.text}" for i, s in enumerate(sources, start=1))
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Answer based on context. If context lacks the answer, say so. Reply briefly."
                )
            ),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}"),
        ]
    )
    return {"answer": str(response.content)}


def _quick_answer(state: AgentState) -> dict:
    response = llm.invoke(
        [
            SystemMessage(content="Give a brief, one-paragraph answer."),
            HumanMessage(content=state["question"]),
        ]
    )
    return {"answer": str(response.content)}


def _reject(state: AgentState) -> dict:
    cls = state.get("classification")
    reason = cls.reasoning if cls else "unsafe content"
    return {"answer": f"Refused: {reason}"}


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
    builder.add_node("contextual_answer", _contextual_answer)
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
    builder.add_edge("retrieve", "contextual_answer")
    builder.add_edge("contextual_answer", END)
    builder.add_edge("quick_answer", END)
    builder.add_edge("reject", END)

    return builder.compile()


graph = _build_graph()
