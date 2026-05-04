"""Test ChatRequest and ChatResponse Pydantic models."""

import pytest
from pydantic import ValidationError

from runline_ai.models import ChatRequest, ChatResponse


def test_valid_chat_request() -> None:
    """A normal question with thread_id passes."""
    req = ChatRequest(question="What is gas welding?", thread_id="user-1")
    assert req.question == "What is gas welding?"
    assert req.thread_id == "user-1"


def test_chat_request_thread_id_optional() -> None:
    """thread_id defaults to None for stateless one-shot calls."""
    req = ChatRequest(question="What is gas welding?")
    assert req.thread_id is None


def test_chat_request_rejects_short_question() -> None:
    """Pydantic enforces min_length=3."""
    with pytest.raises(ValidationError):
        ChatRequest(question="ab")


def test_chat_request_rejects_long_question() -> None:
    """Pydantic enforces max_length=500."""
    with pytest.raises(ValidationError):
        ChatRequest(question="x" * 501)


def test_chat_response_valid() -> None:
    """A complete response with all required fields."""
    resp = ChatResponse(
        category="manufacturing",
        confidence=0.95,
        reasoning="The query is about a manufacturing process",
        answer="Gas welding is...",
        used_sources=3,
    )
    assert resp.category == "manufacturing"
    assert resp.used_sources == 3


def test_chat_response_rejects_negative_sources() -> None:
    """used_sources must be >= 0."""
    with pytest.raises(ValidationError):
        ChatResponse(
            category="manufacturing",
            confidence=0.5,
            reasoning="x",
            answer="y",
            used_sources=-1,
        )


def test_chat_response_rejects_invalid_confidence() -> None:
    """confidence must be between 0 and 1."""
    with pytest.raises(ValidationError):
        ChatResponse(
            category="manufacturing",
            confidence=1.5,
            reasoning="x",
            answer="y",
            used_sources=0,
        )
