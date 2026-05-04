"""Test FastAPI endpoints via TestClient.

NOTE: importing runline_ai.main triggers global LLM, retriever, and graph
initialization. The first test run is slow (~10-15s warm-up). For Phase
beyond Demo 2, mock the LLM/retriever so tests run in milliseconds.
"""

from fastapi.testclient import TestClient

from runline_ai.main import app

client = TestClient(app)


def test_root_returns_health_status() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "runline-ai"}


def test_chat_rejects_short_question() -> None:
    """POST /chat with question < 3 chars returns 422 (Pydantic validation)."""
    response = client.post("/chat", json={"question": "ab"})

    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert any("question" in str(err.get("loc", [])) for err in body["detail"])


def test_chat_missing_question_field() -> None:
    """POST /chat without 'question' field returns 422."""
    response = client.post("/chat", json={})

    assert response.status_code == 422


def test_chat_stream_rejects_short_question() -> None:
    """POST /chat/stream also enforces validation."""
    response = client.post("/chat/stream", json={"question": "ab"})

    assert response.status_code == 422


def test_chat_accepts_thread_id() -> None:
    """thread_id is optional but accepted (not full happy-path test)."""
    # We don't actually invoke — just check the request body is accepted by Pydantic
    # validation. A short question still fails at validation, but we care about
    # whether the schema accepts the thread_id field at all.
    response = client.post("/chat", json={"question": "ab", "thread_id": "user-1"})
    # Should fail on question length, not on thread_id
    assert response.status_code == 422
    body = response.json()
    # The error must be about question, not about thread_id
    error_locs = [str(err.get("loc", [])) for err in body["detail"]]
    assert any("question" in loc for loc in error_locs)
    assert not any("thread_id" in loc for loc in error_locs)
