import json
import logging
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from runline_ai.agent import graph
from runline_ai.models import ChatRequest, ChatResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("runline_ai")

app = FastAPI(
    title="Runline AI",
    description="Manufacturing operator copilot — multi-agent system with LangGraph",
    version="0.1.0",
)


def _to_jsonable(obj: Any) -> Any:
    """Help json.dumps serialize Pydantic instances and other rich types."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if hasattr(obj, "to_json"):
        return obj.to_json()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _build_input(request: ChatRequest) -> dict[str, Any]:
    """Construct the graph input from a chat request.

    The user's question is pushed as a HumanMessage; combined with the
    add_messages reducer on AgentState.messages, this appends to any prior
    history loaded by the checkpointer for this thread_id.
    """
    return {
        "question": request.question,
        "messages": [HumanMessage(content=request.question)],
    }


def _build_config(request: ChatRequest) -> dict[str, Any]:
    """Pass thread_id through so the checkpointer can load/save history."""
    if request.thread_id:
        return {"configurable": {"thread_id": request.thread_id}}
    # Without a thread_id, we still need a config object but no persistence.
    # Using a synthetic one-shot id keeps every call isolated.
    return {"configurable": {"thread_id": "_one_shot"}}


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "runline-ai"}


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    logger.info(f"chat request: {request.question[:50]!r} thread={request.thread_id}")
    result = graph.invoke(_build_input(request), config=_build_config(request))

    cls = result["classification"]
    sources = result.get("sources", [])

    return ChatResponse(
        category=cls.category,
        confidence=cls.confidence,
        reasoning=cls.reasoning,
        answer=result["final_answer"],
        used_sources=len(sources),
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """SSE endpoint — emits one event per agent node as it completes."""
    logger.info(f"chat stream request: {request.question[:50]!r} thread={request.thread_id}")

    async def event_stream():
        try:
            async for chunk in graph.astream(
                _build_input(request),
                config=_build_config(request),
                stream_mode="updates",
            ):
                payload = json.dumps(chunk, default=_to_jsonable)
                yield f"data: {payload}\n\n"
        except Exception as e:
            logger.error(f"stream error: {e}", exc_info=True)
            error_payload = json.dumps({"error": str(e)})
            yield f"data: {error_payload}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
