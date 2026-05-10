import json
import logging
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel

from runline_ai.agent import graph
from runline_ai.models import ChatRequest, ChatResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("runline_ai")

langfuse_handler = CallbackHandler()

app = FastAPI(
    title="Runline AI",
    description="Manufacturing operator copilot — multi-agent system with LangGraph",
    version="0.1.0",
)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if hasattr(obj, "to_json"):
        return obj.to_json()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _build_input(request: ChatRequest) -> dict[str, Any]:
    return {
        "question": request.question,
        "messages": [HumanMessage(content=request.question)],
    }


def _build_config(request: ChatRequest) -> dict[str, Any]:
    thread_id = request.thread_id or "_one_shot"
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": thread_id,
            "langfuse_tags": ["runline-ai", "chat"],
        },
        "run_name": "chat",
    }


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
