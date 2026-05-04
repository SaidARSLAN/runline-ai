import json
import logging
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "runline-ai"}


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    logger.info(f"chat request: {request.question[:50]!r}")
    result = graph.invoke({"question": request.question})

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
    logger.info(f"chat stream request: {request.question[:50]!r}")

    async def event_stream():
        try:
            async for chunk in graph.astream(
                {"question": request.question},
                stream_mode="updates",
            ):
                # chunk: {"node_name": {"field": value, ...}}
                payload = json.dumps(chunk, default=_to_jsonable)
                yield f"data: {payload}\n\n"
        except Exception as e:
            logger.error(f"stream error: {e}", exc_info=True)
            error_payload = json.dumps({"error": str(e)})
            yield f"data: {error_payload}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
