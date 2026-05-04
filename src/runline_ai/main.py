import logging

from fastapi import FastAPI

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
