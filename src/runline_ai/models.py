from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    thread_id: str | None = Field(
        default=None,
        description=(
            "Conversation thread identifier. Reuse the same thread_id across "
            "turns to keep conversation history. Omit for stateless one-shot "
            "calls."
        ),
    )


class ChatResponse(BaseModel):
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    answer: str
    used_sources: int = Field(
        ge=0,
        description="Number of sources retrieved (0 if not RAG path)",
    )
