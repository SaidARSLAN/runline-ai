from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class ChatResponse(BaseModel):
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    answer: str
    used_sources: int = Field(
        ge=0,
        description="Number of sources retrieved (0 if not RAG path)",
    )
