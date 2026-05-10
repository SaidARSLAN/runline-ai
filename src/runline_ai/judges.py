import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langfuse import get_client
from pydantic import BaseModel, Field

logger = logging.getLogger("runline_ai.judges")

_langfuse = get_client()
_judge_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)


class RelevanceVerdict(BaseModel):
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "1.0 if the answer fully and directly addresses the question; "
            "0.5 if it is partially relevant or only tangentially related; "
            "0.0 if it is completely off-topic, generic, or refuses without justification."
        ),
    )
    reasoning: str = Field(
        description="One short sentence justifying the score, citing what was missing or extra."
    )


_relevance_judge_llm = _judge_llm.with_structured_output(RelevanceVerdict)

_RELEVANCE_SYSTEM = """You are a strict evaluator for a manufacturing operator chatbot.

Your only job is to score how well an answer addresses the user's question.

Do NOT evaluate whether the answer is technically correct, whether the safety
reasoning is right, or whether the steps are in the right order. Other
evaluators handle those dimensions.

Focus only on relevance: does the answer respond to what the operator asked?

A safety-blocked refusal CAN be relevant if it explains why the request cannot
be fulfilled and ties back to the original question."""


def judge_answer_relevance(question: str, answer: str, trace_id: str) -> None:
    """Score how well the answer addresses the question. Writes to trace_id.

    Designed to run as a FastAPI background task so the user does not wait
    for this extra LLM call.
    """
    try:
        verdict = _relevance_judge_llm.invoke(
            [
                SystemMessage(content=_RELEVANCE_SYSTEM),
                HumanMessage(content=f"Question:\n{question}\n\nAnswer:\n{answer}"),
            ]
        )
        _langfuse.create_score(
            trace_id=trace_id,
            name="answer_relevance",
            value=verdict.score,
            data_type="NUMERIC",
            comment=verdict.reasoning,
        )
        _langfuse.flush()
        logger.info(f"judge answer_relevance: score={verdict.score:.2f} trace={trace_id}")
    except Exception as e:
        logger.error(f"judge failed for trace {trace_id}: {e}", exc_info=True)
