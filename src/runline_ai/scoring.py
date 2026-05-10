from typing import Any


def score_trace(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute deterministic scores from final agent state.

    Returns score dicts ready to splat into langfuse.create_score(trace_id=..., **score).
    Pure function: no I/O, no Langfuse calls. Easy to unit test.
    """
    scores: list[dict[str, Any]] = []

    classification = state.get("classification")
    if classification is not None:
        confident = classification.confidence >= 0.85
        scores.append(
            {
                "name": "classification_confident",
                "value": 1.0 if confident else 0.0,
                "data_type": "BOOLEAN",
                "comment": f"confidence={classification.confidence:.2f}",
            }
        )

    solution = state.get("solution")
    if solution is not None and solution.steps:
        n_safety = sum(1 for s in solution.steps if s.safety_critical)
        has_safety = n_safety > 0
        scores.append(
            {
                "name": "solution_has_safety_step",
                "value": 1.0 if has_safety else 0.0,
                "data_type": "BOOLEAN",
                "comment": f"{n_safety}/{len(solution.steps)} steps marked safety_critical",
            }
        )

    return scores
