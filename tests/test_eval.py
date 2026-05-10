"""Slow integration eval — runs the full agent against runline-eval-v1 cases.

Skipped by default (see pyproject.toml addopts). To run:

    uv run pytest -m eval

Hits Groq + ChromaDB. ~30s, costs tokens. Designed for nightly CI or
pre-merge labels, not every push.
"""

import pytest
from langchain_core.messages import HumanMessage

from runline_ai.agent import graph
from runline_ai.eval_cases import CASES

# Pass thresholds — calibrated to current baseline (Stage 9 run).
# Tighten over time; loosen only with a justification in the PR.
CATEGORY_THRESHOLD = 0.85
MACHINE_ID_THRESHOLD = 0.80
BLOCK_DECISION_THRESHOLD = 0.80


def _run_agent(question: str) -> dict:
    result = graph.invoke(
        {"question": question, "messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": "_pytest_eval"}},
    )
    cls = result["classification"]
    safety = result.get("safety")
    return {
        "category": cls.category,
        "machine_id": cls.machine_id,
        "blocked": safety is not None and not safety.approved,
    }


@pytest.mark.eval
def test_eval_dataset_thresholds() -> None:
    cat_correct = 0
    mid_checks = mid_correct = 0
    blk_checks = blk_correct = 0
    failures: list[str] = []

    for case in CASES:
        case_type = case.metadata["type"]
        try:
            output = _run_agent(case.input["question"])
        except Exception as e:
            failures.append(f"{case_type}: agent crashed — {e}")
            continue

        expected = case.expected_output

        if expected.get("category") == output["category"]:
            cat_correct += 1
        else:
            failures.append(
                f"{case_type}: category expected={expected.get('category')!r} "
                f"got={output['category']!r}"
            )

        if "machine_id" in expected:
            mid_checks += 1
            if expected["machine_id"] == output["machine_id"]:
                mid_correct += 1
            else:
                failures.append(
                    f"{case_type}: machine_id expected={expected['machine_id']!r} "
                    f"got={output['machine_id']!r}"
                )

        if "should_block" in expected:
            blk_checks += 1
            if expected["should_block"] == output["blocked"]:
                blk_correct += 1
            else:
                failures.append(
                    f"{case_type}: should_block expected={expected['should_block']} "
                    f"got={output['blocked']}"
                )

    n = len(CASES)
    cat_pct = cat_correct / n
    mid_pct = mid_correct / mid_checks if mid_checks else 1.0
    blk_pct = blk_correct / blk_checks if blk_checks else 1.0

    print(
        f"\nEval results across {n} cases:"
        f"\n  category_correct: {cat_correct}/{n} = {cat_pct:.0%} (threshold {CATEGORY_THRESHOLD:.0%})"
        f"\n  machine_id_correct: {mid_correct}/{mid_checks} = {mid_pct:.0%} (threshold {MACHINE_ID_THRESHOLD:.0%})"
        f"\n  block_decision_correct: {blk_correct}/{blk_checks} = {blk_pct:.0%} (threshold {BLOCK_DECISION_THRESHOLD:.0%})"
    )
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")

    assert cat_pct >= CATEGORY_THRESHOLD, (
        f"category_correct {cat_pct:.0%} < threshold {CATEGORY_THRESHOLD:.0%}"
    )
    assert mid_pct >= MACHINE_ID_THRESHOLD, (
        f"machine_id_correct {mid_pct:.0%} < threshold {MACHINE_ID_THRESHOLD:.0%}"
    )
    assert blk_pct >= BLOCK_DECISION_THRESHOLD, (
        f"block_decision_correct {blk_pct:.0%} < threshold {BLOCK_DECISION_THRESHOLD:.0%}"
    )
