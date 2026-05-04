"""Test the structured-output Pydantic schemas used by each agent.

These don't call the LLM — they just validate the schemas accept correct
inputs and reject invalid ones, so we know the shape contract is honored
when LangChain forces the LLM to conform.
"""

import pytest
from pydantic import ValidationError

from runline_ai.agent import (
    Classification,
    DiagnosisItem,
    DiagnosisList,
    SafetyVerdict,
    SolutionPlan,
    SolutionStep,
)


def test_classification_valid() -> None:
    cls = Classification(
        category="manufacturing",
        confidence=0.92,
        reasoning="Asks about a manufacturing process",
    )
    assert cls.category == "manufacturing"
    assert cls.machine_id is None
    assert cls.error_code is None


def test_classification_with_machine_and_error() -> None:
    cls = Classification(
        category="manufacturing",
        confidence=0.99,
        reasoning="Specific error report",
        machine_id="P-203",
        error_code="E4521",
    )
    assert cls.machine_id == "P-203"
    assert cls.error_code == "E4521"


def test_classification_rejects_invalid_category() -> None:
    with pytest.raises(ValidationError):
        Classification(category="foo", confidence=0.5, reasoning="x")


def test_diagnosis_list_requires_at_least_one_item() -> None:
    with pytest.raises(ValidationError):
        DiagnosisList(items=[])


def test_diagnosis_list_caps_at_five() -> None:
    items = [
        DiagnosisItem(cause=f"cause{i}", likelihood=0.1, reasoning="r")
        for i in range(6)
    ]
    with pytest.raises(ValidationError):
        DiagnosisList(items=items)


def test_diagnosis_item_likelihood_in_range() -> None:
    with pytest.raises(ValidationError):
        DiagnosisItem(cause="x", likelihood=1.5, reasoning="r")


def test_solution_plan_valid() -> None:
    plan = SolutionPlan(
        steps=[SolutionStep(order=1, action="check pressure", safety_critical=True)],
        estimated_minutes=10,
    )
    assert plan.steps[0].safety_critical is True
    assert plan.estimated_minutes == 10


def test_solution_plan_rejects_zero_minutes() -> None:
    with pytest.raises(ValidationError):
        SolutionPlan(
            steps=[SolutionStep(order=1, action="x", safety_critical=False)],
            estimated_minutes=0,
        )


def test_safety_verdict_approved_with_warnings() -> None:
    verdict = SafetyVerdict(
        approved=True,
        warnings=["wear PPE", "verify pressure"],
    )
    assert verdict.approved is True
    assert len(verdict.warnings) == 2
    assert verdict.blocking_issue is None


def test_safety_verdict_blocked() -> None:
    verdict = SafetyVerdict(
        approved=False,
        blocking_issue="LOTO not applied before cylinder access",
    )
    assert verdict.approved is False
    assert verdict.blocking_issue is not None


def test_safety_verdict_default_warnings_is_empty_list() -> None:
    """Mutable default safety — every instance gets its own empty list."""
    a = SafetyVerdict(approved=True)
    b = SafetyVerdict(approved=True)
    a.warnings.append("test")
    assert b.warnings == []  # not shared
