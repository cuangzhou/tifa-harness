from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskPhase(str, Enum):
    UNDERSTAND = "UNDERSTAND"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    REPAIR = "REPAIR"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass
class TaskContract:
    goal: str
    allowed_tools: list[str] = field(default_factory=list)
    writable_paths: list[str] = field(default_factory=list)
    verifier: dict[str, Any] | None = None
    max_steps: int | None = None
    max_repairs: int = 2
    require_verifier: bool = True
    contract_id: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskContract":
        known = {"goal", "allowed_tools", "writable_paths", "verifier", "max_steps", "max_repairs", "require_verifier", "contract_id"}
        unknown = set(payload) - known
        if unknown: raise ValueError(f"unknown task contract fields: {sorted(unknown)}")
        contract = cls(**payload)
        if not contract.goal.strip(): raise ValueError("task contract goal is required")
        if contract.max_repairs < 0: raise ValueError("max_repairs must be non-negative")
        return contract


@dataclass
class PlanStep:
    step_id: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    expected_artifact: str | None = None


@dataclass
class ExecutionPlan:
    goal: str
    steps: list[PlanStep]
    completion_conditions: list[str] = field(default_factory=list)


@dataclass
class CompletionDecision:
    complete: bool
    reason: str
    verifier_status: str


@dataclass
class RepairFeedback:
    attempt: int
    failure_category: str
    summary: str
    failed_checks: list[dict[str, Any]] = field(default_factory=list)


def default_plan(contract: TaskContract) -> ExecutionPlan:
    return ExecutionPlan(contract.goal, [PlanStep("execute", contract.goal)], ["task contract verifier passes"])
