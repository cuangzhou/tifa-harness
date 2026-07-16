from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CaseAssistancePolicy:
    default_enabled: bool = False
    minimum_pass_rate_delta: float = 0.02
    require_no_regression: bool = True


def evaluate_case_assistance(baseline: list[dict[str, Any]], assisted: list[dict[str, Any]], policy: CaseAssistancePolicy | None = None) -> dict[str, Any]:
    selected = policy or CaseAssistancePolicy()
    if not baseline or len(baseline) != len(assisted): raise ValueError("paired non-empty baseline and assisted results are required")
    baseline_rate = sum(item.get("passed") is True for item in baseline) / len(baseline); assisted_rate = sum(item.get("passed") is True for item in assisted) / len(assisted); delta = assisted_rate - baseline_rate
    regressions = sum(base.get("passed") is True and assist.get("passed") is not True for base, assist in zip(baseline, assisted))
    enable = delta >= selected.minimum_pass_rate_delta and (not selected.require_no_regression or regressions == 0)
    return {"baseline_pass_rate": baseline_rate, "assisted_pass_rate": assisted_rate, "delta": delta, "regressions": regressions, "default_enabled": enable, "decision": "enable" if enable else "remain_opt_in"}
