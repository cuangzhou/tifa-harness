from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any
import builtins
import uuid
from datetime import datetime, timezone

from .stores import atomic_json


@dataclass
class CaseCard:
    task_signature: dict[str, Any]
    failure_signature: dict[str, Any]
    minimal_delta: dict[str, Any]
    evidence_refs: list[str]
    applicability: dict[str, Any]
    summary: str = ""
    case_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    verification_status: str = "candidate"
    verification_run_id: str | None = None
    schema_version: str = "case-card.v1"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str | None = None
    supersedes: str | None = None


class CaseStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, card: CaseCard) -> Path:
        if card.verification_status not in {"candidate", "verified", "rejected", "superseded"}: raise ValueError("invalid case status")
        if not card.evidence_refs or not card.applicability.get("allowed_categories"): raise ValueError("case evidence and applicability are required")
        path = self.root / f"{card.case_id}.json"; atomic_json(path, asdict(card)); return path

    def load(self, case_id: str) -> CaseCard:
        payload = json.loads((self.root / f"{case_id}.json").read_text(encoding="utf-8")); return CaseCard(**payload)

    def list(self) -> builtins.list[CaseCard]:
        return [self.load(p.stem) for p in sorted(self.root.glob("*.json"))]

    def promote(self, card: CaseCard, replay: dict[str, Any]) -> CaseCard:
        report = replay.get("report", {}); spec = replay.get("spec", {}); bundle = replay.get("replay_bundle", {}); overrides = spec.get("overrides", {})
        variable = card.minimal_delta.get("variable")
        if len(overrides) != 1 or variable not in overrides or overrides[variable] != card.minimal_delta.get("after"): raise ValueError("case minimal-delta gate failed")
        if report.get("confounded") or bundle.get("verifier", {}).get("passed") is not True or replay.get("budget_exceeded"): raise ValueError("case verification gate failed")
        if not card.failure_signature.get("category") or not replay.get("same_task_contract") or not replay.get("same_snapshot") or not report.get("source_unchanged"): raise ValueError("case provenance gate failed")
        for existing in self.list():
            if existing.case_id != card.case_id and existing.verification_status == "verified" and existing.task_signature == card.task_signature and existing.minimal_delta == card.minimal_delta: raise ValueError(f"duplicate verified case: {existing.case_id}")
            if existing.case_id != card.case_id and existing.verification_status == "verified" and existing.task_signature == card.task_signature and existing.minimal_delta != card.minimal_delta and not card.supersedes: raise ValueError(f"conflicting verified case requires supersedes: {existing.case_id}")
        if card.supersedes:
            previous = self.load(card.supersedes)
            if previous.verification_status != "verified": raise ValueError("only verified cases can be superseded")
            previous.verification_status = "superseded"; self.save(previous)
        card.verification_status = "verified"; card.verification_run_id = replay.get("replay_run_id"); self.save(card); return card

    def reject(self, case_id: str) -> CaseCard:
        card = self.load(case_id); card.verification_status = "rejected"; self.save(card); return card

    def search(self, category: str, tool_pattern: builtins.list[str] | None = None, top_k: int = 3, code_version: str | None = None) -> builtins.list[CaseCard]:
        result = []
        for card in self.list():
            if card.expires_at and datetime.fromisoformat(card.expires_at) <= datetime.now(timezone.utc): continue
            allowed = card.applicability.get("allowed_categories", [])
            excluded = card.applicability.get("excluded_conditions", [])
            if card.verification_status != "verified" or category not in allowed or (code_version and f"code_version:{code_version}" in excluded): continue
            if tool_pattern and not set(tool_pattern).intersection(card.task_signature.get("tool_pattern", [])): continue
            result.append(card)
        return result[:top_k]
