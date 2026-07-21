from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, List
import uuid

from .stores import atomic_json


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    schema_version: str = "case-card.v2"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str | None = None
    supersedes: str | None = None
    source_run_id: str | None = None
    repo_snapshot: str | None = None
    dependency_digests: dict[str, str] = field(default_factory=dict)
    tool_schema_digest: str | None = None
    context_policy_version: str | None = None
    suggested_override: dict[str, Any] = field(default_factory=dict)
    review_status: str = "ready"
    freshness_status: str = "unknown"
    stale_reasons: List[str] = field(default_factory=list)
    verification: dict[str, Any] | None = None
    rejection_reason: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CaseCard":
        data = dict(payload)
        if data.get("schema_version", "case-card.v1") == "case-card.v1":
            data["schema_version"] = "case-card.v2"
            data.setdefault("review_status", "legacy_review_required")
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in allowed})


def suggest_override(failure_category: str | None) -> tuple[dict[str, Any], str]:
    if failure_category in {"transport", "timeout", "rate_limit", "auth", "invalid_response"}:
        return {"variable": "provider", "after": "review-required-provider"}, "ready"
    if failure_category in {"context_missing", "irrelevant_context", "stale_context"}:
        return {"variable": "context_policy", "after": "expanded"}, "ready"
    if failure_category in {"loop_detected", "memory_stale", "memory_conflict"}:
        return {"variable": "memory_enabled", "after": False}, "ready"
    return {}, "needs_review"


class CaseStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, card: CaseCard) -> Path:
        if card.verification_status not in {"candidate", "verified", "rejected", "superseded"}:
            raise ValueError("invalid case status")
        if not card.evidence_refs or not card.applicability.get("allowed_categories"):
            raise ValueError("case evidence and applicability are required")
        path = self.root / f"{card.case_id}.json"
        atomic_json(path, asdict(card))
        return path

    def load(self, case_id: str) -> CaseCard:
        payload = json.loads((self.root / f"{case_id}.json").read_text(encoding="utf-8"))
        return CaseCard.from_dict(payload)

    def list(self) -> List[CaseCard]:
        return [self.load(path.stem) for path in sorted(self.root.glob("*.json"))]

    def propose_from_run(self, workspace: Path, run_id: str) -> CaseCard:
        for existing in self.list():
            if existing.source_run_id == run_id and existing.verification_status == "candidate": return existing
        run = workspace / ".tifa" / "runs" / run_id
        report = json.loads((run / "report.json").read_text(encoding="utf-8"))
        bundle = json.loads((run / "evidence_bundle.json").read_text(encoding="utf-8"))
        if report.get("verifier", {}).get("passed") is True:
            raise ValueError("successful runs are not case candidates")
        contract_id = bundle.get("task_contract", {}).get("contract_id")
        derived_category = contract_id.rsplit("-", 1)[0] if isinstance(contract_id, str) and "-" in contract_id else contract_id
        category = bundle.get("task_contract", {}).get("category") or derived_category or report.get("failure_category") or "unknown"
        suggested, review = suggest_override(report.get("failure_category"))
        affected = report.get("affected_paths", [])
        dependencies = {item["path"]: item.get("digest", "") for item in bundle.get("artifacts", []) if item.get("path") in affected}
        card = CaseCard(task_signature={"category": category, "contract_id": bundle.get("task_contract", {}).get("contract_id"), "tool_pattern": [event.get("payload", {}).get("name") for event in bundle.get("events", []) if event.get("type") == "tool_commit"]}, failure_signature={"category": report.get("failure_category"), "stop_reason": report.get("stop_reason")}, minimal_delta=suggested, suggested_override=suggested, evidence_refs=[str(run / "report.json"), str(run / "evidence_bundle.json")], applicability={"allowed_categories": [category], "excluded_conditions": []}, summary=f"Failure {report.get('failure_category') or 'unknown'} in run {run_id}; use only after verified counterfactual replay.", source_run_id=run_id, repo_snapshot=bundle.get("repo_snapshot", {}).get("workspace_digest"), dependency_digests=dependencies, tool_schema_digest=bundle.get("repo_snapshot", {}).get("tool_schema_digest"), context_policy_version=bundle.get("context_manifest", {}).get("policy"), review_status=review, freshness_status="fresh")
        self.save(card)
        return card

    def assess_freshness(self, card: CaseCard, *, repo_snapshot: str | None = None, dependency_digests: dict[str, str] | None = None, tool_schema_digest: str | None = None, context_policy_version: str | None = None) -> tuple[bool, List[str]]:
        reasons = []
        if card.expires_at and datetime.fromisoformat(card.expires_at) <= datetime.now(timezone.utc):
            reasons.append("expired")
        if repo_snapshot and card.repo_snapshot and repo_snapshot != card.repo_snapshot:
            reasons.append("repo_snapshot_changed")
        for path, expected in card.dependency_digests.items():
            if dependency_digests is not None and dependency_digests.get(path) != expected:
                reasons.append(f"dependency_changed:{path}")
        if tool_schema_digest and card.tool_schema_digest and tool_schema_digest != card.tool_schema_digest:
            reasons.append("tool_schema_changed")
        if context_policy_version and card.context_policy_version and context_policy_version != card.context_policy_version:
            reasons.append("context_policy_changed")
        card.stale_reasons = reasons
        card.freshness_status = "stale" if reasons else "fresh"
        return not reasons, reasons

    def record_verification(self, case_id: str, replay: dict[str, Any]) -> CaseCard:
        card = self.load(case_id)
        card.verification = replay
        card.verification_run_id = replay.get("replay_run_id")
        self.save(card)
        return card

    def promote(self, card: CaseCard, replay: dict[str, Any] | None = None) -> CaseCard:
        evidence = replay or card.verification or {}
        report = evidence.get("report", {}); spec = evidence.get("spec", {}); bundle = evidence.get("replay_bundle", {}); overrides = spec.get("overrides", {})
        variable = card.minimal_delta.get("variable")
        if len(overrides) != 1 or variable not in overrides or overrides[variable] != card.minimal_delta.get("after"):
            raise ValueError("case minimal-delta gate failed")
        if report.get("confounded") or bundle.get("verifier", {}).get("passed") is not True or evidence.get("budget_exceeded"):
            raise ValueError("case verification gate failed")
        if not card.failure_signature.get("category") or not evidence.get("same_task_contract") or not evidence.get("same_snapshot") or not report.get("source_unchanged"):
            raise ValueError("case provenance gate failed")
        if card.freshness_status == "stale":
            raise ValueError("stale case cannot be promoted")
        for existing in self.list():
            if existing.case_id != card.case_id and existing.verification_status == "verified" and existing.task_signature == card.task_signature and existing.minimal_delta == card.minimal_delta:
                raise ValueError(f"duplicate verified case: {existing.case_id}")
            if existing.case_id != card.case_id and existing.verification_status == "verified" and existing.task_signature == card.task_signature and existing.minimal_delta != card.minimal_delta and not card.supersedes:
                raise ValueError(f"conflicting verified case requires supersedes: {existing.case_id}")
        if card.supersedes:
            previous = self.load(card.supersedes)
            if previous.verification_status != "verified":
                raise ValueError("only verified cases can be superseded")
            previous.verification_status = "superseded"; self.save(previous)
        card.verification_status = "verified"; card.verification_run_id = evidence.get("replay_run_id"); self.save(card); return card

    def reject(self, case_id: str, reason: str | None = None) -> CaseCard:
        card = self.load(case_id); card.verification_status = "rejected"; card.rejection_reason = reason; self.save(card); return card

    def search(self, category: str, tool_pattern: List[str] | None = None, top_k: int = 3, code_version: str | None = None, **freshness: Any) -> List[CaseCard]:
        result = []
        for card in self.list():
            allowed = card.applicability.get("allowed_categories", []); excluded = card.applicability.get("excluded_conditions", [])
            fresh, _ = self.assess_freshness(card, **freshness)
            if card.verification_status != "verified" or not fresh or category not in allowed or (code_version and f"code_version:{code_version}" in excluded):
                continue
            if tool_pattern and not set(tool_pattern).intersection(card.task_signature.get("tool_pattern", [])):
                continue
            result.append(card)
        return result[:top_k]


def tool_schema_digest(schemas: list[dict[str, Any]]) -> str:
    return _digest(schemas)
