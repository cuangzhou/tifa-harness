from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=3).strip()
    except Exception:
        return None


@dataclass
class WorkspaceContext:
    repo_root: Path
    branch: str | None
    status: str
    documents: dict[str, str]

    @classmethod
    def build(cls, root: str | Path) -> "WorkspaceContext":
        repo = Path(root).resolve()
        if not repo.is_dir():
            raise ValueError(f"workspace is not a directory: {repo}")
        documents: dict[str, str] = {}
        for name in ("README.md", "README.rst", "pyproject.toml", "package.json", "Cargo.toml", "AGENTS.md"):
            path = repo / name
            if path.is_file():
                documents[name] = path.read_text(encoding="utf-8", errors="replace")[:8000]
        return cls(repo, _git(repo, "branch", "--show-current"), _git(repo, "status", "--short") or "", documents)

    def fingerprint(self) -> str:
        payload = {"root": str(self.repo_root), "head": _git(self.repo_root, "rev-parse", "HEAD"), "status": self.status, "documents": self.documents}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def text(self) -> str:
        docs = "\n\n".join(f"## {name}\n{content}" for name, content in self.documents.items())
        return f"Repository: {self.repo_root}\nBranch: {self.branch or 'unknown'}\nGit status:\n{self.status or 'clean/unknown'}\n\n{docs}"

    def resolve(self, value: str, *, must_exist: bool = False) -> Path:
        candidate = (self.repo_root / value).resolve()
        try:
            candidate.relative_to(self.repo_root)
        except ValueError as exc:
            raise ValueError("path escapes workspace") from exc
        if must_exist and not candidate.exists():
            raise ValueError(f"path does not exist: {value}")
        return candidate
