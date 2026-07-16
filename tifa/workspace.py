from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_EXCLUDES = {".git", ".tifa", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build"}


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=3).strip()
    except Exception:
        return None


def _patterns(root: Path) -> list[str]:
    patterns: list[str] = []
    for name in (".gitignore", ".tifaignore"):
        path = root / name
        if path.is_file():
            patterns.extend(line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip() and not line.lstrip().startswith("#") and not line.startswith("!"))
    return patterns


def _ignored(relative: str, patterns: list[str]) -> bool:
    parts = relative.split("/")
    if any(part in DEFAULT_EXCLUDES for part in parts): return True
    return any(fnmatch.fnmatch(relative, p.rstrip("/")) or fnmatch.fnmatch(Path(relative).name, p.rstrip("/")) or relative.startswith(p.rstrip("/") + "/") for p in patterns)


def _files(root: Path, patterns: list[str]):
    pending = [root]
    while pending:
        folder = pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                path = Path(entry.path); relative = path.relative_to(root).as_posix()
                if _ignored(relative, patterns): continue
                if entry.is_dir(follow_symlinks=False): pending.append(path)
                elif entry.is_file(follow_symlinks=False): yield path, entry.stat(follow_symlinks=False)


@dataclass
class WorkspaceContext:
    repo_root: Path
    branch: str | None
    status: str
    documents: dict[str, str]
    index: dict[str, dict[str, Any]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, root: str | Path) -> "WorkspaceContext":
        repo = Path(root).resolve()
        if not repo.is_dir(): raise ValueError(f"workspace is not a directory: {repo}")
        documents: dict[str, str] = {}
        for name in ("README.md", "README.rst", "pyproject.toml", "package.json", "Cargo.toml", "AGENTS.md"):
            path = repo / name
            if path.is_file(): documents[name] = path.read_text(encoding="utf-8", errors="replace")[:8000]
        cache_path = repo / ".tifa" / "cache" / "workspace-index.json"
        try: previous = json.loads(cache_path.read_text(encoding="utf-8")).get("files", {})
        except (OSError, ValueError, TypeError): previous = {}
        patterns = _patterns(repo); index: dict[str, dict[str, Any]] = {}; changed = 0; extensions: dict[str, int] = {}
        for path, stat in _files(repo, patterns):
            rel = path.relative_to(repo).as_posix()
            if _ignored(rel, patterns): continue
            old = previous.get(rel, {})
            item: dict[str, Any] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if old.get("size") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns and old.get("digest"):
                item["digest"] = old["digest"]
            else:
                item["digest"] = hashlib.sha256(path.read_bytes()).hexdigest(); changed += 1
            index[rel] = item; ext = path.suffix.lower() or "[none]"; extensions[ext] = extensions.get(ext, 0) + 1
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = cache_path.with_suffix(".tmp"); temp.write_text(json.dumps({"schema_version": "tifa-workspace-index.v1", "files": index}, sort_keys=True), encoding="utf-8"); temp.replace(cache_path)
        return cls(repo, _git(repo, "branch", "--show-current"), _git(repo, "status", "--short") or "", documents, index, {"file_count": len(index), "changed_files": changed, "extensions": extensions})

    def fingerprint(self) -> str:
        payload = {"files": {path: item["digest"] for path, item in self.index.items()}, "head": _git(self.repo_root, "rev-parse", "HEAD")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def text(self) -> str:
        docs = "\n\n".join(f"## {name}\n{content}" for name, content in self.documents.items())
        return f"Repository: {self.repo_root}\nBranch: {self.branch or 'unknown'}\nGit status:\n{self.status or 'clean/unknown'}\nIndexed files: {self.stats.get('file_count', 0)}\n\n{docs}"

    def resolve(self, value: str, *, must_exist: bool = False) -> Path:
        candidate = (self.repo_root / value).resolve()
        try: candidate.relative_to(self.repo_root)
        except ValueError as exc: raise ValueError("path escapes workspace") from exc
        if must_exist and not candidate.exists(): raise ValueError(f"path does not exist: {value}")
        return candidate
