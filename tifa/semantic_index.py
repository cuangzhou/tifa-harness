from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

from .stores import atomic_json
from .workspace import WorkspaceContext


@dataclass
class Symbol:
    name: str
    kind: str
    path: str
    line: int
    language: str


class SemanticIndex:
    schema_version = "tifa-semantic-index.v1"

    def __init__(self, workspace: WorkspaceContext) -> None:
        self.workspace = workspace; self.path = workspace.repo_root / ".tifa" / "index" / "semantic.json"
        self.symbols: list[Symbol] = []; self.references: dict[str, list[dict[str, Any]]] = {}; self.dependencies: dict[str, list[str]] = {}; self.file_digests: dict[str, str] = {}

    def _python(self, relative: str, content: str) -> None:
        try: tree = ast.parse(content)
        except SyntaxError: return
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): self.symbols.append(Symbol(node.name, "function", relative, node.lineno, "python"))
            elif isinstance(node, ast.ClassDef): self.symbols.append(Symbol(node.name, "class", relative, node.lineno, "python"))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]; self.dependencies.setdefault(relative, []).extend(names)
            elif isinstance(node, ast.Name): self.references.setdefault(node.id, []).append({"path": relative, "line": node.lineno})

    def _javascript(self, relative: str, content: str) -> None:
        for match in re.finditer(r"\b(?:export\s+)?(?:async\s+)?(function|class|const|let|var)\s+([A-Za-z_$][\w$]*)", content):
            self.symbols.append(Symbol(match.group(2), match.group(1), relative, content.count("\n", 0, match.start()) + 1, "typescript" if relative.endswith((".ts", ".tsx")) else "javascript"))
        imports = re.findall(r"\b(?:from\s+|require\s*\(\s*)['\"]([^'\"]+)['\"]", content); self.dependencies.setdefault(relative, []).extend(imports)
        for name in re.findall(r"\b[A-Za-z_$][\w$]*\b", content): self.references.setdefault(name, []).append({"path": relative})

    def build(self) -> dict[str, Any]:
        previous = self.load(optional=True); previous_digests = previous.get("file_digests", {}) if previous else {}
        self.symbols, self.references, self.dependencies = [], {}, {}
        changed = 0
        for relative, metadata in self.workspace.index.items():
            if not relative.endswith((".py", ".js", ".jsx", ".ts", ".tsx")): continue
            digest = str(metadata["digest"]); self.file_digests[relative] = digest
            if previous_digests.get(relative) != digest: changed += 1
            content = (self.workspace.repo_root / relative).read_text(encoding="utf-8", errors="replace")
            if relative.endswith(".py"): self._python(relative, content)
            else: self._javascript(relative, content)
        payload = {"schema_version": self.schema_version, "workspace_fingerprint": self.workspace.fingerprint(), "file_digests": self.file_digests, "symbols": [asdict(symbol) for symbol in self.symbols], "references": self.references, "dependencies": self.dependencies}
        atomic_json(self.path, payload)
        return {"file_count": len(self.file_digests), "symbol_count": len(self.symbols), "changed_files": changed, "path": str(self.path)}

    def load(self, optional: bool = False) -> dict[str, Any]:
        if not self.path.exists():
            if optional: return {}
            self.build()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.schema_version: raise ValueError("unsupported semantic index schema")
        self.symbols = [Symbol(**item) for item in payload.get("symbols", [])]; self.references = payload.get("references", {}); self.dependencies = payload.get("dependencies", {}); self.file_digests = payload.get("file_digests", {})
        return payload

    def search_symbols(self, query: str, kind: str | None = None) -> list[Symbol]:
        self.load(); needle = query.casefold(); return [symbol for symbol in self.symbols if needle in symbol.name.casefold() and (kind is None or symbol.kind == kind)][:100]

    def find_references(self, symbol: str) -> list[dict[str, Any]]:
        self.load(); return self.references.get(symbol, [])[:500]

    def dependency_context(self, path: str) -> dict[str, list[str]]:
        self.load(); direct = self.dependencies.get(path, []); reverse = [source for source, targets in self.dependencies.items() if path in targets or Path(path).stem in targets]; return {"direct": direct, "reverse": reverse}

    def related_tests(self, path: str) -> list[str]:
        self.load(); stem = Path(path).stem.casefold(); return [relative for relative in self.file_digests if ("test" in Path(relative).name.casefold() or "spec" in Path(relative).name.casefold()) and (stem in Path(relative).name.casefold() or any(stem in dep.casefold() for dep in self.dependencies.get(relative, [])))]


def search_symbols(workspace: str | Path, query: str, kind: str | None = None) -> list[Symbol]:
    return SemanticIndex(WorkspaceContext.build(workspace)).search_symbols(query, kind)


def find_references(workspace: str | Path, symbol: str) -> list[dict[str, Any]]:
    return SemanticIndex(WorkspaceContext.build(workspace)).find_references(symbol)


def dependency_context(workspace: str | Path, path: str) -> dict[str, list[str]]:
    return SemanticIndex(WorkspaceContext.build(workspace)).dependency_context(path)


def related_tests(workspace: str | Path, path: str) -> list[str]:
    return SemanticIndex(WorkspaceContext.build(workspace)).related_tests(path)
