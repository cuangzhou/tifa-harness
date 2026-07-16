from pathlib import Path

import pytest

from tifa.context_manager import ContextManager
from tifa.memory import LayeredMemory
from tifa.tools import ToolRegistry
from tifa.workspace import WorkspaceContext


def registry(tmp_path: Path, **kwargs):
    return ToolRegistry(WorkspaceContext.build(tmp_path), **kwargs)


def test_path_escape_and_patch_uniqueness(tmp_path: Path):
    tools = registry(tmp_path, approval_policy="never")
    with pytest.raises(ValueError, match="escapes"):
        tools.run("read_file", {"path": "../secret"})
    (tmp_path / "a.txt").write_text("x x", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly once"):
        tools.run("patch_file", {"path": "a.txt", "old_text": "x", "new_text": "y"})


def test_risky_tool_requires_approval(tmp_path: Path):
    tools = registry(tmp_path, approval_policy="on-risk", approver=lambda *_: False)
    with pytest.raises(PermissionError):
        tools.run("write_file", {"path": "a.txt", "content": "x"})


def test_memory_summary_invalidates_after_write():
    memory = LayeredMemory(); memory.after_tool("read_file", {"path": "a.py"}, "print('x')")
    assert "a.py" in memory.state["file_summaries"]
    memory.after_tool("write_file", {"path": "a.py"}, "ok")
    assert "a.py" not in memory.state["file_summaries"]


def test_context_budget_records_reductions(tmp_path: Path):
    workspace = WorkspaceContext.build(tmp_path); manager = ContextManager(workspace, total_budget=5000)
    history = [{"role": "tool", "tool": "read_file", "arguments": {"path": f"{i}.py"}, "content": "x" * 3000} for i in range(10)]
    built = manager.build("request", LayeredMemory(), history, registry(tmp_path, approval_policy="never").schemas(), ["m" * 3000])
    assert built.metadata["budget_reductions"]
    assert built.cache_key


def test_delegate_registry_is_read_only(tmp_path: Path):
    tools = ToolRegistry(WorkspaceContext.build(tmp_path), approval_policy="never", delegate=lambda *_: "ok", depth=1, max_depth=1)
    assert set(tools.names) == {"list_files", "read_file", "search"}
