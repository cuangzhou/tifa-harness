from tifa.workspace import WorkspaceContext


def test_workspace_index_respects_ignores_and_is_incremental(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / ".tifaignore").write_text("private/**\n", encoding="utf-8")
    (tmp_path / "kept.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("x=2\n", encoding="utf-8")
    (tmp_path / "private").mkdir(); (tmp_path / "private" / "secret.py").write_text("token\n", encoding="utf-8")
    first = WorkspaceContext.build(tmp_path); second = WorkspaceContext.build(tmp_path)
    assert "kept.py" in first.index and "ignored.py" not in first.index and "private/secret.py" not in first.index
    assert first.stats["changed_files"] >= 1 and second.stats["changed_files"] == 0


def test_fingerprint_changes_only_when_indexed_content_changes(tmp_path):
    file = tmp_path / "a.py"; file.write_text("x=1\n", encoding="utf-8")
    before = WorkspaceContext.build(tmp_path).fingerprint(); file.write_text("x=2\n", encoding="utf-8")
    assert WorkspaceContext.build(tmp_path).fingerprint() != before
