import json

from tifa.doctor import doctor
from tifa.execution import ExecutionResult
from tifa.performance import benchmark_workspace


def test_workspace_benchmark_small_is_reproducible(tmp_path):
    output = tmp_path / "performance.json"
    report = benchmark_workspace(output, sizes=(20, 50), repeats=2)
    assert report["status"] == "passed" and [row["file_count"] for row in report["sizes"]] == [20, 50]
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "tifa-workspace-benchmark.v1"


def test_doctor_reports_health_from_machine_checks(tmp_path, monkeypatch):
    def check_output(argv, **kwargs):
        if argv[:2] == ["docker", "version"]: return "29.5.3\n"
        if argv[:2] == ["docker", "info"]: return json.dumps({"nvidia": {}, "runc": {}})
        raise AssertionError(argv)

    class URLResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps({"models": [{"name": "qwen2.5-coder:3b"}]}).encode()

    monkeypatch.setattr("tifa.doctor.subprocess.check_output", check_output)
    monkeypatch.setattr("tifa.doctor.urlopen", lambda *args, **kwargs: URLResponse())
    monkeypatch.setattr("tifa.doctor.DockerExecutionBackend.execute", lambda self, request: ExecutionResult(0, "65532\n", "", 1, False, False, "docker", "container_strong", "sha256:test"))
    result = doctor(tmp_path)
    assert result["status"] == "healthy" and result["checks"]["sandbox"]["image_digest"] == "sha256:test"
