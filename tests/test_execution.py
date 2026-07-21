import os
from pathlib import Path
import pytest

from tifa import DockerExecutionBackend, ExecutionPolicy, ExecutionRequest, LocalExecutionBackend, ResourceLimits
from tifa.execution import command_argv


def test_local_backend_is_explicitly_degraded(tmp_path):
    result = LocalExecutionBackend().execute(ExecutionRequest(["python", "-c", "print('ok')"], tmp_path))
    assert result.exit_code == 0 and result.stdout.strip() == "ok"
    assert result.isolation_level == "local_degraded" and "network_not_isolated" in result.security_events


def test_command_argv_rejects_implicit_shell():
    with pytest.raises(ValueError): command_argv("echo ok && echo unsafe")


def test_command_argv_allows_operators_inside_quoted_argument():
    argv = command_argv('python -c "value = 1; assert value < 2; print(value)"')
    assert argv == ["python", "-c", "value = 1; assert value < 2; print(value)"]


@pytest.mark.skipif(os.getenv("TIFA_TEST_DOCKER") != "1", reason="set TIFA_TEST_DOCKER=1 for Docker integration")
def test_docker_backend_non_root_read_only_and_cleanup(tmp_path):
    backend = DockerExecutionBackend()
    result = backend.execute(ExecutionRequest(["python", "-c", "import os,pathlib; print(os.getuid()); pathlib.Path('/blocked').write_text('x')"], tmp_path))
    assert result.exit_code != 0 and "65532" in result.stdout
    assert result.isolation_level == "container_strong" and result.image_digest


@pytest.mark.skipif(os.getenv("TIFA_TEST_DOCKER") != "1", reason="set TIFA_TEST_DOCKER=1 for Docker integration")
def test_docker_timeout_removes_container(tmp_path):
    result = DockerExecutionBackend().execute(ExecutionRequest(["python", "-c", "import time; time.sleep(5)"], tmp_path, limits=ResourceLimits(timeout_seconds=1), policy=ExecutionPolicy()))
    assert result.timed_out


@pytest.mark.skipif(os.getenv("TIFA_TEST_DOCKER") != "1", reason="set TIFA_TEST_DOCKER=1 for Docker integration")
def test_docker_custom_seccomp_profile(tmp_path):
    profile = Path(__file__).parents[1] / "docker" / "seccomp-tifa-v1.json"
    result = DockerExecutionBackend().execute(ExecutionRequest(["python", "-c", "print('isolated')"], tmp_path, policy=ExecutionPolicy(seccomp_profile=str(profile))))
    assert result.exit_code == 0 and "isolated" in result.stdout and "docker_default_seccomp" not in result.security_events
