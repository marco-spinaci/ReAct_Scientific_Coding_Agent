
import os
import signal
import sys
import subprocess

import pytest

from pathlib import Path

from src.tools.python_tool import PythonExecutionTool
from src.sandbox.local_executor import LocalSandboxExecutor
from src.sandbox.policy import SandboxPolicy


def test_local_executor_runs_safe_code(tmp_path: Path) -> None:
    code = """
print("Hello from sandbox")
"""
    executor = LocalSandboxExecutor(
            timeout_seconds = 5,
            output_dir = tmp_path,
            )

    result = executor.run(code)

    assert result.success is True
    assert result.return_code == 0
    assert "Hello from sandbox" in result.stdout
    assert result.stderr == ""
    assert result.generated_files == []
    assert result.error_message is None


def test_local_executor_returns_policy_error(tmp_path: Path) -> None:
    code = """
import os
print(os.getcwd())
"""
    executor = LocalSandboxExecutor(
            timeout_seconds = 5,
            output_dir = tmp_path,
            )

    result = executor.run(code)

    assert result.success is False
    assert result.return_code is None
    assert "os" in result.error_message
    assert result.generated_files == []


def test_local_executor_handles_runtime_error(tmp_path: Path) -> None:
    code = """
print("before error")
raise ValueError("Intentional failure")
"""
    executor = LocalSandboxExecutor(
            timeout_seconds = 5,
            output_dir = tmp_path,
            )

    result = executor.run(code)

    assert result.success is False
    assert result.return_code != 0
    assert "before error" in result.stdout
    assert "ValueError" in result.stderr
    assert result.error_message == "Execution failed."


def test_local_executor_handles_timeout(tmp_path: Path) -> None:
    code = """
while True:
    pass
"""
    executor = LocalSandboxExecutor(
            timeout_seconds = 1,
            output_dir = tmp_path,
            )

    result = executor.run(code)

    assert result.success is False
    assert result.return_code is None
    assert "timed out" in result.error_message


def test_local_executor_collects_generated_html_file(tmp_path: Path) -> None:
    code = """
import pandas as pd
import plotly.express as px

df = pd.DataFrame({
    "x": [1, 2, 3],
    "y": [1, 4, 9],
})

fig = px.line(df, x="x", y="y")
fig.write_html("test_plot.html")

print("plot generated")
"""
    executor = LocalSandboxExecutor(
            timeout_seconds = 10,
            output_dir = tmp_path,
            )

    result = executor.run(code)

    assert result.success is True
    assert "plot generated" in result.stdout
    assert len(result.generated_files) == 1

    generated_file = Path(result.generated_files[0])
    assert generated_file.exists()
    assert generated_file.name == "test_plot.html"
    assert generated_file.suffix == ".html"


def test_local_executor_timeout_preserves_text_output(tmp_path: Path) -> None:
    code = """
import time
import warnings

print("partial stdout", flush=True)
warnings.warn("partial stderr")

time.sleep(60)
"""
    executor = LocalSandboxExecutor(timeout_seconds = 1, output_dir = tmp_path)

    result = executor.run(code)

    assert result.success is False
    assert result.return_code is None
    assert result.stdout == "partial stdout\n"
    assert "partial stderr" in result.stderr
    assert "timed out" in result.error_message


def test_local_executor_rejects_symbolic_link_artifact(tmp_path: Path) -> None:
    code = """
from pathlib import Path

Path("real.html").write_text("<html>ok</html>", encoding="utf-8")
Path("alias.html").symlink_to("real.html")
"""
    executor = LocalSandboxExecutor(output_dir = tmp_path)
    result = executor.run(code)

    assert result.success is False
    assert result.return_code == 0
    assert result.generated_files == []
    assert "Symbolic links" in result.error_message
    assert list(tmp_path.iterdir()) == []


def test_local_executor_rejects_oversized_artifact(tmp_path: Path) -> None:
    code = """
from pathlib import Path

Path("large.html").write_text("0123456789abcdef", encoding="utf-8")
"""
    executor = LocalSandboxExecutor(output_dir = tmp_path, max_artifact_bytes = 8)
    result = executor.run(code)

    assert result.success is False
    assert result.return_code == 0
    assert result.generated_files == []
    assert "exceeds maximum size" in result.error_message


def test_local_executor_reports_artifact_copy_failure(tmp_path: Path, monkeypatch) -> None:
    def fail_copy(*args, **kwargs):
        raise OSError("Simulated copy error")

    monkeypatch.setattr(LocalSandboxExecutor, "_copy_generated_file", fail_copy)

    code = """
from pathlib import Path

Path("ok.html").write_text("<html>ok</html>", encoding="utf-8")
"""
    executor = LocalSandboxExecutor(output_dir = tmp_path)
    result = executor.run(code)

    assert result.success is False
    assert result.return_code == 0
    assert result.generated_files == []
    assert "Simulated copy error" in result.error_message


def test_local_executor_preserves_artifacts_across_runs(tmp_path: Path) -> None:
    executor = LocalSandboxExecutor(output_dir = tmp_path)
    first_code = """
from pathlib import Path
Path("network.html").write_text("First execution", encoding="utf-8")
"""
    second_code = """
from pathlib import Path
Path("network.html").write_text("Second execution", encoding="utf-8")
"""
    first_result = executor.run(first_code)
    second_result = executor.run(second_code)

    assert first_result.success is True
    assert second_result.success is True

    first_file = Path(first_result.generated_files[0])
    second_file = Path(second_result.generated_files[0])

    assert first_file != second_file
    assert first_file.name == second_file.name == "network.html"

    assert first_file.parent != second_file.parent
    assert first_file.parent.parent == tmp_path
    assert second_file.parent.parent == tmp_path

    assert first_file.read_text(encoding = "utf-8") == "First execution"
    assert second_file.read_text(encoding = "utf-8") == "Second execution"


def test_local_executor_groups_artifacts_by_execution(tmp_path: Path) -> None:
    code = """
from pathlib import Path

Path("network.html").write_text("Network", encoding="utf-8")
Path("degrees.html").write_text("Degrees", encoding="utf-8")
"""
    executor = LocalSandboxExecutor(output_dir = tmp_path)

    result = executor.run(code)

    assert result.success is True
    assert len(result.generated_files) == 2

    paths = [Path(file_path) for file_path in result.generated_files]

    assert {path.name for path in paths} == {"network.html", "degrees.html"}
    assert len({path.parent for path in paths}) == 1

    for path in paths:
        assert path.is_file()


def test_local_executor_rolls_back_failed_artifact_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code = """
from pathlib import Path

Path("first.html").write_text("First", encoding="utf-8")
Path("second.html").write_text("Second", encoding="utf-8")
"""
    original_copy = LocalSandboxExecutor._copy_generated_file
    copy_count = 0

    def fail_on_second_copy(self: LocalSandboxExecutor, file_path: Path, destination_dir: Path) -> str:
        nonlocal copy_count

        copy_count += 1

        if copy_count == 2:
            raise OSError("Simulated failure on second artifact")

        return original_copy(self, file_path, destination_dir)

    monkeypatch.setattr(LocalSandboxExecutor, "_copy_generated_file", fail_on_second_copy)

    executor = LocalSandboxExecutor(output_dir = tmp_path)

    result = executor.run(code)

    assert result.success is False
    assert result.return_code == 0
    assert result.generated_files == []
    assert "Simulated failure on second artifact" in result.error_message

    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "This test requires Linux /proc.")
def test_local_executor_kills_child_process_on_timeout(tmp_path: Path) -> None:
    code = """
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

print(child.pid, flush=True)

time.sleep(60)
"""
    # Allow subprocess and sys imports only for this test.
    policy = SandboxPolicy()
    policy.blocked_modules = set(policy.blocked_modules) - {"subprocess", "sys"}

    executor = LocalSandboxExecutor(timeout_seconds = 2, policy = policy, output_dir = tmp_path)

    child_pid: int | None = None

    try:
        result = executor.run(code)

        assert result.success is False
        assert "timed out" in result.error_message

        child_pid = int(result.stdout.strip().splitlines()[0])

        child_status = Path(f"/proc/{child_pid}/stat")

        if child_status.exists():
            process_state = child_status.read_text().split()[2]
            assert process_state in {"Z", "X"}

    finally:
        # Defensive cleanup if the implementation under test did not terminate the child.
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)

            except ProcessLookupError:
                pass


def test_local_executor_rejects_excessive_stdout(tmp_path: Path) -> None:
    code = """
print("X" * 100_000, flush=True)
"""
    executor = LocalSandboxExecutor(output_dir = tmp_path, max_output_bytes = 1024)

    result = executor.run(code)

    assert result.success is False
    assert result.return_code is None
    assert result.generated_files == []

    assert "Captured output exceeded" in result.error_message
    assert len(result.stdout.encode("utf-8")) <= 1024


def test_local_executor_captures_stdout_and_stderr(tmp_path: Path) -> None:
    code = """
import warnings

print("Standard output", flush=True)
warnings.warn("Standard error")
"""
    executor = LocalSandboxExecutor(output_dir = tmp_path, max_output_bytes = 4096)

    result = executor.run(code)

    assert result.success is True
    assert result.return_code == 0

    assert "Standard output" in result.stdout
    assert "Standard error" in result.stderr


def test_local_executor_enforces_combined_output_limit(tmp_path: Path) -> None:
    code = """
import warnings

print("A" * 1000, flush=True)
warnings.warn("B" * 1000)
"""
    executor = LocalSandboxExecutor(output_dir = tmp_path, max_output_bytes = 1200)

    result = executor.run(code)

    assert result.success is False
    assert "Captured output exceeded" in result.error_message

    captured_bytes = len(result.stdout.encode("utf-8")) + len(result.stderr.encode("utf-8"))

    assert captured_bytes <= 1200


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "Linux-specific resource limits.")
def test_local_executor_enforces_file_size_limit(tmp_path: Path) -> None:
    code = """
from pathlib import Path

Path("large.bin").write_bytes(b"A" * 4096)
"""
    executor = LocalSandboxExecutor(
            timeout_seconds = 10,
            output_dir = tmp_path,
            max_file_size_bytes = 1024,
            )

    result = executor.run(code)

    assert result.success is False
    assert result.return_code != 0
    assert "File too large" in result.stderr

    for file_path in result.generated_files:
        assert Path(file_path).stat().st_size <= 1024


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "Linux-specific resource limits.")
def test_local_executor_enforces_cpu_time_limit(tmp_path: Path) -> None:
    code = """
while True:
    pass
"""
    executor = LocalSandboxExecutor(
            timeout_seconds = 30,
            output_dir = tmp_path,
            cpu_time_limit_seconds = 1,
            )

    result = executor.run(code)

    assert result.success is False
    assert result.return_code == -signal.SIGXCPU
    assert "CPU time limit exceeded" in result.error_message


@pytest.mark.parametrize("parameter_name, value", [
    ("cpu_time_limit_seconds", 0),
    ("cpu_time_limit_seconds", -1),
    ("cpu_time_limit_seconds", 1.5),
    ("max_file_size_bytes", 0),
    ("max_file_size_bytes", -1024),
    ("max_file_size_bytes", 2.5),
    ])
def test_local_executor_rejects_invalid_resource_limits(
        tmp_path: Path, parameter_name: str, value: int | float
        ) -> None:
    with pytest.raises(ValueError):
        LocalSandboxExecutor(output_dir = tmp_path, **{parameter_name: value})


