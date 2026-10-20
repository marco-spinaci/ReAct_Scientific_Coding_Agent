
from pathlib import Path

from src.sandbox.local_executor import LocalSandboxExecutor


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
