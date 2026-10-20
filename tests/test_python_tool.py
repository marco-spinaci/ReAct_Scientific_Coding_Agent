
from pathlib import Path

from src.sandbox.local_executor import LocalSandboxExecutor
from src.tools.python_tool import PythonExecutionTool


def test_python_execution_tool_returns_observation(tmp_path: Path) -> None:
    code = """
print("hello from tool")
"""

    sandbox = LocalSandboxExecutor(
            timeout_seconds = 5,
            output_dir = tmp_path
            )

    tool = PythonExecutionTool(sandbox = sandbox)

    result = tool.run(code)

    assert result.success is True
    assert "success: True" in result.observation
    assert "hello from tool" in result.observation
    assert result.generated_files == []


def test_python_execution_tool_reports_generated_files(tmp_path: Path) -> None:
    code = """
import pandas as pd
import plotly.express as px

df = pd.DataFrame({
    "x": [1, 2, 3],
    "y": [1, 4, 9],
})

fig = px.line(df, x="x", y="y")
fig.write_html("tool_plot.html")
print("done")
"""

    sandbox = LocalSandboxExecutor(
            timeout_seconds = 10,
            output_dir=tmp_path,
            )

    tool = PythonExecutionTool(sandbox=sandbox)

    result = tool.run(code)

    assert result.success is True
    assert "generated_files:" in result.observation
    assert len(result.generated_files) == 1

    generated_file = Path(result.generated_files[0])
    assert generated_file.exists()
    assert generated_file.name == "tool_plot.html"


def test_python_execution_tool_reports_policy_error(tmp_path: Path) -> None:
    code = """
import os
print(os.getcwd())
"""

    sandbox = LocalSandboxExecutor(
            timeout_seconds = 5,
            output_dir = tmp_path)

    tool = PythonExecutionTool(sandbox = sandbox)

    result = tool.run(code)

    assert result.success is False
    assert "success: False" in result.observation
    assert "error_message:" in result.observation
    assert "os" in result.observation

