
from pathlib import Path

from src.agent.react_agent import ReActCodingAgent
from src.llm.fake_client import FakeLLMClient
from src.sandbox.local_executor import LocalSandboxExecutor
from src.tools.python_tool import PythonExecutionTool


def build_agent(tmp_path: Path, mode: str) -> ReActCodingAgent:
    
    sandbox = LocalSandboxExecutor(timeout_seconds = 10, output_dir = tmp_path)
    
    python_tool = PythonExecutionTool(sandbox = sandbox)
    
    llm = FakeLLMClient(mode = mode)

    return ReActCodingAgent(
            llm = llm,
            python_tool = python_tool,
            max_iterations = 2,
            )


def test_react_agent_success_generates_html(tmp_path: Path) -> None:
    
    agent = build_agent(tmp_path = tmp_path, mode = "success")

    result = agent.run("Generate a synthetic supplier-customer network and save a Plotly HTML plot.")

    assert result.success is True
    assert result.iterations == 1
    assert len(result.generated_files) == 1

    generated_file = Path(result.generated_files[0])
    assert generated_file.exists()
    assert generated_file.suffix == ".html"
    assert "completed successfully" in result.final_answer


def test_react_agent_retries_after_failure(tmp_path: Path) -> None:

    agent = build_agent(tmp_path = tmp_path, mode = "broken_then_success")

    result = agent.run("Generate a synthetic supplier-customer network and save a Plotly HTML plot.")

    assert result.success is True
    assert result.iterations == 2
    assert len(result.generated_files) == 1

    generated_file = Path(result.generated_files[0])
    assert generated_file.exists()
    assert generated_file.name == "synthetic_network_degree_distribution.html"


def test_react_agent_final_answer_without_tool(tmp_path: Path) -> None:
    
    agent = build_agent(tmp_path = tmp_path, mode = "final_only")

    result = agent.run("Answer directly.")

    assert result.success is True
    assert result.iterations == 1
    assert result.generated_files == []
    assert "deterministic final answer" in result.final_answer
