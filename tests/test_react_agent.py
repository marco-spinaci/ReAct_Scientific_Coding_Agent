import json
import pytest

from pathlib import Path

from src.agent.react_agent import ReActCodingAgent
from src.llm.base import BaseLLMClient, ChatMessage
from src.llm.errors import LLMClientError, LLMConnectionError, LLMServiceError, LLMTimeoutError
from src.llm.fake_client import FakeLLMClient
from src.sandbox.local_executor import LocalSandboxExecutor
from src.tools.python_tool import PythonExecutionTool


class SequenceLLMClient(BaseLLMClient):
    """Return predefined LLM responses in sequence."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0
        self.messages_per_call: list[list[ChatMessage]] = []

    def generate(self, messages: list[ChatMessage]) -> str:
        self.messages_per_call.append(list(messages))

        response = self.responses[self.call_count]
        self.call_count += 1

        return response


class FailingLLMClient(BaseLLMClient):
    """Simulate an LLM backend that raises a predefined error."""
    def __init__(self, error: LLMClientError) -> None:
        self.error = error
        self.call_count = 0

    def generate(self, messages: list[ChatMessage]) -> str:
        self.call_count += 1
        raise self.error


def make_action_response(code: str | None = None, final_answer: str | None = None) -> str:
    return json.dumps({
        "thought": "Complete the requested task.",
        "action": "execute_python" if code is not None else "final_answer",
        "code": code,
        "final_answer": final_answer,
        })


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


def test_react_agent_recovers_from_invalid_llm_response(tmp_path: Path) -> None:
    valid_response = json.dumps({
        "thought": "Return the requested answer.",
        "action": "final_answer",
        "code": None,
        "final_answer": "Recovery successful.",
        })

    llm = SequenceLLMClient(responses = ["This is not JSON.", valid_response])

    sandbox = LocalSandboxExecutor(timeout_seconds = 10, output_dir = tmp_path)
    python_tool = PythonExecutionTool(sandbox = sandbox)

    agent = ReActCodingAgent(llm = llm, python_tool = python_tool, max_iterations = 2)

    result = agent.run("Return a final answer.")

    assert result.success is True
    assert result.iterations == 2
    assert result.final_answer == "Recovery successful."
    assert llm.call_count == 2

    retry_prompt = llm.messages_per_call[1][-1].content

    assert "Validation error:" in retry_prompt


def test_react_agent_fails_after_repeated_invalid_responses(tmp_path: Path) -> None:
    llm = SequenceLLMClient(responses = [
        "This is not JSON.",
        "[]",
        ])

    sandbox = LocalSandboxExecutor(timeout_seconds = 10, output_dir = tmp_path)
    python_tool = PythonExecutionTool(sandbox = sandbox)

    agent = ReActCodingAgent(llm = llm, python_tool = python_tool, max_iterations = 2)

    result = agent.run("Return a final answer.")

    assert result.success is False
    assert result.iterations == 2
    assert result.generated_files == []
    assert llm.call_count == 2
    assert "Invalid LLM response" in result.final_answer


@pytest.mark.parametrize("error", [
    LLMTimeoutError("Request timed out."),
    LLMConnectionError("Server unreachable."),
    LLMServiceError("HTTP 503."),
    ])
def test_react_agent_returns_controlled_failure_on_llm_error(tmp_path: Path, error: LLMClientError) -> None:
    llm = FailingLLMClient(error = error)

    sandbox = LocalSandboxExecutor(timeout_seconds = 10, output_dir = tmp_path)
    python_tool = PythonExecutionTool(sandbox = sandbox)

    agent = ReActCodingAgent(llm = llm, python_tool = python_tool, max_iterations = 3)

    result = agent.run("Generate a plot.")

    assert result.success is False
    assert result.iterations == 1
    assert result.generated_files == []
    assert llm.call_count == 1

    assert "LLM backend failed" in result.final_answer
    assert type(error).__name__ in result.final_answer


def test_react_agent_validates_expected_html_artifact(tmp_path: Path) -> None:
    code = """
from pathlib import Path
Path("plot.html").write_text("<html><body>Test</body></html>", encoding="utf-8")
"""
    llm = SequenceLLMClient(responses = [make_action_response(code = code)])

    sandbox = LocalSandboxExecutor(timeout_seconds = 10, output_dir = tmp_path)
    agent = ReActCodingAgent(llm = llm, python_tool = PythonExecutionTool(sandbox = sandbox), max_iterations = 2)

    result = agent.run("Generate plot.html", expected_html_files = ["plot.html"])

    assert result.success is True
    assert result.iterations == 1
    assert len(result.generated_files) == 1
    assert Path(result.generated_files[0]).is_file()


def test_react_agent_retries_when_html_artifact_is_missing(tmp_path: Path) -> None:
    failing_code = "print('Task completed')"

    corrected_code = """
from pathlib import Path
Path("plot.html").write_text("<html><body>Test</body></html>", encoding="utf-8")
"""
    llm = SequenceLLMClient(responses = [
        make_action_response(code = failing_code),
        make_action_response(code = corrected_code),
        ])

    sandbox = LocalSandboxExecutor(timeout_seconds = 10, output_dir = tmp_path)
    agent = ReActCodingAgent(llm = llm, python_tool = PythonExecutionTool(sandbox = sandbox), max_iterations = 2)

    result = agent.run("Generate plot.html", expected_html_files = ["plot.html"])

    assert result.success is True
    assert result.iterations == 2
    assert llm.call_count == 2
    assert "Artifact validation failed" in llm.messages_per_call[1][-1].content
    assert len(result.generated_files) == 1


def test_react_agent_fails_when_expected_html_is_never_generated(tmp_path: Path) -> None:
    response = make_action_response(code = "print('Done')")

    llm = SequenceLLMClient(responses = [response, response])

    sandbox = LocalSandboxExecutor(timeout_seconds = 10, output_dir = tmp_path)
    agent = ReActCodingAgent(llm = llm, python_tool = PythonExecutionTool(sandbox = sandbox), max_iterations = 2)

    result = agent.run("Generate plot.html", expected_html_files = ["plot.html"])

    assert result.success is False
    assert result.iterations == 2
    assert result.generated_files == []
    assert "plot.html" in result.final_answer


def test_react_agent_rejects_final_answer_when_html_is_required(tmp_path: Path) -> None:
    response = make_action_response(final_answer = "The plot has been generated.")

    llm = SequenceLLMClient(responses = [response, response])

    sandbox = LocalSandboxExecutor(timeout_seconds = 10, output_dir = tmp_path)
    agent = ReActCodingAgent(llm = llm, python_tool = PythonExecutionTool(sandbox = sandbox), max_iterations = 2)

    result = agent.run("Generate plot.html", expected_html_files = ["plot.html"])

    assert result.success is False
    assert result.iterations == 2
    assert result.generated_files == []
    assert llm.call_count == 2


def test_react_agent_does_not_accept_stale_html_artifact(tmp_path: Path) -> None:
    (tmp_path / "plot.html").write_text("<html>Old result</html>", encoding = "utf-8")

    response = make_action_response(code = "print('Done')")

    llm = SequenceLLMClient(responses = [response])

    sandbox = LocalSandboxExecutor(timeout_seconds = 10, output_dir = tmp_path)
    agent = ReActCodingAgent(llm = llm, python_tool = PythonExecutionTool(sandbox = sandbox), max_iterations = 1)

    result = agent.run("Generate plot.html", expected_html_files = ["plot.html"])

    assert result.success is False
    assert result.generated_files == []


def test_react_agent_includes_expected_html_in_initial_prompt(tmp_path: Path) -> None:
    llm = SequenceLLMClient(responses = [
        make_action_response(final_answer = "Done.")
        ])

    sandbox = LocalSandboxExecutor(timeout_seconds = 10, output_dir = tmp_path)
    agent = ReActCodingAgent(llm = llm, python_tool = PythonExecutionTool(sandbox = sandbox), max_iterations = 1)

    result = agent.run("Generate a visualization.", expected_html_files = ["network.html"])

    initial_prompt = llm.messages_per_call[0][-1].content

    assert "network.html" in initial_prompt
    assert "Required HTML artifacts" in initial_prompt
    assert result.success is False


def test_react_agent_recovers_from_invalid_json_and_generates_html(tmp_path: Path) -> None:
    code = """
from pathlib import Path
Path("network.html").write_text("<html><body>Network</body></html>", encoding="utf-8")
"""
    llm = SequenceLLMClient(responses = [
        "This is not valid JSON.",
        make_action_response(code = code),
        ])

    sandbox = LocalSandboxExecutor(timeout_seconds = 10, output_dir = tmp_path)
    agent = ReActCodingAgent(llm = llm, python_tool = PythonExecutionTool(sandbox = sandbox), max_iterations = 2)

    result = agent.run("Generate a visualization.", expected_html_files = ["network.html"])

    assert result.success is True
    assert result.iterations == 2
    assert llm.call_count == 2
    assert len(result.generated_files) == 1
    assert Path(result.generated_files[0]).name == "network.html"


def test_react_agent_rejects_final_answer_after_missing_html(tmp_path: Path) -> None:
    llm = SequenceLLMClient(responses = [
        make_action_response(code = "print('Execution completed')"),
        make_action_response(final_answer = "The visualization was generated."),
        ])

    sandbox = LocalSandboxExecutor(timeout_seconds = 10, output_dir = tmp_path)
    agent = ReActCodingAgent(llm = llm, python_tool = PythonExecutionTool(sandbox = sandbox), max_iterations = 2)

    result = agent.run("Generate a visualization.", expected_html_files = ["network.html"])

    assert result.success is False
    assert result.iterations == 2
    assert result.generated_files == []
    assert llm.call_count == 2
    assert "network.html" in result.final_answer
