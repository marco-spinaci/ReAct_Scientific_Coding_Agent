from __future__ import annotations

from dataclasses import dataclass

from src.sandbox.base import BaseSandboxExecutor, SandboxResult


@dataclass
class ToolResult:
    """
    Standard result returned by an agent tool.

    This object is intentionally more agent-friendly than SandboxResult.
    The sandbox result contains low-level execution details, while ToolResult
    provides a compact observation that can be passed back to the agent.
    """

    success: bool
    observation: str
    generated_files: list[str]
    raw_result: SandboxResult


class PythonExecutionTool:
    """
    Agent tool used to execute Python code through a sandbox backend.

    The agent should not call the sandbox directly. 
    Instead, it calls this tool, which converts the sandbox output into an observation 
    suitable for the ReAct loop.

    This class acts as an adapter:
    - input: Python code as string;
    - internal step: execute through sandbox;
    - output: ToolResult with an agent-friendly observation.
    """

    
    name = "execute_python"
    description = (
            "Execute Python code in a controlled sandbox and return stdout, stderr, "
            "errors, and generated files.")

    
    def __init__(self, sandbox: BaseSandboxExecutor) -> Nonei:
        """
        Initialize the Python execution tool.

        Parameters
        ----------
        sandbox:
            Any concrete sandbox executor implementing BaseSandboxExecutor.

        Examples
        --------
        sandbox could be:
        - LocalSandboxExecutor
        - DockerSandboxExecutor
        - E2BSandboxExecutor

        The tool does not need to know which backend is used.
        """
        
        self.sandbox = sandbox

    
    def run(self, code: str) -> ToolResult:
        """
        Execute Python code through the configured sandbox.

        Parameters
        ----------
        code:
            Python source code to execute.

        Returns
        -------
        ToolResult
            Agent-friendly result containing:
            - success flag;
            - textual observation;
            - generated files;
            - raw low-level sandbox result.
        """
        
        sandbox_result = self.sandbox.run(code)
        
        observation = self._build_observation(sandbox_result)

        return ToolResult(
                success = sandbox_result.success,
                observation = observation,
                generated_files = sandbox_result.generated_files,
                raw_result = sandbox_result)

    
    def _build_observation(self, result: SandboxResult) -> str:
        """
        Build a textual observation from a SandboxResult.

        This method converts structured execution details into plain text.
        The resulting string is suitable for an agent loop, logs, or debugging.
        """

        parts: list[str] = []

        parts.append(f"success: {result.success}")

        if result.return_code is not None:
            parts.append(f"return_code: {result.return_code}")

        if result.stdout.strip():
            parts.append("stdout:")
            parts.append(result.stdout.strip())

        if result.stderr.strip():
            parts.append("stderr:")
            parts.append(result.stderr.strip())

        if result.error_message:
            parts.append("error_message:")
            parts.append(result.error_message)

        if result.generated_files:
            parts.append("generated_files:")
            for file_path in result.generated_files:
                parts.append(f"- {file_path}")

        if not result.stdout.strip() and not result.stderr.strip() and not result.error_message:
            parts.append("No output was produced.")

        return "\n".join(parts)
