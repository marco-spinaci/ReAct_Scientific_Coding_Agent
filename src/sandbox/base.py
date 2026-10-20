# Enable postponed evaluation of type annotations.
# Useful when using modern type hints and forward references.
from __future__ import annotations


# dataclass automatically generates methods such as __init__.
from dataclasses import dataclass


# ABC and abstractmethod are used to define an abstract base class,
# i.e. an interface that specific sandbox executors must implement.
from abc import ABC, abstractmethod


@dataclass
class SandboxResult:
    """
    Result returned by a sandbox execution.

    This object standardizes the output of different sandbox backends
    such as a local subprocess executor, Docker executor, or E2B executor.

    Attributes
    ----------
    success:
        True if the code passed the policy checks and executed with return code 0.
    stdout:
        Standard output captured from the executed process.
    stderr:
        Standard error captured from the executed process.
    return_code:
        Process return code. None if execution did not start.
    generated_files:
        List of files generated inside the sandbox working directory.
    execution_time:
        Wall-clock execution time in seconds.
    error_message:
        Optional high-level error message.
    """

    success: bool
    stdout: str
    stderr: str
    return_code: int | None
    generated_files: list[str]
    execution_time: float
    error_message: str | None = None


class BaseSandboxExecutor(ABC):
    """Abstract base class for all sandbox executors."""

    @abstractmethod
    def run(self, code: str) -> SandboxResult:
        """
        Execute Python code inside a sandbox and return a SandboxResult.

        Parameters
        ----------
        code: 
            Python source code to execute.
        """
        
        raise NotImplementedError

