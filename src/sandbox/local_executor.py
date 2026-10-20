from __future__ import annotations

import sys, logging

# Used to run the generated Python script in a separate process.
import subprocess

# Used to create a temporary working directory for each execution.
import tempfile

import time
from pathlib import Path

from src.sandbox.base import BaseSandboxExecutor, SandboxResult
from src.sandbox.errors import SandboxPolicyError
from src.sandbox.policy import SandboxPolicy


logger = logging.getLogger(__name__)


class LocalSandboxExecutor(BaseSandboxExecutor):
    """
    Local subprocess-based sandbox executor.

    This backend is intended for local development and testing.
    It provides:
    - separate process execution;
    - timeout;
    - temporary working directory;
    - stdout/stderr capture;
    - generated file collection;
    - AST-based policy checks.
    
    Important:
    ----------
    This implementation is useful for development and testing,
    but it is NOT a full production-grade security sandbox.
    """

    def __init__(self, timeout_seconds: int = 10, policy: SandboxPolicy | None = None, output_dir: str | Path = "outputs") -> None:
        """
        Initialize the local sandbox executor.

        Parameters
        ----------
        timeout_seconds:
            Maximum execution time allowed for generated code.

        policy:
            Optional SandboxPolicy instance.
            If None, a default SandboxPolicy is created.

        output_dir:
            Directory where files generated inside the temporary sandbox
            will be copied after execution.
        """

        self.timeout_seconds = timeout_seconds
        self.policy = policy or SandboxPolicy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    
    def run(self, code: str) -> SandboxResult:
        """
        Validate and execute generated Python code.

        The execution flow is:
        1. measure start time;
        2. validate the code with the policy;
        3. create a temporary working directory;
        4. write the code to generated_code.py;
        5. execute it in a subprocess;
        6. collect generated files;
        7. return a standardized SandboxResult.
        """

        start_time = time.perf_counter()
        logger.info("Starting local sandbox execution.")

        # Validate the generated code before execution.
        # This does not run the code; it only checks the AST.
        try:
            self.policy.validate(code)
        except SandboxPolicyError as exc:
            logger.warning("Sandbox policy violation: %s", exc)
            return SandboxResult(
                    success = False,
                    stdout = "",
                    stderr = "",
                    return_code = None,
                    generated_files = [],
                    execution_time = time.perf_counter() - start_time,
                    error_message = str(exc),
                    )

        # Create a temporary working directory.
        # It will be automatically deleted when the with-block exits.
        with tempfile.TemporaryDirectory(prefix = "agent_sandbox_") as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "generated_code.py"

            script_path.write_text(code, encoding = "utf-8")
            logger.debug("Generated code written to temporary script: %s", script_path)
            
            # Execute the generated script in a separate Python process.
            # -> sys.executable is the path to the Python interpreter currently running this program
            try:
                completed = subprocess.run(
                        [sys.executable, str(script_path)],
                        cwd = tmp_path,
                        capture_output = True,
                        text = True,
                        timeout = self.timeout_seconds,
                        check = False,
                        )
            except subprocess.TimeoutExpired as exc:
                logger.error("Sandbox execution timed out after %s seconds.", self.timeout_seconds)
                return SandboxResult(
                        success = False,
                        stdout = exc.stdout or "",
                        stderr = exc.stderr or "",
                        return_code = None,
                        generated_files = [],
                        execution_time = time.perf_counter() - start_time,
                        error_message = (
                            f"Execution timed out after "
                            f"{self.timeout_seconds} seconds."),
                        )

            generated_files = self._collect_generated_files(tmp_path)

            copied_files = []
            for file_path in generated_files:
                destination = self.output_dir / file_path.name
                destination.write_bytes(file_path.read_bytes())
                copied_files.append(str(destination))

            execution_time = time.perf_counter() - start_time
            success = completed.returncode == 0

            logger.info(
                    "Sandbox execution finished. success=%s return_code=%s "
                    "generated_files=%d execution_time=%.3fs",
                    success, completed.returncode, len(copied_files), execution_time
                    )

            if completed.stderr.strip():
                logger.debug("Sandbox stderr: %s", completed.stderr.strip())

            return SandboxResult(
                    success = success,
                    stdout = completed.stdout,
                    stderr = completed.stderr,
                    return_code = completed.returncode,
                    generated_files = copied_files,
                    execution_time = execution_time,
                    error_message = None if success else "Execution failed.",
                    )

    
    def _collect_generated_files(self, workdir: Path) -> list[Path]:
        """
        Collect files generated by the executed script.
        
        Parameters
        ----------
        workdir:
            Temporary working directory used during execution.

        Returns
        -------
        list[Path]
            Files created inside workdir, excluding generated_code.py.
        """
        
        files = []

        for path in workdir.iterdir():
            if path.is_file() and path.name != "generated_code.py":
                files.append(path)

        return files
