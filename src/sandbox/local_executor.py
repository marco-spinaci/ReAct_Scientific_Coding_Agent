from __future__ import annotations

import os
import signal
import sys, logging

# Used to run the generated Python script in a separate process.
import subprocess

# Used to create a temporary working directory for each execution.
import tempfile

import time

import shutil

import selectors

from uuid import uuid4

from pathlib import Path

from src.sandbox.base import BaseSandboxExecutor, SandboxResult
from src.sandbox.errors import SandboxExecutionError, SandboxPolicyError
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

    def __init__(self, 
            timeout_seconds: int = 10, 
            policy: SandboxPolicy | None = None, 
            output_dir: str | Path = "outputs",
            max_artifact_bytes: int = 50 * 1024 * 1024,
            max_output_bytes: int = 1024 * 1024,
            cpu_time_limit_seconds: int | None = None,
            max_file_size_bytes: int | None = None,
            ) -> None:
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

        max_artifact_bytes:
            Maximum size per artifact.
        """
        if max_artifact_bytes <= 0:
            raise ValueError("max_artifact_bytes must be greater than zero.")

        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be greater than zero.")
        
        for parameter_name, value in (
                ("cpu_time_limit_seconds", cpu_time_limit_seconds),
                ("max_file_size_bytes", max_file_size_bytes),
                ):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{parameter_name} must be a positive integer or None.")

        if (cpu_time_limit_seconds is not None or max_file_size_bytes is not None) and not sys.platform.startswith("linux"):
            raise NotImplementedError("Process resource limits are currently supported only on Linux.")
        
        self.cpu_time_limit_seconds = cpu_time_limit_seconds
        self.max_file_size_bytes = max_file_size_bytes
        self.max_output_bytes = max_output_bytes
        self.max_artifact_bytes = max_artifact_bytes
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

            # Start the generated script in a new process session.
            process = subprocess.Popen(
                    self.build_execution_command(script_path),
                    cwd = tmp_path,
                    stdout = subprocess.PIPE,
                    stderr = subprocess.PIPE,
                    start_new_session = True,
                    )
            
            # `bytearray` allows binary data to be accumulated progressively, 
            # which is subsequently converted into strings.
            captured_output = {"stdout": bytearray(), "stderr": bytearray()}
            captured_bytes = 0

            failure_reason: str | None = None
            deadline = time.monotonic() + self.timeout_seconds

            try:
                with selectors.DefaultSelector() as selector:
                    assert process.stdout is not None
                    assert process.stderr is not None
                    
                    # Adds stdout and stderr to the list of file descriptors to be checked.
                    selector.register(process.stdout, selectors.EVENT_READ, data = "stdout")
                    selector.register(process.stderr, selectors.EVENT_READ, data = "stderr")

                    while selector.get_map():
                        remaining_time = deadline - time.monotonic()

                        if remaining_time <= 0:
                            failure_reason = f"Execution timed out after {self.timeout_seconds} seconds."
                            break
                        
                        # Waits until at least one of the descriptors is ready to be read.
                        ready_streams = selector.select(timeout = remaining_time)

                        for key, _ in ready_streams:
                            # The limit `max_output_bytes` applies to the total number of bytes collected 
                            # from stdout and stderr, not separately to the two streams.
                            remaining_capacity = self.max_output_bytes - captured_bytes

                            chunk = os.read(key.fileobj.fileno(), min(64 * 1024, remaining_capacity + 1))

                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue

                            accepted_chunk = chunk[:remaining_capacity]

                            captured_output[key.data].extend(accepted_chunk)
                            captured_bytes += len(accepted_chunk)

                            if len(chunk) > remaining_capacity:
                                failure_reason = f"Captured output exceeded {self.max_output_bytes} bytes."
                                break

                        if failure_reason is not None:
                            break

                # The process may still be running even if its output pipes are closed.
                if failure_reason is None:
                    try:
                        process.wait(timeout = max(0.0, deadline - time.monotonic()))

                    except subprocess.TimeoutExpired:
                        failure_reason = f"Execution timed out after {self.timeout_seconds} seconds."

            finally:
                # Stop the process group on timeout, output overflow, or incomplete execution.
                if failure_reason is not None or process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)

                    except ProcessLookupError:
                        pass

                    try:
                        process.wait(timeout = 2)

                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout = 2)

                if process.stdout is not None:
                    process.stdout.close()

                if process.stderr is not None:
                    process.stderr.close()

            # Converting bytes to strings.
            stdout = captured_output["stdout"].decode("utf-8", errors = "replace")
            stderr = captured_output["stderr"].decode("utf-8", errors = "replace")

            if failure_reason is not None:
                logger.warning("Sandbox execution failed: %s", failure_reason)

                return SandboxResult(
                        success = False,
                        stdout = stdout,
                        stderr = stderr,
                        return_code = None,
                        generated_files = [],
                        execution_time = time.perf_counter() - start_time,
                        error_message = failure_reason,
                        )

            completed = subprocess.CompletedProcess(
                    args = process.args,
                    returncode = process.returncode,
                    stdout = stdout,
                    stderr = stderr,
                    )

            # try:
            #     completed = subprocess.run(
            #             [sys.executable, str(script_path)],
            #             cwd = tmp_path,
            #             capture_output = True,
            #             text = True,
            #             timeout = self.timeout_seconds,
            #             check = False,
            #             )

            # except subprocess.TimeoutExpired as exc:
            #     logger.error("Sandbox execution timed out after %s seconds.", self.timeout_seconds)

            #     stdout = exc.stdout.decode("utf-8", errors = "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")

            #     stderr = exc.stderr.decode("utf-8", errors = "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")

            #     return SandboxResult(
            #             success = False,
            #             stdout = stdout,
            #             stderr = stderr,
            #             return_code = None,
            #             generated_files = [],
            #             execution_time = time.perf_counter() - start_time,
            #             error_message = (
            #                 f"Execution timed out after "
            #                 f"{self.timeout_seconds} seconds."),
            #             )

            try:
                generated_files = self._collect_generated_files(tmp_path)
                copied_files = self._save_generated_files(generated_files)

            except (SandboxExecutionError, OSError) as exc:
                logger.error("Artifact collection failed: %s", exc)

                return SandboxResult(
                        success = False,
                        stdout = completed.stdout,
                        stderr = completed.stderr,
                        return_code = completed.returncode,
                        generated_files = [],
                        execution_time = time.perf_counter() - start_time,
                        error_message = f"Artifact collection failed: {exc}",
                        )

            execution_time = time.perf_counter() - start_time
            success = completed.returncode == 0
            
            if success:
                error_message = None
            elif self.cpu_time_limit_seconds is not None and completed.returncode == -signal.SIGXCPU:
                error_message = f"CPU time limit exceeded ({self.cpu_time_limit_seconds} CPU seconds)."
            else:
                error_message = "Execution failed."
            
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
                    error_message = error_message,
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
        files: list[Path] = []

        for path in workdir.iterdir():
            if path.name == "generated_code.py":
                continue

            if path.is_symlink():
                raise SandboxExecutionError(f"Symbolic links are not accepted as artifacts: {path.name}")

            if not path.is_file():
                continue

            if path.stat().st_size > self.max_artifact_bytes:
                raise SandboxExecutionError(f"Artifact exceeds maximum size: {path.name}")

            files.append(path)

        return files


    def _copy_generated_file(self, file_path: Path, destination_dir: Path) -> str:
        """
        Copy an artifact to the specify directory, enforcing the size limit.
        """
        destination = destination_dir / file_path.name
        staging_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(dir = destination_dir, prefix = ".agent_artifact_", delete = False) as staging_file:
                staging_path = Path(staging_file.name)
                copied_bytes = 0

                with file_path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        copied_bytes += len(chunk)

                        if copied_bytes > self.max_artifact_bytes:
                            raise SandboxExecutionError(f"Artifact exceeds maximum size during copy: {file_path.name}")

                        staging_file.write(chunk)

            staging_path.replace(destination)
            return str(destination)

        finally:
            if staging_path is not None:
                staging_path.unlink(missing_ok = True)


    def _save_generated_files(self, generated_files: list[Path]) -> list[str]:
        """
        Save all artifacts in a unique directory, publishing them only after successful copying.
        """
        if not generated_files:
            return []

        staging_directory = Path(tempfile.mkdtemp(prefix = ".agent_stage_", dir = self.output_dir))

        run_directory = self.output_dir / f"run_{uuid4().hex}"

        try:
            for file_path in generated_files:
                self._copy_generated_file(file_path, destination_dir = staging_directory)

            if run_directory.exists() or run_directory.is_symlink():
                raise SandboxExecutionError(f"Artifact directory already exists: {run_directory}")

            staging_directory.rename(run_directory)

            return [str(run_directory / file_path.name) for file_path in generated_files]

        finally:
            if staging_directory.exists():
                shutil.rmtree(staging_directory, ignore_errors = True)

    
    def build_execution_command(self, script_path: Path) -> list[str]:
        """
        Build the Python command, optionally applying process resource limits.
        """
        if self.cpu_time_limit_seconds is None and self.max_file_size_bytes is None:
            return [sys.executable, str(script_path)]

        launcher_path = Path(__file__).with_name("resource_launcher.py").resolve()

        return [
                sys.executable,
                str(launcher_path),
                str(self.cpu_time_limit_seconds or 0),
                str(self.max_file_size_bytes or 0),
                str(script_path),
                ]


