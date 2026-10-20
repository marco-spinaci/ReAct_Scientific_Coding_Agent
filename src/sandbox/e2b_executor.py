
from __future__ import annotations

import logging
import os
import time

from typing import Any
from pathlib import Path
from dotenv import load_dotenv

from src.sandbox.base import BaseSandboxExecutor, SandboxResult
from src.sandbox.errors import SandboxPolicyError
from src.sandbox.policy import SandboxPolicy


logger = logging.getLogger(__name__)


class E2BSandboxExecutor(BaseSandboxExecutor):
    """
    E2B-backed sandbox executor.
    
    This backend uses E2B Code Interpreter to execute AI-generated Python code
    inside an isolated cloud sandbox.
    
    This class has the same public interface as LocalSandboxExecutor:
        run(code: str) -> SandboxResult

    Notes
    -----
    - This backend is optional.
    - It requires the e2b-code-interpreter package.
    - It requires an E2B_API_KEY environment variable.
    - The local SandboxPolicy is still applied before sending code to E2B.
    - File artifact retrieval is intentionally conservative in this first
      implementation; stdout/text output is supported, and generated file
      handling can be refined depending on the E2B SDK version in use.
    """

    DEFAULT_REMOTE_WORKDIR = "/home/user"

    def __init__(self, timeout_seconds: int = 10, policy: SandboxPolicy | None = None, output_dir: str | Path = "outputs", remote_workdir: str = DEFAULT_REMOTE_WORKDIR, ) -> None:
        """
        Initialize the E2B sandbox executor.

        Parameters
        ----------
        timeout_seconds:
            Maximum execution time allowed inside the E2B sandbox.

        policy:
            Optional local SandboxPolicy.
            If None, a default policy is created.

        output_dir:
            Local directory where downloaded artifacts could be saved.

        remote_workdir:
            Remote directory inside the E2B sandbox where the html files are stored
        """

        load_dotenv()
        
        self.timeout_seconds = timeout_seconds
        
        self.policy = policy or SandboxPolicy()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents = True, exist_ok = True)

        self.remote_workdir = remote_workdir.rstrip("/")

        if not os.getenv("E2B_API_KEY"):
            raise RuntimeError(
                    "E2B_API_KEY is not set. Add it to your environment or .env file "
                    "to use the E2B sandbox backend."
                    )

    
    def run(self, code: str) -> SandboxResult:
        """
        Validate and execute Python code inside an E2B sandbox.
        """

        start_time = time.perf_counter()
        logger.info("Starting E2B sandbox execution.")

        
        try:
            self.policy.validate(code)
        except SandboxPolicyError as exc:
            logger.warning("Sandbox policy violation before E2B execution: %s", exc)

            return SandboxResult(
                    success = False,
                    stdout = "",
                    stderr = "",
                    return_code = None,
                    generated_files = [],
                    execution_time = time.perf_counter() - start_time,
                    error_message=str(exc),
                    )
        
        
        # Import E2B; This keeps the dependency optional unless this backend is used.
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError as exc:
            raise RuntimeError(
                    "The E2B backend requires the optional dependency "
                    "'e2b-code-interpreter'. Install it with:\n\n"
                    "    pip install e2b-code-interpreter\n"
                    ) from exc

        
        # Create an E2B sandbox.
        # The with-block ensures resources are cleaned up afterwards.
        try:
            with Sandbox.create(timeout = self.timeout_seconds) as sandbox:
                
                prepared_code = self._wrap_code_in_remote_workdir(code)

                execution = sandbox.run_code(prepared_code)

                stdout = self._extract_stdout(execution)
                stderr = self._extract_stderr(execution)
                error_message = self._extract_error_message(execution)
                
                generated_files: list[str] = []
                if error_message is None:
                    generated_files.extend(self._download_html_artifacts(sandbox = sandbox))

                execution_time = time.perf_counter() - start_time
                
                success = error_message is None

                logger.info(
                        "E2B sandbox execution finished. success=%s "
                        "generated_files=%d execution_time=%.3fs",
                        success, len(generated_files), execution_time,
                        )
                
                # Convert E2B execution output into the common SandboxResult format.
                # E2B does not necessarily expose a normal subprocess return code; we map success/failure to 0/1.
                return SandboxResult(
                        success = success,
                        stdout = stdout,
                        stderr = stderr,
                        return_code = 0 if success else 1,
                        generated_files = generated_files,
                        execution_time = execution_time,
                        error_message = error_message,
                        )

        except Exception as exc:
            logger.exception("E2B sandbox execution failed unexpectedly.")
            return SandboxResult(
                    success = False,
                    stdout = "",
                    stderr = "",
                    return_code = None,
                    generated_files = [],
                    execution_time = time.perf_counter() - start_time,
                    error_message = f"E2B execution failed: {exc}",
                    )

    
    def _wrap_code_in_remote_workdir(self, code: str) -> str:
        """
        Ensure generated files are written under the expected remote workdir.

        The policy blocks direct os imports in user-generated code, 
        but this wrapper is controlled by us and executed inside the E2B sandbox.
        """

        return f"""
import os
os.makedirs({self.remote_workdir!r}, exist_ok = True)
os.chdir({self.remote_workdir!r})

{code}
""".strip()


    def _extract_stdout(self, execution: object) -> str:
        """
        Extract text output from an E2B execution object.

        E2B SDK versions may expose output through slightly different fields,
        so this method uses getattr defensively instead of assuming one schema.
        """

        # Some SDK versions expose a direct .text field.        
        text = getattr(execution, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        logs = getattr(execution, "logs", None)
        if logs is not None:
            stdout = getattr(logs, "stdout", None)
            if isinstance(stdout, list):
                return "\n".join(str(item) for item in stdout)
            if isinstance(stdout, str):
                return stdout

        return ""


    def _extract_stderr(self, execution: object) -> str:
        """
        Extract stderr text from an E2B execution object.
        """

        logs = getattr(execution, "logs", None)
        if logs is not None:
            stderr = getattr(logs, "stderr", None)
            if isinstance(stderr, list):
                return "\n".join(str(item) for item in stderr)
            if isinstance(stderr, str):
                return stderr

        return ""


    def _extract_error_message(self, execution: object) -> str | None:
        """
        Extract an error message from an E2B execution object.

        Returns None if no error is present.
        """

        error = getattr(execution, "error", None)
        if error is None:
            return None

        name = getattr(error, "name", None)
        value = getattr(error, "value", None)
        traceback = getattr(error, "traceback", None)
        
        # Build a readable error message from available pieces.
        parts = []
        if name:
            parts.append(str(name))
        if value:
            parts.append(str(value))
        if traceback:
            parts.append(str(traceback))

        return "\n".join(parts) if parts else str(error)


    def _download_html_artifacts(self, sandbox: object) -> list[str]:
        """
        Find and download generated HTML files from the E2B sandbox.

        The generated Python code is expected to save Plotly figures using fig.write_html("some_plot.html"). 
        This method searches the configured remote working directory for .html/.htm files and copies them 
        to self.output_dir.
        """
        
        downloaded_files: list[str] = []

        try:
            remote_files = self._list_remote_files(sandbox = sandbox, remote_dir = self.remote_workdir)
        except Exception as exc:
            logger.warning("Could not list E2B remote files: %s", exc)
            return downloaded_files

        
        html_paths = [remote_path for remote_path in remote_files if remote_path.lower().endswith((".html", ".htm"))]

        
        for remote_path in html_paths:

            local_path = self.output_dir / Path(remote_path).name

            try:
                content = self._read_remote_file(sandbox = sandbox, remote_path = remote_path)   
                self._write_local_file(local_path = local_path, content = content)

                downloaded_files.append(str(local_path))
                logger.info("Downloaded E2B HTML artifact: %s", local_path)
            except Exception as exc:
                logger.warning("Could not download E2B HTML artifact %s: %s", remote_path, exc)

        return downloaded_files


    def _list_remote_files(self, sandbox: object, remote_dir: str) -> list[str]:
        """
        List files in a remote E2B directory.

        Returns remote file paths as strings.
        """
        
        files_api = getattr(sandbox, "files", None)
        if files_api is None:
            return []

        raw_items = files_api.list(remote_dir)

        paths: list[str] = []
        for item in raw_items:
            path = self._extract_remote_path(item = item, remote_dir = remote_dir)
            if path is not None:
                paths.append(path)

        return paths


    def _extract_remote_path(self, item: Any, remote_dir: str) -> str | None:
        """
        Extract a remote path from E2B file metadata.

        The E2B SDK may return strings, dictionaries, or objects with path/name attributes depending on the version.
        
        Normalize different possible SDK return formats into one path string.
        """

        if isinstance(item, str):
            if item.startswith("/"):
                return item
            return f"{remote_dir}/{item}"

        if isinstance(item, dict):
            raw_path = item.get("path") or item.get("name")
            if raw_path is None:
                return None

            raw_path = str(raw_path)
            if raw_path.startswith("/"):
                return raw_path

            return f"{remote_dir}/{raw_path}"

        raw_path = getattr(item, "path", None) or getattr(item, "name", None)
        if raw_path is None:
            return None

        raw_path = str(raw_path)
        if raw_path.startswith("/"):
            return raw_path

        return f"{remote_dir}/{raw_path}"


    def _read_remote_file(self, sandbox: object, remote_path: str) -> bytes | str:
        """
        Read or download a file from the E2B sandbox.

        Different E2B SDK versions may expose either files.read(...)
        or files.download(...), so we support both.
        """

        files_api = getattr(sandbox, "files", None)
        
        if files_api is None:
            raise RuntimeError("E2B sandbox does not expose a files API.")

        if hasattr(files_api, "read"):
            return files_api.read(remote_path)

        if hasattr(files_api, "download"):
            return files_api.download(remote_path)

        raise RuntimeError("E2B files API exposes neither read() nor download().")

    
    def _write_local_file(self, local_path: Path, content: bytes | str) -> None:
        """
        Write downloaded E2B file content to the local output directory.
        """

        local_path.parent.mkdir(parents = True, exist_ok = True)

        if isinstance(content, bytes):
            local_path.write_bytes(content)
        else:
            local_path.write_text(str(content), encoding = "utf-8")

    
