
class SandboxError(Exception):
    """Base exception for sandbox-related errors."""

class SandboxPolicyError(SandboxError):
    """Raised when generated code violates the sandbox policy."""

class SandboxExecutionError(SandboxError):
    """Raised when sandbox execution fails unexpectedly."""
