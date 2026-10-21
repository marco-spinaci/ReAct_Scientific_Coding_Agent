class LLMClientError(RuntimeError):
    """Base exception for errors raised by LLM backends."""


class LLMTimeoutError(LLMClientError):
    """Raised when an LLM request exceeds its timeout."""


class LLMConnectionError(LLMClientError):
    """Raised when the LLM backend cannot be reached."""


class LLMServiceError(LLMClientError):
    """Raised when the LLM service returns an error."""

