from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

from src.llm.base import BaseLLMClient, ChatMessage
from src.llm.errors import LLMConnectionError, LLMServiceError, LLMTimeoutError


# JSON schema that constrains the model output.
# Ollama supports structured output through the "format" field.
AGENT_ACTION_JSON_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Brief explanation of the next step.",
                },
            "action": {
                "type": "string",
                "enum": ["execute_python", "final_answer"],
                "description": "The next action selected by the agent.",
                },
            "code": {
                "type": ["string", "null"],
                "description": (
                    "Python code to execute when action is execute_python. "
                    "Use null when action is final_answer."
                    ),
                },
            "final_answer": {
                "type": ["string", "null"],
                "description": (
                    "Final answer to the user when action is final_answer. "
                    "Use null when action is execute_python."
                    ),
                },
            },
        "required": ["thought", "action", "code", "final_answer"],
        "additionalProperties": False,
        }


class OllamaClient(BaseLLMClient):
    """
    Ollama-backed implementation of BaseLLMClient.

    This client talks to a local Ollama server and asks the model to return an AgentAction-compatible JSON object.

    It is useful when the project should run with a real local LLM and without external API keys.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout_seconds: int = 300, temperature: float = 0.0) -> None:
        """
        Initialize the Ollama client.

        Parameters
        ----------
        model:
            Name of the Ollama model to use.
            If None, read OLLAMA_MODEL from .env or use qwen2.5-coder:3b.

        base_url:
            Base URL of the Ollama server.
            If None, read OLLAMA_BASE_URL from .env or use localhost.

        timeout_seconds:
            Maximum time allowed for the HTTP request.

        temperature:
            Sampling temperature.
            0.0 makes outputs more deterministic.
        """

        load_dotenv()
        
        # Select the Ollama model.
        # Priority: 1. explicit argument; 2. OLLAMA_MODEL environment variable; 3. default qwen2.5-coder:3b
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
        
        # Select the Ollama server URL.
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        
        self.timeout_seconds = timeout_seconds
        
        self.temperature = temperature

    
    def generate(self, messages: list[ChatMessage]) -> str:
        """
        Generate a response using Ollama's local chat API.

        The response is expected to be JSON text matching AGENT_ACTION_JSON_SCHEMA.
        """
        
        # Build request payload for Ollama /api/chat.
        payload = {
                "model": self.model,
                "messages": self._to_ollama_messages(messages),
                "stream": False,
                "format": AGENT_ACTION_JSON_SCHEMA,
                "options": {
                    "temperature": self.temperature,
                    },
                }

        # Send the request to the local Ollama server.
        try:
            response = requests.post(f"{self.base_url}/api/chat", json = payload, timeout = self.timeout_seconds)
        
        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(f"Ollama request timed out after {self.timeout_seconds} seconds.") from exc
        
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(f"Cannot connect to the Ollama server at {self.base_url}.") from exc
        
        except requests.exceptions.RequestException as exc:
            raise LLMServiceError(f"Ollama HTTP request failed: {exc}") from exc

        # Check the HTTP status before interpreting the response body.
        try:
            response.raise_for_status()
        
        except requests.exceptions.HTTPError as exc:
            raise LLMServiceError(f"Ollama returned HTTP {response.status_code}: {response.text[:500]}") from exc

        # Validate the server response, independently of AgentAction.from_json().
        try:
            data = response.json()

        except ValueError as exc:
            raise LLMServiceError("Ollama returned an invalid JSON HTTP response.") from exc

        message = data.get("message") if isinstance(data, dict) else None
        
        content = message.get("content") if isinstance(message, dict) else None
        
        # Validate that the model returned non-empty text.
        if not isinstance(content, str) or not content.strip():
            raise LLMServiceError("Ollama returned missing or empty 'message.content'.")

        return content
        

    def _to_ollama_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """
        Convert provider-independent ChatMessage objects to Ollama messages.

        Ollama accepts roles such as system, user and assistant.
        
        Ollama expects each message to be a dictionary with:
        - role
        - content
        """

        return [{
            "role": message.role,
            "content": message.content,
            } for message in messages]
