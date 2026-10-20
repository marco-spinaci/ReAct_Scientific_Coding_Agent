
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """
    Minimal chat message representation.

    We keep this class provider-independent so that the agent can work with
    different LLM backends, such as OpenAI, Ollama, or a fake deterministic
    client used for tests.
    """

    role: str 
    content: str


class BaseLLMClient(ABC):
    """Abstract interface for language model clients."""

    @abstractmethod
    def generate(self, messages: list[ChatMessage]) -> str:
        """
        Generate a response from a list of chat messages.

        Parameters
        ----------
        messages:
            Conversation history represented as role/content messages.

        Returns
        -------
        str
            Raw text returned by the language model.
        """
        
        raise NotImplementedError

