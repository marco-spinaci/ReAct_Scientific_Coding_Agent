
from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI

from src.llm.base import BaseLLMClient, ChatMessage
from src.llm.errors import LLMConnectionError, LLMServiceError, LLMTimeoutError


class OpenAIClient(BaseLLMClient):
    """
    OpenAI-backed implementation of BaseLLMClient.

    This client uses the OpenAI Responses API and returns the model output as plain text. 

    In this project, the agent expects that text to be a JSON object conforming to AgentAction.
    """


    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        load_dotenv()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.2")
        self.client = OpenAI(api_key = api_key or os.getenv("OPENAI_API_KEY"), max_retries = 0)


    def generate(self, messages: list[ChatMessage]) -> str:
        """
        Generate a model response from chat-style messages.

        The Responses API accepts a list of input messages. 
        
        We map our provider-independent ChatMessage objects into the OpenAI format.
        """
        instructions = "\n\n".join(message.content for message in messages if message.role == "system")

        input_messages = [{"role": message.role, "content": message.content} for message in messages if message.role != "system"]
        
        try:
            response = self.client.responses.create(model = self.model, instructions = instructions or None, input = input_messages)

        except APITimeoutError as exc:
            raise LLMTimeoutError(f"OpenAI request timed out: {exc}") from exc

        except APIConnectionError as exc:
            raise LLMConnectionError(f"Cannot connect to the OpenAI API: {exc}") from exc

        except APIStatusError as exc:
            raise LLMServiceError(f"OpenAI returned HTTP {exc.status_code}: {exc}") from exc

        except APIError as exc:
            raise LLMServiceError(f"OpenAI API request failed: {exc}") from exc

        content = response.output_text

        if not isinstance(content, str) or not content.strip():
            raise LLMServiceError("OpenAI returned missing or empty output_text.")

        return content


