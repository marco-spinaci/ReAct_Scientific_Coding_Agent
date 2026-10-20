
from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from src.llm.base import BaseLLMClient, ChatMessage


class OpenAIClient(BaseLLMClient):
    """
    OpenAI-backed implementation of BaseLLMClient.

    This client uses the OpenAI Responses API and returns the model output as plain text. 

    In this project, the agent expects that text to be a JSON object conforming to AgentAction.
    """


    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        
        load_dotenv()

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.2")
        
        self.client = OpenAI(api_key = api_key or os.getenv("OPENAI_API_KEY"))


    def generate(self, messages: list[ChatMessage]) -> str:
        """
        Generate a model response from chat-style messages.

        The Responses API accepts a list of input messages. 
        
        We map our provider-independent ChatMessage objects into the OpenAI format.
        """
        
        instructions = "\n\n".join(message.content for message in messages if message.role == "system")

        input_messages = [{"role": message.role, "content": message.content} for message in messages if message.role != "system"]

        response = self.client.responses.create(model = self.model, instructions = instructions or None, input = input_messages)

        return response.output_text

