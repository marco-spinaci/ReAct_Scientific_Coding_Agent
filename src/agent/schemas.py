
from __future__ import annotations

import json
import re

from dataclasses import dataclass

# Literal restricts a type to specific allowed string values.
from typing import Any, Literal


AgentActionType = Literal["execute_python", "final_answer"]


def extract_json_payload(raw_response: str) -> dict[str, Any]:
    """
    Parse a JSON object from a raw LLM response or a single Markdown code block.
    """

    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("LLM response must be non-empty text.")

    response_text = raw_response.strip()

    try:
        # Convert the raw JSON string into a Python dictionary.
        payload = json.loads(response_text)

    except json.JSONDecodeError as exc:
        code_blocks = re.findall(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", response_text, flags = re.IGNORECASE | re.DOTALL)

        if len(code_blocks) != 1:
            raise ValueError("LLM response must be a JSON object or contain exactly one Markdown JSON code block.") from exc

        try:
            payload = json.loads(code_blocks[0])

        except json.JSONDecodeError as block_exc:
            raise ValueError(f"LLM response contains invalid JSON: {block_exc}") from block_exc

    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object, not a list or scalar value.")

    return payload


@dataclass
class AgentAction:
    """
    Structured action returned by the LLM.

    The agent expects the LLM to return a JSON object with:
    - thought: short explanation of the next step
    - action: either execute_python or final_answer
    - code: Python code when action is execute_python
    - final_answer: final response when action is final_answer
    """

    thought: str
    action: AgentActionType
    code: str | None = None
    final_answer: str | None = None

    @classmethod
    def from_json(cls, raw_response: str) -> "AgentAction":
        """
        Parse and validate a raw JSON response from the LLM.

        Parameters
        ----------
        raw_response:
            Raw string returned by the LLM.

        Returns
        -------
        AgentAction
            Validated structured action.

        Raises
        ------
        ValueError
            If the response is not valid JSON or does not follow the expected schema.
        """

        payload = extract_json_payload(raw_response)
        
        action = payload.get("action")
        if not isinstance(action, str) or action not in {"execute_python", "final_answer"}:
            raise ValueError(
                    "Invalid action. Expected 'execute_python' or 'final_answer', "
                    f"got: {action}")


        thought = payload.get("thought")
        # Accept only text that contains at least one non-whitespace character;
        # the same applies to the other “if” statements.
        if not isinstance(thought, str) or not thought.strip():
            raise ValueError("Missing or invalid 'thought' field.")

        code = payload.get("code")
        
        final_answer = payload.get("final_answer")
        
        if code is not None and not isinstance(code, str):
            raise ValueError("Invalid 'code' field: expected a string or null.")

        if final_answer is not None and not isinstance(final_answer, str):
            raise ValueError("Invalid 'final_answer' field: expected a string or null.")
        
        if action == "execute_python":
            if not isinstance(code, str) or not code.strip():
                raise ValueError("Action 'execute_python' requires non-empty code.")
        
        if action == "final_answer":
            if not isinstance(final_answer, str) or not final_answer.strip():
                raise ValueError("Action 'final_answer' requires a final answer.")
        
        return cls(
                thought = thought,
                action = action,
                code = code,
                final_answer = final_answer)


@dataclass
class AgentRunResult:
    """
    Final result returned by the ReActCodingAgent.
    """

    success: bool
    final_answer: str
    generated_files: list[str]
    iterations: int

