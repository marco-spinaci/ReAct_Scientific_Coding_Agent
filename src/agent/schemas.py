
from __future__ import annotations

import json

from dataclasses import dataclass

# Literal restricts a type to specific allowed string values.
from typing import Literal


AgentActionType = Literal["execute_python", "final_answer"]


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
            If the response is not valid JSON or does not follow
            the expected schema.
        """

        try:
            # Convert the raw JSON string into a Python dictionary.
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response is not valid JSON: {exc}") from exc

        
        action = payload.get("action")
        if action not in {"execute_python", "final_answer"}:
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

