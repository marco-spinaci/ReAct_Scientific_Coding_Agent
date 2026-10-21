import json

import pytest

from src.agent.schemas import AgentAction


VALID_ACTION = {
        "thought": "Create a plot",
        "action": "execute_python",
        "code": "print('Hello')",
        "final_answer": None,
        }


@pytest.mark.parametrize("response", [
    json.dumps(VALID_ACTION),
    f"```json\n{json.dumps(VALID_ACTION)}\n```",
    f"Here is the action:\n```json\n{json.dumps(VALID_ACTION)}\n```",
    f"```\n{json.dumps(VALID_ACTION)}\n```",
    ])

def test_agent_action_parses_valid_json_formats(response: str) -> None:

    result = AgentAction.from_json(response)

    assert result.action == "execute_python"
    assert result.code == "print('Hello')"


@pytest.mark.parametrize("response", [
    "not JSON",
    "[]",
    '"just text"',
    json.dumps({"thought": "hi", "action": [], "code": "print(1)"}),
    json.dumps({"thought": "hi", "action": "execute_python", "code": None}),
    json.dumps({"thought": "hi", "action": "final_answer", "final_answer": 42}),
    "```json\n{not-valid-json}\n```",
    "",
    ])

def test_agent_action_rejects_invalid_json_or_schema(response: str) -> None:

    with pytest.raises(ValueError):
        AgentAction.from_json(response)


