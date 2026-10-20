
import json

from src.llm.base import ChatMessage
from src.llm.fake_client import FakeLLMClient


def test_fake_llm_success_returns_execute_python_action() -> None:
    
    llm = FakeLLMClient(mode="success")

    response = llm.generate([
        ChatMessage(
            role = "user",
            content = "Generate a synthetic network plot.")])

    parsed = json.loads(response)

    assert parsed["action"] == "execute_python"
    assert parsed["code"] is not None
    assert "synthetic_network_degree_distribution.html" in parsed["code"]
    assert parsed["final_answer"] is None


def test_fake_llm_broken_then_success() -> None:
    
    llm = FakeLLMClient(mode="broken_then_success")

    first_response = json.loads(
            llm.generate([
                ChatMessage(
                    role = "user",
                    content = "Generate a plot.")]))

    second_response = json.loads(
            llm.generate([
                ChatMessage(
                    role = "user",
                    content = "Fix the previous error.")]))

    assert first_response["action"] == "execute_python"
    assert "missing_column" in first_response["code"]

    assert second_response["action"] == "execute_python"
    assert "synthetic_network_degree_distribution.html" in second_response["code"]


def test_fake_llm_final_only() -> None:

    llm = FakeLLMClient(mode="final_only")

    response = json.loads(
            llm.generate([
                ChatMessage(
                    role = "user",
                    content = "Answer directly.")]))

    assert response["action"] == "final_answer"
    assert response["code"] is None
    assert response["final_answer"] is not None

