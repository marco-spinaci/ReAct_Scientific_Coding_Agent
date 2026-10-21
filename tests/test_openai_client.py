from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("openai")
pytest.importorskip("dotenv")

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

from src.llm.base import ChatMessage
from src.llm.errors import LLMConnectionError, LLMServiceError, LLMTimeoutError
from src.llm.openai_client import OpenAIClient


def make_openai_client(monkeypatch: pytest.MonkeyPatch, response = None, error: Exception | None = None):

    create_mock = Mock(side_effect = error) if error is not None else Mock(return_value = response)

    fake_sdk_client = SimpleNamespace(responses = SimpleNamespace(create = create_mock))
    client_factory = Mock(return_value = fake_sdk_client)

    monkeypatch.setattr("src.llm.openai_client.OpenAI", client_factory)

    client = OpenAIClient(model = "test-model", api_key = "test-key")

    return client, create_mock, client_factory


def make_api_status_error(status_code: int) -> APIStatusError:

    response = Mock()
    response.request = Mock()
    response.status_code = status_code
    response.headers = {}

    return APIStatusError("HTTP request failed", response = response, body = None)


@pytest.mark.parametrize("sdk_error, expected_error", [
    (APITimeoutError(request = Mock()), LLMTimeoutError),
    (APIConnectionError(request = Mock()), LLMConnectionError),
    (make_api_status_error(401), LLMServiceError),
    (make_api_status_error(429), LLMServiceError),
    (make_api_status_error(503), LLMServiceError),
    (APIError("Unexpected API error", request = Mock(), body = None), LLMServiceError),
    ])
def test_openai_client_converts_sdk_errors(
        monkeypatch: pytest.MonkeyPatch, sdk_error: Exception, expected_error: type[Exception]
        ) -> None:

    client, create_mock, _ = make_openai_client(monkeypatch, error = sdk_error)

    with pytest.raises(expected_error) as exc_info:
        client.generate([ChatMessage(role = "user", content = "Test")])

    assert exc_info.value.__cause__ is sdk_error
    create_mock.assert_called_once()


@pytest.mark.parametrize("content", [
    None,
    "",
    "   ",
    123,
    ])
def test_openai_client_rejects_invalid_output_text(monkeypatch: pytest.MonkeyPatch, content) -> None:

    response = SimpleNamespace(output_text = content)

    client, _, _ = make_openai_client(monkeypatch, response = response)

    with pytest.raises(LLMServiceError, match = "missing or empty output_text"):
        client.generate([ChatMessage(role = "user", content = "Test")])


def test_openai_client_returns_valid_output(monkeypatch: pytest.MonkeyPatch) -> None:

    response = SimpleNamespace(output_text = '{"action": "final_answer"}')

    client, create_mock, client_factory = make_openai_client(monkeypatch, response = response)

    messages = [
            ChatMessage(role = "system", content = "Return JSON."),
            ChatMessage(role = "user", content = "Test"),
            ]

    result = client.generate(messages)

    assert result == response.output_text

    client_factory.assert_called_once_with(api_key = "test-key", max_retries = 0)

    create_mock.assert_called_once_with(
            model = "test-model",
            instructions = "Return JSON.",
            input = [{"role": "user", "content": "Test"}],
            )
