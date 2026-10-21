import pytest

requests = pytest.importorskip("requests")
pytest.importorskip("dotenv")

from src.llm.base import ChatMessage
from src.llm.errors import LLMConnectionError, LLMServiceError, LLMTimeoutError
from src.llm.ollama_client import OllamaClient


def make_response(status_code: int, body: bytes) -> requests.Response:
    """Build an HTTP response for testing."""
    response = requests.Response()
    response.status_code = status_code
    response._content = body

    return response


@pytest.mark.parametrize("request_error, expected_error", [
    (requests.exceptions.ReadTimeout("read timeout"), LLMTimeoutError),
    (requests.exceptions.ConnectTimeout("connect timeout"), LLMTimeoutError),
    (requests.exceptions.ConnectionError("connection refused"), LLMConnectionError),
    (requests.exceptions.RequestException("transport error"), LLMServiceError),
    ])
def test_ollama_client_converts_request_errors(
        monkeypatch: pytest.MonkeyPatch, request_error: Exception, expected_error: type[Exception]
        ) -> None:
    def failing_post(*args, **kwargs):
        raise request_error

    monkeypatch.setattr("src.llm.ollama_client.requests.post", failing_post)

    client = OllamaClient(model = "test-model")

    with pytest.raises(expected_error):
        client.generate([ChatMessage(role = "user", content = "Test")])


@pytest.mark.parametrize("status_code", [400, 503])
def test_ollama_client_converts_http_errors(monkeypatch: pytest.MonkeyPatch, status_code: int) -> None:
    def fake_post(*args, **kwargs):
        return make_response(status_code, b'{"error": "failed"}')

    monkeypatch.setattr("src.llm.ollama_client.requests.post", fake_post)

    client = OllamaClient(model = "test-model")

    with pytest.raises(LLMServiceError, match = str(status_code)):
        client.generate([ChatMessage(role = "user", content = "Test")])


@pytest.mark.parametrize("body", [
    b"not json",
    b"[]",
    b"{}",
    b'{"message": []}',
    b'{"message": {"content": ""}}',
    ])
def test_ollama_client_rejects_invalid_response(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    def fake_post(*args, **kwargs):
        return make_response(200, body)

    monkeypatch.setattr("src.llm.ollama_client.requests.post", fake_post)

    client = OllamaClient(model = "test-model")

    with pytest.raises(LLMServiceError):
        client.generate([ChatMessage(role = "user", content = "Test")])


def test_ollama_client_returns_message_content(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url, json, timeout):
        assert url == "http://localhost:11434/api/chat"
        assert json["format"]["type"] == "object"
        assert timeout == 300

        return make_response(200, b'{"message": {"content": "hello"}}')

    monkeypatch.setattr("src.llm.ollama_client.requests.post", fake_post)

    client = OllamaClient(model = "test-model", base_url = "http://localhost:11434")

    result = client.generate([ChatMessage(role = "user", content = "Test")])

    assert result == "hello"
