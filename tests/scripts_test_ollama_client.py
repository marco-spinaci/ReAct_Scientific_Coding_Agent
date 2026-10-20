
from src.llm.base import ChatMessage
from src.llm.ollama_client import OllamaClient


def main() -> None:
    llm = OllamaClient(model="qwen2.5-coder:3b")

    response = llm.generate([
        ChatMessage(
            role="system",
            content=(
                "You are a coding agent. Return only valid JSON matching "
                "the required schema."),
            ),
        ChatMessage(
            role="user",
            content=(
                "Return a final_answer action saying that the local "
                "Ollama backend works."
                ),
            ),
        ])

    print(response)


if __name__ == "__main__":
    main()

