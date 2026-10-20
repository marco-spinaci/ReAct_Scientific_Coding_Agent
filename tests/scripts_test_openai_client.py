
from src.llm.base import ChatMessage
from src.llm.openai_client import OpenAIClient


def main() -> None:
    llm = OpenAIClient()

    response = llm.generate([
        ChatMessage(
            role = "user",
            content = (
                "Return only a JSON object with keys "
                "'thought', 'action', 'code', 'final_answer'. "
                "Use action='final_answer'."),
            )
        ])

    print(response)


if __name__ == "__main__":
    main()

