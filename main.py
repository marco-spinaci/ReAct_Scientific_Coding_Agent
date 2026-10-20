#!/usr/bin/env python3

from __future__ import annotations

import argparse

from src.utils.logging_config import configure_logging
from src.sandbox.local_executor import LocalSandboxExecutor
from src.tools.python_tool import PythonExecutionTool
from src.llm.fake_client import FakeLLMClient
from src.agent.react_agent import ReActCodingAgent
from src.memory.json_store import JsonlRunStore
from src.domain.synthetic_network_prompt import get_synthetic_network_request


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description = "Run the ReAct-style coding agent.")
    
    parser.add_argument("--sandbox",
            choices=["local", "e2b"],
            default="local",
            help="Sandbox backend to use.")

    parser.add_argument("--llm", 
            choices = ["fake", "openai", "ollama"], 
            default = "fake", 
            help = "LLM backend to use.")

    parser.add_argument("--fake-mode", 
            choices = ["success", "broken_then_success", "final_only"], 
            default = "success",
            help = "Fake LLM behavior mode.")
    
    parser.add_argument("--ollama-model",
            default = None,
            help = "Ollama model to use, e.g. qwen2.5-coder:3b.")

    parser.add_argument("--ollama-base-url",
            default = None,
            help = "Ollama base URL, e.g. http://localhost:11434.")
    
    parser.add_argument("--output-dir",
            default = "outputs",
            help = "Directory where generated files are copied.")

    parser.add_argument("--timeout",
            type = int,
            default = 10,
            help = "Sandbox execution timeout in seconds.")

    parser.add_argument("--max-iterations",
            type = int,
            default = 2,
            help = "Maximum number of ReAct iterations.")

    parser.add_argument("--request",
            default = get_synthetic_network_request(),
            help = "User request passed to the agent.")

    return parser


def main() -> None:
    
    configure_logging()
    
    args = build_parser().parse_args()
    
    
    
    if args.sandbox == "local":
        sandbox = LocalSandboxExecutor(timeout_seconds = args.timeout, output_dir=args.output_dir)
    elif args.sandbox == "e2b":
        try:
            from src.sandbox.e2b_executor import E2BSandboxExecutor
        except ImportError as exc:
            raise RuntimeError(
                    "The E2B sandbox backend requires optional dependencies. "
                    "Install them with:\n\n"
                    "    pip install -r requirements-e2b.txt\n") from exc

        sandbox = E2BSandboxExecutor(timeout_seconds = args.timeout, output_dir=args.output_dir)
    else:
        raise ValueError(f"Unsupported sandbox backend: {args.sandbox}")

    

    python_tool = PythonExecutionTool(sandbox = sandbox)

    

    if args.llm == "fake":
        llm = FakeLLMClient(mode=args.fake_mode)
    elif args.llm == "openai":
        try:
            from src.llm.openai_client import OpenAIClient
        except ImportError as exc:
            raise RuntimeError(
                    "The OpenAI backend requires the optional dependencies "
                    "'openai' and 'python-dotenv'. Install them with:\n\n"
                    "    pip install openai python-dotenv\n") from exc

        llm = OpenAIClient()
    elif args.llm == "ollama":
        try:
            from src.llm.ollama_client import OllamaClient
        except ImportError as exc:
            raise RuntimeError(
                    "The Ollama backend requires optional dependencies. "
                    "Install them with:\n\n"
                    "    pip install -r requirements-ollama.txt\n") from exc

        llm = OllamaClient(model = args.ollama_model, base_url = args.ollama_base_url)
    else:
        raise ValueError(f"Unsupported LLM backend: {args.llm}")



    agent = ReActCodingAgent(llm = llm, python_tool = python_tool, max_iterations = args.max_iterations)

    result = agent.run(args.request)

    run_store = JsonlRunStore()
    run_store.append_run({
        "llm_backend": args.llm,
        "sandbox_backend": args.sandbox,
        "fake_mode": args.fake_mode if args.llm == "fake" else None,
        "ollama_model": args.ollama_model if args.llm == "ollama" else None,
        "user_request": args.request,
        "success": result.success,
        "iterations": result.iterations,
        "generated_files": result.generated_files,
        })

    print("Success:", result.success)
    print("Iterations:", result.iterations)
    print("Generated files:", result.generated_files)
    print()
    print(result.final_answer)


if __name__ == "__main__":
    main()

