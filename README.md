# ReAct Scientific Coding Agent

**A modular research prototype for LLM-assisted Python analysis and interactive visualization.**

This project implements a lightweight **ReAct-style coding agent** that translates natural-language requests into Python code, executes that code through a configurable execution backend, and returns execution feedback and generated artifacts.

The implementation emphasizes readable components, explicit interfaces, and reproducible development workflows rather than a large agent framework. 
At the moment, it is intended for experimentation and local development, **not for production use with untrusted generated code**.

## Research motivation

The project is motivated in part by research on **economic network reconstruction and link prediction**, in particular directed supplier–customer networks. 
Such research involves recurring computational tasks: inspecting network structure, evaluating models, checking experiments, and preparing figures for scientific reporting.

The current demonstration generates a **synthetic directed supplier–customer network**, computes in-degree and out-degree, and produces an interactive Plotly visualization. 
It provides a starting point for investigating how an AI coding agent might assist with scientific workflows.
No real-world or proprietary supply-chain dataset is included.

## Current capabilities

- Accept a natural-language request through a command-line interface.
- Query an LLM backend for a structured action (`execute_python` or `final_answer`).
- Execute generated Python scripts with a configurable execution backend.
- Capture execution output, errors, elapsed time, and generated files.
- Retry generated code after a failed execution, up to a configured iteration limit.
- Produce interactive Plotly visualizations as HTML artifacts.
- Record application logs and lightweight run metadata in JSONL format.
- Run deterministic tests without external LLM services through `FakeLLMClient`.

**Available backends**

| Component | Implementations | Notes |
| --- | --- | --- |
| LLM | `FakeLLMClient` | Backend used for deterministic testing and local development. |
| LLM | `OllamaClient` | Uses a local Ollama server and a downloaded model. |
| LLM | `OpenAIClient` | Requires API credentials and separate API billing. |
| Execution | `LocalSandboxExecutor` | Local subprocess in a temporary working directory; **not a security boundary**. |
| Execution | `E2BSandboxExecutor` | Optional remote execution backend; requires an E2B account/key. |

The E2B integration has not been validated by the project's automated integration tests. 
Actual behavior depends on the installed SDK version and remote environment.

## Quick start: deterministic local demo

The default demo does not require an API key, an Ollama installation, or network access after Python dependencies are installed.
The original implementation was developed and tested on Ubuntu 22.04 with Python 3.10.

**Requirements:** Python 3.10 or newer. 

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the default supplier–customer network example:

```bash
python3 main.py --llm fake --sandbox local
```

The run should create `outputs/synthetic_network_degree_distribution.html` and print a short summary. 
Open the HTML file in a browser to explore the plot.

Test the execution-recovery path:

```bash
python3 main.py --llm fake --fake-mode broken_then_success --sandbox local
```

In this mode, the fake client deliberately generates failing code on the first attempt. 
The agent passes the execution error back to the client and tries a corrected action on the next iteration.

> **What this test proves:** the local tool/execution path and deterministic retry flow work together. 
It does **not** establish the reliability of a real LLM, the correctness of arbitrary generated analyses, or secure code isolation.

## Running with a local LLM: Ollama

An Ollama backend is available for experiments with a local coding model. 
A lightweight example is `qwen2.5-coder:3b`.

Install Ollama separately, then download the model:

```bash
ollama pull qwen2.5-coder:3b
```

Install the additional Python dependencies:

```bash
pip install -r requirements-ollama.txt
```

The Ollama server must be running and accessible (by default at `http://localhost:11434`). 
Run the default demo with a real local model:

```bash
python3 main.py --llm ollama --sandbox local --ollama-model qwen2.5-coder:3b
```
The Ollama backend uses structured outputs through a JSON schema to make the model return an AgentAction-compatible response.

### Minimal Plotly smoke test

Start with a short task to verify the end-to-end path without requiring the model to write an entire network-analysis script:

```bash
python3 main.py \
  --llm ollama \
  --sandbox local \
  --ollama-model qwen2.5-coder:3b \
  --request "Generate Python code that creates a simple Plotly line chart with x=[1,2,3] and y=[1,4,9], saves it as simple_plot.html, and prints a short summary."
```

When successful, this exercises:

```text
natural-language request
    |
local LLM -> structured JSON action
    |
PythonExecutionTool -> local executor
    |
Python execution -> HTML artifact -> agent result
```

The complete supplier–customer example is more demanding: it requires synthetic data generation, directed-edge construction, degree calculations, and plotting. 
Small models, especially on CPU-only machines, may generate code slowly, return invalid actions, or reach the HTTP read timeout.

**Timeout distinction:** `--timeout` controls *Python execution inside the selected sandbox*. 
The Ollama HTTP timeout is currently configured in `OllamaClient` (`timeout_seconds`, default 300 seconds); this version of the CLI does **not** expose an `--ollama-timeout` flag.

A successful smoke test confirms that one simple end-to-end request worked. It does not guarantee that more complex generated code is correct.

## Optional OpenAI backend

Install the optional dependencies:

```bash
pip install -r requirements-openai.txt
cp .env.example .env
```

Set your API key and model in `.env`:

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=your_supported_model
```

Then run:

```bash
python3 main.py --llm openai --sandbox local
```

OpenAI API usage requires separately configured API billing; a ChatGPT subscription does not by itself supply API credits. 
Do not commit `.env` to version control.

## Optional remote execution with E2B sandbox backend

Install the dependency:

```bash
pip install -r requirements-e2b.txt
```

Add `E2B_API_KEY` to `.env`, and select the backend:

```bash
python3 main.py --llm fake --sandbox e2b
```

The E2B adapter applies the local AST policy before remote execution and attempts to download `.html`/`.htm` files from the remote working directory. 
This is an **optional, not yet end-to-end verified integration**; successful operation depends on API access and SDK compatibility.

## Command-line options

The main options in `main.py` are:

| Option | Default | Purpose |
| --- | --- | --- |
| `--llm` | `fake` | `fake`, `ollama`, or `openai`. |
| `--sandbox` | `local` | `local` or `e2b`. |
| `--fake-mode` | `success` | `success`, `broken_then_success`, or `final_only`. |
| `--ollama-model` | from client configuration | Override Ollama's model name. |
| `--ollama-base-url` | from client configuration | Override the Ollama server address. |
| `--request` | synthetic-network prompt | Override the user request. |
| `--output-dir` | `outputs` | Directory used for collected generated files. |
| `--timeout` | `10` | Python execution timeout in seconds. |
| `--max-iterations` | `2` | Maximum number of agent iterations. |

More example requests are available in `examples/example_prompts.md`.

## Architecture

The design uses pragmatic module names while following a ports-and-adapters approach.

```text
./
├── main.py                              # Composition root and CLI
├── src/agent/
│   ├── react_agent.py                   # Main application use case
│   ├── prompts.py                       # LLM instructions and retry prompt
│   └── schemas.py                       # Agent action/result models
├── src/llm/
│   ├── base.py                          # BaseLLMClient port and ChatMessage
│   ├── fake_client.py                   # Deterministic adapter
│   ├── ollama_client.py                 # Ollama adapter
│   └── openai_client.py                 # OpenAI adapter
├── src/tools/
│   └── python_tool.py                   # Agent-facing Python tool
├── src/sandbox/
│   ├── base.py                          # BaseSandboxExecutor port and result
│   ├── policy.py                        # AST checks
│   ├── errors.py                        # Sandbox-related errors
│   ├── local_executor.py                # Local execution adapter
│   └── e2b_executor.py                  # Remote execution adapter
├── src/memory/
│   └── json_store.py                    # JSONL run metadata
├── src/domain/
│   └── synthetic_network_prompt.py      # Demonstration request
├── src/utils/
│   └── logging_config.py                # Logging setup
├── tests/                                # Automated and manual tests
├── examples/
│   └── example_prompts.md
├── requirements.txt
├── requirements-ollama.txt
├── requirements-openai.txt
├── requirements-e2b.txt
├── .env.example
└── .gitignore
```

Runtime files are written under `logs/` and `outputs/`; both directories are ignored by Git.
Files named `__init__.py` are omitted from the tree for readability.

### Ports, adapters, and application logic

- **Ports:** `BaseLLMClient` and `BaseSandboxExecutor` define provider-independent interfaces.
- **Adapters:** concrete LLM and execution classes implement those interfaces.
- **Application use case:** `ReActCodingAgent` coordinates model actions, Python execution, and execution-based retries.
- **Composition root:** `main.py` selects concrete implementations and wires dependencies together.

The agent depends on the abstract LLM contract and on `PythonExecutionTool`, rather than importing provider-specific LLM or sandbox clients directly. 
`PythonExecutionTool` itself depends on the abstract sandbox executor.

### Minimal ReAct-style loop

The workflow is deliberately simple:

1. Ask the LLM for a structured action.
2. Parse the action (`execute_python` or `final_answer`).
3. For `execute_python`, run the generated script and collect an observation.
4. If execution fails, send the previous code and error back for a retry.
5. Stop after an execution success, a `final_answer` action, or the maximum number of iterations.

The current implementation is **ReAct-inspired**, not a general planning agent: successful code execution ends the run immediately, and there is no independent assessment that the analysis fulfills the scientific request.

A typical LLM action looks like:

```json
{
  "thought": "Create a small directed network and visualize its degrees.",
  "action": "execute_python",
  "code": "print('Example')",
  "final_answer": null
}
```

The `thought` field is a short action rationale, not a guarantee of an interpretable reasoning trace.

## Tests

Install the base dependencies, then run:

```bash
pytest -q
```

The automated tests cover the AST policy, local execution, the Python tool, deterministic LLM behavior, and selected agent success/retry paths. 
Additional scripts in `tests/` allow manual checks of the OpenAI and Ollama clients.

The tests do **not** verify comprehensive sandbox isolation, all responses from real LLMs, remote E2B execution, or the scientific validity of generated analyses.

## Security and known limitations

**Warning: `LocalSandboxExecutor` is not a secure sandbox.** A temporary working directory, a separate subprocess, and AST-based policy checks do not restrict the Python process to that directory, remove its filesystem permissions, or enforce network isolation. 
Do not use the local executor to run untrusted code or process sensitive data without additional containment and review.

Other known limitations:

- The policy checks are incomplete and may be bypassed; they are not an OS-level security mechanism.
- Invalid or Markdown-wrapped LLM JSON can raise a parsing error without a controlled retry.
- Errors such as local-model HTTP timeouts are not yet handled as structured agent failures.
- The agent may report success even if requested HTML artifacts are absent or the analysis is incomplete.
- An LLM may return `final_answer` without executing a tool; this is not verification of the requested task.
- The local executor collects generated files but does not establish that those files are trustworthy or scientifically correct.
- E2B behavior has not been verified by automated integration tests.
- Dependencies are not yet pinned for reproducible environment reconstruction.

For less-trusted workloads, a future version should use stronger isolation, restricted permissions and networking, resource controls, and approval or review before executing generated code.

## Future improvements

Possible extensions:

- robust JSON extraction from Markdown-formatted LLM responses.
  This extension is particularly useful for real-world backends such as OpenAI, Ollama or other local models.

- Support for uploaded CSV files.

- Persistent memory with Neo4j or SQL.

- Migrate the agent flow to LangGraph.
  This project implements the ReAct loop manually inside `ReActCodingAgent` to keep the MVP lightweight, transparent, and easy to inspect.
  In a production setting, the same flow could be migrated to LangGraph to obtain explicit state management, conditional routing, checkpointing, 
  and human-in-the-loop approval before executing generated code.

- Stronger sandbox isolation.

- Artifact previews. 
  A more advanced version could automatically display a preview of the graph, a list of generated artifacts, and file metadata.

- Richer retry and recovery strategies.
  Now, retrying is very simple. In a more advanced version, we could distinguish between syntax error, import error, policy error, runtime error, HTML file not generated, and user request not fulfilled. 
 
- Richer CLI using Typer or Rich.
  A more elegant CLI could use Typer for structured commands, Rich for coloured output, tables and panels, progress bars, and improved error formatting.

- Structured logging and tracing.  
  The current implementation uses standard file-based logging. 
  A production version could add structured logs, run identifiers, per-iteration traces, and latency measurements.

- Domain-specific tools for experiment diagnostics and research assistant.

## Development Notes

This project was developed with the assistance of AI-based coding tools.
Architectural decisions, integration, testing, validation, and final review were performed manually.

## License


This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the [LICENSE](LICENSE) file for details.


