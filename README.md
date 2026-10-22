# ReAct Scientific Coding Agent

**A modular research prototype for LLM-assisted Python analysis and interactive visualization.**

This project implements a lightweight **ReAct-style coding agent** that translates natural-language requests into Python code, executes that code through a configurable execution backend, and returns execution feedback and generated artifacts.

The implementation emphasizes readable components, explicit interfaces, and testable development workflows rather than a large agent framework. 
At the moment, it is intended for experimentation and local development, **not for production use with untrusted generated code**.

## Research motivation

The project is motivated in part by research on **economic network reconstruction and link prediction**, in particular directed supplier–customer networks. 
Such research involves recurring computational tasks: inspecting network structure, evaluating models, checking experiments, and preparing figures for scientific reporting.

The current demonstration generates a **synthetic directed supplier–customer network**, computes in-degree and out-degree, and produces an interactive Plotly visualization. 
It provides a starting point for investigating how an AI coding agent might assist with scientific workflows.
No real-world or proprietary supply-chain dataset is included.

## Current capabilities

The project provides a modular, ReAct-style agent for LLM-assisted Python execution and scientific visualization. 
Its current functionality is organized into three main areas.

### Agent workflow and error recovery

- Accept natural-language requests through a command-line interface.
- Generate structured LLM actions (`execute_python` or `final_answer`).
- Parse and validate JSON responses, including JSON enclosed in Markdown code blocks.
- Retry malformed LLM actions using validation feedback.
- Retry failed Python executions using observations returned by the execution backend.
- Handle recognized LLM connection, timeout, and service errors through a common exception hierarchy.
- Return structured results indicating execution success or failure, generated artifacts, and the number of agent iterations.

The agent follows a lightweight, ReAct-inspired execution loop. 
It does not implement autonomous planning or independent verification of scientific results.

### Python execution and artifact management

The execution layer supports configurable backends, including local subprocess execution and an optional E2B integration.

The local executor provides:

- Configurable execution timeouts and termination of ordinary child processes within the same process group.
- Incremental collection of stdout and stderr, with a configurable combined output-size limit.
- Validation of generated artifacts, including size limits and rejection of symbolic links during collection.
- Chunked artifact copying and staged publication to reduce the risk of incomplete results.
- Separate output directories for each execution, preventing subsequent runs from overwriting previous artifacts.
- Optional Linux process CPU-time and file-size limits. They are disabled by default and are not currently exposed through the command-line interface.

The agent can also validate explicitly required HTML artifacts before reporting task completion. 
Missing artifacts trigger recovery attempts within the configured iteration limit.

These mechanisms improve execution reliability and resource management. 
They do not provide complete operating-system isolation.

### Available backends 

| Component | Implementations | Notes |
| --- | --- | --- |
| LLM | `FakeLLMClient` | Backend used for deterministic testing and local development. |
| LLM | `OllamaClient` | Uses a local Ollama server and a downloaded model. |
| LLM | `OpenAIClient` | Requires API credentials and separate API billing. |
| Execution | `LocalSandboxExecutor` | Local subprocess in a temporary working directory; **not a security boundary**. |
| Execution | `E2BSandboxExecutor` | Optional remote execution backend; requires an E2B account/key. |

The E2B integration has not been validated by the project's automated integration tests. 
Actual behavior depends on the installed SDK version and remote environment.

The current demonstration generates a synthetic directed supplier–customer network and produces an interactive Plotly visualization.

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

The run should generate an interactive HTML artifact under `outputs/run_<unique-id>/synthetic_network_degree_distribution.html` and print a short summary.
Open the HTML file in a browser to explore the plot.

Test the execution-recovery path:

```bash
python3 main.py --llm fake --fake-mode broken_then_success --sandbox local
```

In this mode, the fake client deliberately generates failing code on the first attempt. 
The agent passes the execution error back to the client and tries a corrected action on the next iteration.

> **What this test proves:** the local tool/execution path and deterministic retry flow work together. 
It does **not** establish the reliability of a real LLM, the correctness of arbitrary generated analyses, or secure code isolation.

### HTML artifact validation

The agent supports optional validation of explicitly required HTML artifacts.

For example:

```bash
python3 main.py \
    --llm fake \
    --sandbox local \
    --expect-html synthetic_network_degree_distribution.html
```

When `--expect-html` is provided, the agent checks whether the specified HTML artifact was generated and is not empty.

If the required artifact is missing, the agent provides validation feedback to the LLM and retries, subject to the configured iteration limit.

Multiple HTML files can be required by repeating the `--expect-html` option.

Artifact validation does not guarantee valid HTML, successful Plotly rendering, or the scientific correctness of the generated analysis.

The local executor stores artifacts from each execution in a separate `run_<unique-id>` directory under the configured output directory.
This prevents subsequent executions from overwriting previously generated artifacts with the same filename.

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
| `--expect-html` | `None` | Require specified HTML artifacts; repeat the option for multiple files. |

More example requests are available in `examples/example_prompts.md`.

## Architecture

The design uses pragmatic module names while following a ports-and-adapters approach.

```text
./
├── main.py                              # Composition root and CLI
├── src/agent/
│   ├── react_agent.py                   # Main application use case
│   ├── prompts.py                       # LLM instructions and retry prompt
│   ├── artifact_validation.py           # Expected HTML artifact validation
│   └── schemas.py                       # Agent action/result models
├── src/llm/
│   ├── base.py                          # BaseLLMClient port and ChatMessage
│   ├── errors.py                        # Shared LLM exception hierarchy (error handling)
│   ├── fake_client.py                   # Deterministic adapter
│   ├── ollama_client.py                 # Ollama adapter
│   └── openai_client.py                 # OpenAI adapter
├── src/tools/
│   └── python_tool.py                   # Agent-facing Python tool
├── src/sandbox/
│   ├── base.py                          # BaseSandboxExecutor port and result
│   ├── resource_launcher.py             # Apply optional POSIX resource limits 
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
├── tests/                               # Automated and manual tests
├── examples/
│   └── example_prompts.md
├── LICENSE
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
2. Parse and validate the response, retrying if the action is malformed.
3. Execute Python code when requested.
4. Check execution results and optionally validate expected HTML artifacts.
5. Provide failure observations to the LLM when recovery is possible.
6. Return success or a structured failure result within the iteration limit.


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

The automated tests cover:

- AST-based policy validation and local Python execution.
- Python execution tool behavior.
- Deterministic LLM interactions and agent execution flows.
- Structured LLM response parsing and validation-based retries.
- LLM backend error handling using mocked OpenAI and Ollama clients.
- Expected HTML artifact validation and recovery strategies.
- Local execution timeouts, process-group cleanup, bounded output collection, artifact management, and optional resource limits.

Additional scripts in `tests/` allow manual checks of the OpenAI and Ollama clients.

The tests do not establish comprehensive sandbox isolation, reliable behavior for arbitrary real LLM responses, remote E2B execution,
or the scientific validity of generated analyses.

## Security and known limitations

**This project is an experimental research prototype and is not intended for production execution of untrusted generated code.**

The local execution backend includes several mechanisms designed to improve reliability and limit resource consumption. 
However, these should not be confused with secure sandbox isolation.

### Local execution and security boundaries

`LocalSandboxExecutor` executes generated Python code in a subprocess running with the permissions of the user launching the agent.

Although the executor uses a temporary working directory and performs AST-based policy checks, these mechanisms do not prevent the executed code from accessing other host resources.

In particular:

- The AST policy restricts selected operations and imports but is not a complete security mechanism.
- The temporary working directory does not restrict filesystem access.
- Host filesystem permissions and environment variables remain accessible according to the privileges of the executing process.
- Network access is not isolated.
- Artifact validation and symbolic-link checks do not provide comprehensive filesystem protection.

**The current CLI executes LLM-generated Python code automatically, without requiring explicit user approval.**

Because the local executor does not provide secure filesystem or network isolation, it should only be used in controlled environments where the risks of executing generated code are understood.

Generated code should be inspected before execution when possible. 
For untrusted workloads or sensitive data, stronger operating-system isolation and explicit execution approval are recommended.

### Process and resource limitations

The local executor provides execution timeouts, process-group termination, bounded output collection, and optional POSIX resource limits.

These controls have important limitations:

- Process-group termination does not guarantee termination of processes that detach from the original group.
- CPU-time limits apply to individual processes rather than the aggregate CPU consumption of a complete process hierarchy.
- File-size limits apply to individual files and do not impose a total disk-usage quota.
- Artifact-size limits are enforced during collection and do not prevent excessive disk usage during script execution.
- No general limit on process RAM consumption is currently enforced.
- Linux-specific resource controls are optional and disabled unless explicitly configured.

These features reduce certain operational risks but do not prevent all forms of resource exhaustion.

### Agent and artifact validation limitations

- Recovery from malformed LLM responses or failed Python executions is not guaranteed.
- Recognized LLM backend failures are reported as structured failures; automatic retries for connection, timeout, and service errors are not currently implemented.
- HTML artifact validation is optional and requires explicitly specified filenames.
- Artifact validation checks file existence and non-empty content, not HTML validity, successful Plotly rendering, or scientific correctness.
- Without explicit artifact requirements, the agent may consider successful Python execution or a `final_answer` sufficient to complete a task.
- Successful execution does not guarantee that generated analyses or numerical results are scientifically valid.

### Integration and reproducibility limitations

- The E2B execution backend is optional and has not yet been validated through automated end-to-end integration tests.
- Real LLM behavior may differ from deterministic and mocked test scenarios.
- Dependencies are not fully pinned, so identical environments cannot yet be guaranteed across installations.

### Future security improvements

A future version may introduce an optional container-based execution backend to provide stronger filesystem and network isolation, together with memory, CPU, and process limits.

Until such mechanisms are implemented and tested, `LocalSandboxExecutor` should be treated as a **controlled local execution environment, not a security boundary**.

## Future improvements

- Support for uploaded CSV files.

- Persistent memory with Neo4j or SQL.

- Migrate the agent flow to LangGraph.
  This project implements the ReAct loop manually inside `ReActCodingAgent` to keep the MVP lightweight, transparent, and easy to inspect.
  In a production setting, the same flow could be migrated to LangGraph to obtain explicit state management, conditional routing, checkpointing, 
  and human-in-the-loop approval before executing generated code.

- Stronger sandbox isolation.

- Artifact previews. 
  A more advanced version could automatically display a preview of the graph, a list of generated artifacts, and file metadata.

- Richer CLI using Typer or Rich.
  A more elegant CLI could use Typer for structured commands, Rich for coloured output, tables and panels, progress bars, and improved error formatting.

- Structured logging and tracing.  
  The current implementation uses standard file-based logging. 
  A production version could add structured logs, run identifiers, per-iteration traces, and latency measurements.

- Domain-specific tools for experiment diagnostics and research support.

## Development Notes

This project was developed with the assistance of AI-based coding tools.

Architectural decisions, integration, testing, validation, and final review were performed manually.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

See the [LICENSE](LICENSE) file for details.

