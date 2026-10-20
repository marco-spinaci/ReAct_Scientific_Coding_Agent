# Reusable ReAct-style Coding Agent with Sandboxed Python Execution

**Author**: Marco Spinaci
**Date**: 27-05-2026
**Environment tested**:
- Ubuntu 22.04.5 LTS
- Python 3.10.12


This project implements a minimal ReAct-style coding agent for Python data analysis and visualization tasks.

The implementation is intentionally lightweight: the goal is to show the core agentic workflow clearly.

The agent can:

- receive a natural-language user request;
- ask an LLM backend for a structured action;
- execute generated Python code through a controlled sandbox;
- collect stdout, stderr, errors and generated files;
- save Plotly visualizations as HTML files;
- retry once when execution fails;
- log the execution flow;
- store lightweight run metadata in JSONL format.

The demonstration task uses a **synthetic directed supplier-customer network**, inspired by firm-level supply-chain link prediction problems. 
No proprietary or real FactSet data is used.



## 1. Project structure

```text
ai_agent_coding_task/
│
├── README.md
│
├── main.py
│
├── .env.example
│
├── requirements.txt
├── requirements-openai.txt
├── requirements-e2b.txt
│
├── examples
│   └── example_prompts.md
│
├── logs        # created at runtime
│   └── ...
│
├── outputs     # created at runtime
│   └── ...
│
├── src
│   ├── agent
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   ├── react_agent.py
│   │   └── schemas.py
│   │
│   ├── domain
│   │   ├── __init__.py
│   │   └── synthetic_network_prompt.py
│   │
│   ├── llm
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── fake_client.py
│   │   ├── openai_client.py
│   │   └── ollama_client.py
│   │
│   ├── memory
│   │   ├── __init__.py
│   │   └── json_store.py
│   │
│   ├── sandbox
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── e2b_executor.py
│   │   ├── errors.py
│   │   ├── local_executor.py
│   │   └── policy.py
│   │
│   ├── tools
│   │   ├── __init__.py
│   │   └── python_tool.py
│   │
│   └── utils
│       ├── __init__.py
│       └── logging_config.py
│
└── tests
    ├── __init__.py
    ├── scripts_test_openai_client.py
    ├── scripts_test_ollama_client.py
    ├── test_fake_llm_client.py
    ├── test_local_executor.py
    ├── test_python_tool.py
    ├── test_react_agent.py
    └── test_sandbox_policy.py

12 directories, 40 files
```



## 2. Installation

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the offline dependencies:

```bash
pip install -r requirements.txt
```

Optional OpenAI backend, and/or E2B backend:

```bash
pip install -r requirements-openai.txt

pip install -r requirements-e2b.txt
```



## 3. Running the deterministic offline demo

The project is fully runnable without paid API access using the deterministic `FakeLLMClient`. 

```bash
python3 main.py --llm fake --sandbox local
```

This runs the default synthetic supplier-customer network demo and should generate a Plotly HTML file under
`outputs/synthetic_network_degree_distribution.html`

You can also test the retry flow:

```bash
python3 main.py --llm fake --fake-mode broken_then_success --sandbox local
```

In this mode, the fake LLM intentionally returns broken code on the first iteration. 
The agent observes the error, retries, and produces a valid HTML output on the second iteration.



## 4. Optional LLM backends

### 4.1. Fake LLM Backend

Backend used for deterministic testing and local development.

### 4.2. OpenAI backend

The OpenAI backend is optional and requires separate API billing.

Create a `.env` file with `cp .env.example .env`, then set:

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.2
```

and run `python3 main.py --llm openai --sandbox local` 

### 4.3. Ollama local backend

The project can also use a local LLM through Ollama, without external API keys.

Install Ollama and pull a lightweight coder model:

```bash
ollama pull qwen2.5-coder:3b
```

Install Python dependencies:

```bash
pip install -r requirements-ollama.txt
```

and run `python3 main.py --llm ollama --sandbox local --ollama-model qwen2.5-coder:3b`.

The Ollama backend uses structured outputs through a JSON schema to make the model return an AgentAction-compatible response.

#### 4.3.1. Simple Plotly smoke test with Ollama

When using a small local model such as `qwen2.5-coder:3b`, it is useful to start with a minimal plotting task before running 
the full synthetic supplier-customer network demo.

The complete supplier-customer task requires the model to generate a longer script involving synthetic data generation, directed edges, 
degree computation, Plotly visualization and HTML artifact creation. 
On CPU-only machines, this can be slow and may lead to local LLM timeouts.

A smaller smoke test can be used to validate the full agentic pipeline:

```bash
python3 main.py \
  --llm ollama \
  --sandbox local \
  --ollama-model qwen2.5-coder:3b \
  --request "Generate Python code that creates a simple Plotly line chart with x=[1,2,3] and y=[1,4,9], saves it as simple_plot.html, and prints a short summary."
```

This test checks the following components:

1. local Ollama model;
2. structured AgentAction JSON;
3. Python code generation;
4. sandboxed execution;
5. Plotly HTML artifact generation;
6. final agent response.

If this task succeeds, the local LLM backend, sandbox, tool interface and artifact collection are working correctly.



## 5. Optional E2B sandbox backend

The framework supports a pluggable sandbox architecture. 
The local backend is used for deterministic development and artifact collection, while an optional E2B backend 
is provided to align with the assignment requirement of controlled sandbox execution.

The E2B backend requires a separate E2B API key.
Therefore, set your key in `.env` file by adding `E2B_API_KEY=your_e2b_api_key_here`, and run:

```bash
python3 main.py --llm fake --sandbox e2b
```

The E2B backend:

- applies the local AST policy before sending code to E2B;
- executes the generated code inside an E2B sandbox;
- searches for .html / .htm files in the remote working directory;
- downloads generated HTML artifacts to the local output directory.



## 6. Running tests

The scripts in the `tests` folder cover:

- sandbox policy checks;
- local sandbox execution;
- generated HTML collection;
- Python execution tool behavior;
- fake LLM behavior;
- ReAct agent success and retry flows.

Run:

```bash
pytest -q
```

The `-q` option decreases output verbosity, while `-v` increases it.

Additional scripts, `scripts_test_openai_client.py` and `scripts_test_ollama_client.py`, are provided for manually testing the OpenAI API key and the Ollama client, respectively.



## 7. Architecture overview

The core flow is:

1. User request
2. ReActCodingAgent
3. LLM backend
4. Structured AgentAction JSON
5. PythonExecutionTool
6. SandboxExecutor
7. SandboxResult
8. Observation / retry / final answer

The framework is deliberately modular:

- `agent/` contains orchestration logic;
- `llm/` abstracts the language model backend;
- `sandbox/` abstracts code execution backend;
- `tools/` exposes sandbox execution as an agent tool;
- `memory/` stores lightweight run metadata;
- `domain/` contains the synthetic network demo prompt.

Folder names are chosen for practical reasons, but the project attempts to follow the required architectural guidelines.
The abstract ports are defined by `BaseLLMClient` and `BaseSandboxExecutor`, the concrete adapters implement these ports, 
and the main use case is implemented by `ReActCodingAgent`. Dependencies are wired exclusively in `main.py`, which acts 
as the composition root.

### 7.1. Agent layer

The agent is implemented in `src/agent/react_agent.py`.

It is responsible for:

- building the conversation with the LLM;
- parsing the structured LLM response;
- executing Python code through a tool;
- observing failures;
- retrying when needed;
- returning the final result.

The agent does not know whether the sandbox is local or remote. 
It depends only on the `PythonExecutionTool`.

### 7.2. LLM layer

The LLM interface is defined in `src/llm/base.py`.

As aforementioned, different clients are currently available:

- `FakeLLMClient`;
- `OpenAIClient`;
- `OllamaClient`.

The first is deterministic and used for offline tests and demos, 
the second is optional and requires separate API billing, and the third 
runs an open-source coding model locally without requiring external API keys.

### 7.3. Tool layer
The tool layer is implemented in `src/tools/python_tool.py`.

The `PythonExecutionTool` converts a low-level `SandboxResult` into an agent-friendly observation.
This keeps the agent decoupled from sandbox-specific implementation details.

### 7.4. Sandbox layer
The sandbox interface is defined in `src/sandbox/base.py`.

As aforementioned, two different implementations are currently available:

- `LocalSandboxExecutor`;
- `E2BSandboxExecutor`.

#### 7.4.1. LocalSandboxExecutor

The local sandbox is intended for local development and deterministic testing.

It provides:

- subprocess execution;
- timeout;
- temporary working directory;
- stdout/stderr capture;
- artifact collection;
- AST-based policy checks.

It is not a full security boundary.

#### 7.4.2. E2BSandboxExecutor

The E2B sandbox is optional and designed to align with the assignment requirement for controlled sandbox execution.

It provides:

- remote sandbox execution;
- local policy validation before sending code;
- controlled remote working directory;
- HTML artifact download.

### 7.5. Sandbox Policy layer

The policy is implemented in `src/sandbox/policy.py`.

It uses Python's `ast` module to reject:

- dangerous imports;
- dangerous builtins;
- invalid Python syntax.

This policy is intentionally conservative and should be considered a first layer of defense only.

### 7.6. Memory layer

Run metadata is stored by `src/memory/json_store.py`.

The store appends one JSON object per run to `logs/runs.jsonl`.

The record includes:

- timestamp;
- LLM backend;
- sandbox backend;
- user request;
- success flag;
- number of iterations;
- generated files.

This supports traceability without adding a database dependency.

### 7.7. Domain layer

The default demo prompt is stored in `src/domain/synthetic_network_prompt.py`.

The current domain example uses synthetic supplier-customer network data, but the core framework is domain-independent.

The same architecture could later support a research assistant for firm-level supply-chain link prediction workflows.

Potential future tools:

- read experiment configurations;
- inspect model logs;
- summarize LightGBM, GAT or MFRGAT runs;
- check train/test leakage;
- compute degree diagnostics;
- generate Plotly diagnostics;
- generate LaTeX tables for reports and papers.

This extension is intentionally not included in the assignment demo to avoid using proprietary data and to keep the deliverable focused.



## 8. Design decisions

### 8.1. ReAct-style loop

The agent follows a minimal ReAct-inspired loop:

1. ask the LLM for the next action;
2. execute code when the selected action is `execute_python`;
3. observe the result;
4. retry when execution fails;
5. return a final answer when the task succeeds or iterations are exhausted.

The implementation limits the number of iterations to avoid uncontrolled execution loops.

### 8.2. Structured LLM output

The LLM is expected to return a JSON object with:

```json
{
  "thought": "...",
  "action": "execute_python",
  "code": "...",
  "final_answer": null
}
```

or:

```json
{
  "thought": "...",
  "action": "final_answer",
  "code": null,
  "final_answer": "..."
}
```

This keeps the control flow explicit and testable.

### 8.3. Pluggable sandbox backends

The project defines a common `BaseSandboxExecutor` interface.

As aforementioned, implemented backends are:

- `LocalSandboxExecutor`;
- `E2BSandboxExecutor`.

This makes it possible to change the execution backends or implement new ones without modifying the agent’s logic.




## 9. Security limitations

The local sandbox is designed for a coding exercise and local development. 
It is not a complete isolation mechanism.

Current limitations:

- AST checks can reduce obvious risks but cannot guarantee full security;
- no OS-level isolation is enforced in the local backend;
- network blocking is approximated by policy checks, not by a firewall;
- a production system should use stronger isolation such as E2B, Docker with strict profiles, Firejail, gVisor or a dedicated remote execution service.

The E2B backend provides stronger isolation but still depends on external service configuration and API availability.



## 10. Future improvements

Possible extensions:

- robust JSON extraction from Markdown-formatted LLM responses.
  This extension is particularly useful for real-world backends such as 
  OpenAI, Ollama or other local models.

- Support for uploaded CSV files.

- Persistent memory with Neo4j or SQL.

- Migrate the agent flow to LangGraph.
  This project implements the ReAct loop manually inside `ReActCodingAgent` to keep the MVP lightweight, transparent, and easy to inspect.
  In a production setting, the same flow could be migrated to LangGraph to obtain explicit state management, 
  conditional routing, checkpointing, and human-in-the-loop approval before executing generated code.

- Stronger sandbox isolation.

- Artifact previews. 
  A more advanced version could automatically display a preview of the graph, a list of generated artifacts, and 
  file metadata.

- Richer retry and recovery strategies.
  Now, retrying is very simple. In a more advanced version, we could distinguish between syntax error, import error,
  policy error, runtime error, HTML file not generated, and user request not fulfilled. 
 
- Richer CLI using Typer or Rich.
  A more elegant CLI could use Typer for structured commands, Rich for coloured output, tables and panels, progress bars,
  and improved error formatting.

- Structured logging and tracing.
  The current implementation uses standard file-based logging. A production version could add structured logs, run identifiers, 
  per-iteration traces, and latency measurements.

- Domain-specific tools for experiment diagnostics and research assistant.



## Development Notes

This project was developed with the assistance of AI-based coding tools.
Architectural decisions, integration, testing, validation, and final review were performed manually.

Regarding the optional Advanced Features, this project partially implements the “Agent Robustness” option.

Implemented:
- retry after failed Python execution;
- sandbox policy validation before execution;
- timeout handling;
- runtime error capture;
- graceful `SandboxResult` objects;
- deterministic fake LLM for reproducible recovery tests.

Not yet implemented:
- robust JSON extraction from Markdown-formatted LLM outputs;
- retry on invalid LLM responses;
- explicit artifact validation when an HTML output is expected;
- typed error categories;
- fallback between LLM and sandbox backends.

The “Chat Memory & Persistence” option is not fully implemented; the current project only provides lightweight JSONL run metadata.
A full implementation would require a `ChatMemoryPort`, a Neo4j adapter, conversation/message entities, and retrieval methods 
for past conversations and generated artifacts.



