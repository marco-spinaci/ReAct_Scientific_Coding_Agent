
SYSTEM_PROMPT = """
You are a ReAct-style coding agent.

Your task is to help the user solve Python data analysis and visualization tasks.

You can either:
1. execute Python code using the execute_python tool;
2. provide a final answer.

You must always respond with a valid JSON object using this schema:

{
  "thought": "brief explanation of what you are doing",
  "action": "execute_python" or "final_answer",
  "code": "Python code if action is execute_python, otherwise null",
  "final_answer": "Final answer if action is final_answer, otherwise null"
}

Rules:
- Generate complete, executable Python code.
- Prefer pandas, numpy and plotly for data analysis and visualization.
- Save Plotly visualizations as HTML files using fig.write_html(...).
- Do not use external network access.
- Do not read or write files outside the current working directory.
- If you receive an execution error, use the observation to correct the code.
- Keep the final answer concise and mention generated files if any.
""".strip()



def build_user_prompt(user_request: str, expected_html_files: list[str] | None = None) -> str:
    """
    Build the initial user prompt, including explicitly required HTML artifacts.
    """

    artifact_requirements = ""

    if expected_html_files:
        filenames = "\n".join(f"- {filename}" for filename in expected_html_files)

        artifact_requirements = (
                "\n\nRequired HTML artifacts (exact filenames):\n"
                f"{filenames}\n"
                "Use execute_python to generate all required files. "
                "A final_answer without these artifacts is insufficient."
                )

    return f"""
User request:

{user_request}{artifact_requirements}

Decide the next action.
""".strip()



def build_retry_prompt(user_request: str, previous_code: str, observation: str) -> str:

    return f"""
The previous Python code failed or did not fully satisfy the request.

Original user request:

{user_request}

Previous code:

```python
{previous_code}

Execution observation:

{observation}

Please fix the code and return the next action as valid JSON.
""".strip()


