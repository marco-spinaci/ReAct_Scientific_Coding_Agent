
from __future__ import annotations

import logging

from src.agent.prompts import SYSTEM_PROMPT, build_retry_prompt, build_user_prompt
from src.agent.schemas import AgentAction, AgentRunResult
from src.llm.base import BaseLLMClient, ChatMessage
from src.tools.python_tool import PythonExecutionTool


logger = logging.getLogger(__name__)


class ReActCodingAgent:
    """
    Minimal ReAct-style coding agent.

    The agent asks an LLM for a structured action. 
    If the action is execute_python, it executes the generated code with PythonExecutionTool.
    If the execution fails, the observation is sent back to the LLM for one or more retries.
    
    ReAct means:
    - Reason: the LLM decides what to do;
    - Act: the agent executes a tool;
    - Observe: the tool result is sent back;
    - Repeat if needed.

    This agent supports two LLM actions:
    - execute_python
    - final_answer
    """

    
    def __init__(self, llm: BaseLLMClient, python_tool: PythonExecutionTool, max_iterations: int = 2) -> None:
        """
        Initialize the agent.

        Parameters
        ----------
        llm:
            LLM client used to generate structured actions.

        python_tool:
            Tool used to execute Python code safely.

        max_iterations:
            Maximum number of LLM/tool iterations allowed.
        """

        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")

        self.llm = llm
        self.python_tool = python_tool
        self.max_iterations = max_iterations

    
    def run(self, user_request: str) -> AgentRunResult:
        """
        Run the agent on a user request.

        The loop:
        1. send initial prompt to LLM;
        2. parse structured LLM action;
        3. if final_answer, stop;
        4. if execute_python, run code;
        5. if code succeeds, return success;
        6. if code fails, append observation and retry;
        7. if retries are exhausted, return failure.
        """
        
        logger.info("Starting agent run. max_iterations=%d", self.max_iterations)

        # The system prompt defines behavior and output format.
        # The user prompt wraps the actual user request.
        messages = [
                ChatMessage(role = "system", content = SYSTEM_PROMPT),
                ChatMessage(role = "user", content = build_user_prompt(user_request))
                ]
        
        generated_files: list[str] = []
        last_observation = ""
        last_code = ""

        for iteration in range(1, self.max_iterations + 1):
            logger.info("Agent iteration %d/%d", iteration, self.max_iterations)
            
            # Ask the LLM what to do next.
            raw_response = self.llm.generate(messages)
            
            # Parse and validate the LLM response.
            try:
                action = AgentAction.from_json(raw_response)
            
            except ValueError as exc:
                last_observation = f"Invalid LLM response: {exc}"

                logger.warning(
                        "LLM response validation failed at iteration %d/%d: %s",
                        iteration,
                        self.max_iterations,
                        exc,
                        )

                if iteration < self.max_iterations:
                    messages.append(ChatMessage(role = "assistant", content = raw_response))

                    messages.append(ChatMessage(
                        role = "user",
                        content = (
                            "Your previous response did not match the required AgentAction schema.\n"
                            f"Validation error: {exc}\n"
                            "Return exactly one valid JSON object containing the fields "
                            "'thought', 'action', 'code', and 'final_answer'. "
                            "Do not include Markdown or additional text."
                            ),
                        ))

                continue

            logger.info("LLM selected action: %s", action.action)

            if action.action == "final_answer":
                logger.info("Agent completed with final answer without tool execution.")
                return AgentRunResult(
                        success = True,
                        final_answer = action.final_answer or "",
                        generated_files = generated_files,
                        iterations = iteration,
                        )

            if action.action == "execute_python":
                # Safety check.
                assert action.code is not None

                last_code = action.code
                
                tool_result = self.python_tool.run(action.code)
                
                last_observation = tool_result.observation
                
                generated_files.extend(tool_result.generated_files)

                logger.info(
                        "Tool execution completed. success=%s generated_files=%d",
                        tool_result.success, len(tool_result.generated_files)
                        )

                if tool_result.success:
                    final_answer = self._build_success_answer(
                            observation = tool_result.observation,
                            generated_files = tool_result.generated_files,
                            )

                    logger.info("Agent run completed successfully.")

                    return AgentRunResult(
                            success = True,
                            final_answer = final_answer,
                            generated_files = generated_files,
                            iterations = iteration,
                            )
                
                # If execution fails.
                # This gives the LLM context to repair its code.
                logger.warning("Tool execution failed. Preparing retry prompt.")

                messages.append(ChatMessage(role = "assistant", content = raw_response))
                messages.append(ChatMessage(
                    role = "user", 
                    content = build_retry_prompt(
                        user_request = user_request,
                        previous_code = last_code,
                        observation = last_observation,
                        )))
        
        # If the loop exits, all allowed attempts failed.
        logger.error("Agent failed after maximum iterations.")

        failure_answer = self._build_failure_answer(
                observation = last_observation,
                generated_files = generated_files
                )

        return AgentRunResult(
                success = False,
                final_answer=failure_answer,
                generated_files=generated_files,
                iterations=self.max_iterations,
                )
    

    def _build_success_answer(self, observation: str, generated_files: list[str]) -> str:

        parts = [
                "The task was completed successfully.",
                "",
                "Execution observation:",
                observation,
                ]

        if generated_files:
            parts.append("")
            parts.append("Generated files:")
            for file_path in generated_files:
                parts.append(f"- {file_path}")

        return "\n".join(parts)


    def _build_failure_answer(self, observation: str, generated_files: list[str]) -> str:

        parts = [
                "The task could not be completed successfully within the allowed iterations.",
                "",
                "Last observation:",
                observation or "No observation available.",
                ]

        if generated_files:
            parts.append("")
            parts.append("Generated files before failure:")
            for file_path in generated_files:
                parts.append(f"- {file_path}")

        return "\n".join(parts)


