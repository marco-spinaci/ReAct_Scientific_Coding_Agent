from __future__ import annotations

# ast is the Python module used to parse source code into an Abstract Syntax Tree.
# This allows us to inspect the structure of the code before executing it.
import ast

from src.sandbox.errors import SandboxPolicyError


class SandboxPolicy:
    """
    Minimal AST-based policy for generated Python code.

    This policy is intentionally conservative. 
    It is not a complete security boundary, but it reduces the risk of executing obviously
    dangerous code in the local development backend.
    """

    DEFAULT_BLOCKED_MODULES = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "shutil",
        "glob",
        "pickle",
        "multiprocessing",
        "threading",
    }

    DEFAULT_BLOCKED_BUILTINS = {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
    }


    def __init__(self, blocked_modules: set[str] | None = None, blocked_builtins: set[str] | None = None) -> None:
        """
        Create a SandboxPolicy instance.

        Parameters
        ----------
        blocked_modules:
            Optional custom set of module names to block.
            If None, the default blocked modules are used.

        blocked_builtins:
            Optional custom set of builtin function names to block.
            If None, the default blocked builtins are used.
        """
        
        self.blocked_modules = blocked_modules or self.DEFAULT_BLOCKED_MODULES
        
        self.blocked_builtins = blocked_builtins or self.DEFAULT_BLOCKED_BUILTINS


    def validate(self, code: str) -> None:
        """
        Validate generated code before execution.

        This method parses the code into an AST and walks through all nodes.
        Each node is checked for forbidden imports and forbidden builtin calls.

        Parameters
        ----------
        code:
            Python source code to validate.

        Raises
        ------
        SandboxPolicyError
            Raised if the code has invalid syntax or violates the policy.
        """

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise SandboxPolicyError(f"Invalid Python syntax: {exc}") from exc

        
        for node in ast.walk(tree):
            self._check_imports(node)
            self._check_builtin_calls(node)


    def _check_imports(self, node: ast.AST) -> None:
        """
        Check whether an AST node imports a blocked module.

        Handles both:
        - import os
        - from os import path
        """
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module in self.blocked_modules:
                    raise SandboxPolicyError(f"Import of module '{root_module}' is not allowed.")
        
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                return
            
            root_module = node.module.split(".")[0]
            if root_module in self.blocked_modules:
                raise SandboxPolicyError(f"Import from module '{root_module}' is not allowed.")


    def _check_builtin_calls(self, node: ast.AST) -> None:
        """
        Check whether an AST node calls a blocked builtin function.

        Example blocked calls:
        - open("file.txt")
        - eval("1 + 1")
        - exec("print('hello')")
        """
        if isinstance(node, ast.Call):
            func = node.func

            if isinstance(func, ast.Name):
                if func.id in self.blocked_builtins:
                    raise SandboxPolicyError(f"Call to builtin '{func.id}' is not allowed.")

