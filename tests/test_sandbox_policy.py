
import pytest

from src.sandbox.errors import SandboxPolicyError
from src.sandbox.policy import SandboxPolicy


def test_policy_allows_safe_imports() -> None:
    code = """
import pandas as pd
import numpy as np
import plotly.express as px

print("safe code")
"""

    policy = SandboxPolicy()

    # Should not raise
    policy.validate(code)


def test_policy_blocks_os_import() -> None:
    code = """
import os

print(os.getcwd())
"""

    policy = SandboxPolicy()

    with pytest.raises(SandboxPolicyError, match="os"):
        policy.validate(code)


def test_policy_blocks_subprocess_import() -> None:
    code = """
import subprocess

subprocess.run(["ls"])
"""

    policy = SandboxPolicy()

    with pytest.raises(SandboxPolicyError, match="subprocess"):
        policy.validate(code)


def test_policy_blocks_from_import() -> None:
    code = """
from os import path

print(path)
"""

    policy = SandboxPolicy()

    with pytest.raises(SandboxPolicyError, match="os"):
        policy.validate(code)


def test_policy_blocks_open_builtin() -> None:
    code = """
f = open("file.txt", "w")
f.write("hello")
f.close()
"""

    policy = SandboxPolicy()

    with pytest.raises(SandboxPolicyError, match="open"):
        policy.validate(code)


def test_policy_blocks_eval_builtin() -> None:
    code = """
x = eval("2 + 2")
print(x)
"""

    policy = SandboxPolicy()

    with pytest.raises(SandboxPolicyError, match="eval"):
        policy.validate(code)


def test_policy_blocks_syntax_error() -> None:
    code = """
for i in range(10)
    print(i)
"""

    policy = SandboxPolicy()

    with pytest.raises(SandboxPolicyError, match="Invalid Python syntax"):
        policy.validate(code)
