
from __future__ import annotations

import json

from src.llm.base import BaseLLMClient, ChatMessage


class FakeLLMClient(BaseLLMClient):
    """
    Deterministic LLM client used for local development and tests.

    The fake client returns a structured JSON action that mimics the response
    expected from a real LLM in the ReAct loop:
    - thought
    - action
    - code
    - final_answer
    """

    
    def __init__(self, mode: str = "success") -> None:
        self.mode = mode
        self.call_count = 0

    
    def generate(self, messages: list[ChatMessage]) -> str:
        self.call_count += 1

        if self.mode == "success":
            return self._success_response()

        if self.mode == "broken_then_success":
            if self.call_count == 1:
                return self._broken_response()
            return self._success_response()

        if self.mode == "final_only":
            return self._final_response()

        raise ValueError(f"Unsupported FakeLLMClient mode: {self.mode}")

    
    def _success_response(self) -> str:

        # The r'''...''' prefix means "raw string": backslashes are treated literally.
        # This is useful when storing code as text.
        code = r'''
import numpy as np
import pandas as pd
import plotly.express as px

rng = np.random.default_rng(12342)

n_firms = 100
n_edges = 250
sectors = ["Manufacturing", "Technology", "Healthcare", "Energy", "Finance"]

firms = pd.DataFrame({
    "firm_id": [f"FIRM_{i:03d}" for i in range(n_firms)],
    "sector": rng.choice(sectors, size=n_firms),
})

edges = pd.DataFrame({
    "supplier": rng.choice(firms["firm_id"], size = n_edges),
    "customer": rng.choice(firms["firm_id"], size = n_edges),
})

edges = edges[edges["supplier"] != edges["customer"]].drop_duplicates()

out_degree = edges.groupby("supplier").size().rename("out_degree")
in_degree = edges.groupby("customer").size().rename("in_degree")

degree_df = firms.set_index("firm_id").join([out_degree, in_degree]).fillna(0)
degree_df["out_degree"] = degree_df["out_degree"].astype(int)
degree_df["in_degree"] = degree_df["in_degree"].astype(int)
degree_df["total_degree"] = degree_df["out_degree"] + degree_df["in_degree"]

long_df = degree_df.reset_index().melt(
    id_vars = ["firm_id", "sector"],
    value_vars = ["in_degree", "out_degree"],
    var_name = "degree_type",
    value_name = "degree",
)

fig = px.histogram(
    long_df,
    x = "degree",
    color = "degree_type",
    barmode = "overlay",
    marginal = "box",
    title = "Synthetic supplier-customer network: in/out degree distribution",
)

fig.write_html("synthetic_network_degree_distribution.html")

print(f"Generated {len(firms)} firms.")
print(f"Generated {len(edges)} directed supplier-customer links after removing self-loops and duplicates.")
print("Saved plot to synthetic_network_degree_distribution.html.")
'''
        
        return json.dumps({
            "thought": (
                "I should generate Python code that creates a synthetic "
                "directed supplier-customer network, computes degree "
                "statistics, and saves a Plotly HTML visualization."),
            "action": "execute_python",
            "code": code,
            "final_answer": None})


    def _broken_response(self) -> str:
        code = r'''
import pandas as pd

df = pd.DataFrame({"x": [1, 2, 3]})
print(df["missing_column"])
'''
        return json.dumps({
            "thought": (
                "I will intentionally return code with a runtime error. "
                "This is useful for testing the retry loop."),
            "action": "execute_python",
            "code": code,
            "final_answer": None})


    def _final_response(self) -> str:
        return json.dumps({
            "thought": "No tool execution is needed.",
            "action": "final_answer",
            "code": None,
            "final_answer": "This is a deterministic final answer from the fake client."})


