
from __future__ import annotations


SYNTHETIC_NETWORK_REQUEST = """
Generate a synthetic directed supplier-customer network.

Requirements:
- Create 100 synthetic firms.
- Assign randomly each firm to one synthetic industrial sector.
- Generate directed supplier-to-customer links.
- Remove self-loops and duplicate edges.
- Compute in-degree and out-degree for each firm.
- Create an interactive Plotly visualization of the in-degree and out-degree distributions.
- Save the visualization as an HTML file named synthetic_network_degree_distribution.html.
- Print a short textual summary of the generated network.
""".strip()


def get_synthetic_network_request() -> str:
    """
    Return the default user request used for the synthetic network demo.

    The demo is inspired by supplier-customer link prediction problems, but it
    uses only synthetic data and does not depend on proprietary FactSet data.
    """

    return SYNTHETIC_NETWORK_REQUEST
