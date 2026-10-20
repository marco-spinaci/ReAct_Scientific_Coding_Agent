
# Example prompts

This file contains example prompts that can be passed to the ReAct coding agent.

## Synthetic supplier-customer network demo

```text
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

