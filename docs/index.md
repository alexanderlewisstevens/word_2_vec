# Documentation

`graph_to_vec` is organized around one path: accept graph data in user-friendly
forms, normalize it to PyTorch Geometric `HeteroData`, then train or export
classification-ready representations.

## Start Here

- [Walkthrough](walkthrough.md): end-to-end graph and node classification flow.
- [Matching And Querying](matching.md): candidate matching, evidence graphs, rankers,
  and embedding lookup.
- [Architecture](architecture.md): package structure, model paths, and extension hooks.
- [Environment](environment.md): local virtual environment, dependency snapshots, and
  notebook kernel setup.

## Runnable Examples

- `examples/quickstart.py`: smallest graph classification script.
- `examples/end_to_end_classification.py`: script version of the full walkthrough.
- `examples/matching_workflow.py`: comprehensive matching and querying workflow.
- `notebooks/graph_to_vec_walkthrough.ipynb`: notebook version with explanatory cells.
- `notebooks/matching_walkthrough.ipynb`: notebook for candidate matching and queries.

This project is tracked as a normal Python repository. Generated outputs belong under
`runs/` or `artifacts/`; the local `.venv/` is intentionally ignored.
