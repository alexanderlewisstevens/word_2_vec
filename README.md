# Graph To Vec

`graph_to_vec` is a Python framework for graph-level and node-level classification over heterogeneous graphs and knowledge-graph-shaped data.

NetworkX is the ergonomic input/output layer. PyTorch Geometric `HeteroData` is the canonical training representation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

PyTorch wheels vary by platform and accelerator. If needed, install the appropriate `torch` build first, then run the editable install.

## Quick Start

```python
import networkx as nx
from graph_to_vec import Graph2VecTransformer, GraphClassificationPipeline

graphs = []
labels = []

for i in range(6):
    g = nx.MultiDiGraph()
    g.add_node("u0", type="user")
    g.add_node("u1", type="user")
    g.add_node("p0", type="product")
    relation = "buys" if i % 2 else "views"
    g.add_edge("u0", "p0", relation=relation)
    g.add_edge("u1", "p0", relation="views")
    graphs.append(g)
    labels.append(i % 2)

pipeline = GraphClassificationPipeline(
    embedder=Graph2VecTransformer(iterations=2, embedding_dim=16, random_state=7)
)
pipeline.fit(graphs, labels)
print(pipeline.predict(graphs))
```

## CLI

```bash
g2v embed --config configs/graph_classification.yaml
g2v train --config configs/graph_classification.yaml
g2v evaluate --config configs/graph_classification.yaml
g2v predict --config configs/graph_classification.yaml
```

The CLI is YAML-driven and persists run config, fitted artifacts, metrics, and optional embeddings.

## V1 Scope

Included:

- NetworkX, triple, and table adapters into `torch_geometric.data.HeteroData`.
- Typed WL/Graph2Vec-style graph embeddings with sklearn compatibility.
- Meta-path/random-walk node embeddings for heterogeneous graphs.
- PyG hetero GraphSAGE and R-GCN classification baselines.
- Library APIs and `g2v` CLI.
- Model registry and persistence helpers.

Reserved for later phases:

- Full HGT.
- GraphMAE-style pretraining.
- GraphGPS-style graph transformers.
- Link prediction, temporal graphs, and distributed training.
