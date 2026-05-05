# Graph-To-Vec Walkthrough

This walkthrough covers the intended v1 workflow:

1. Build or load heterogeneous graphs.
2. Convert user-facing data into PyG `HeteroData`.
3. Train a graph-level classifier with typed WL/Graph2Vec embeddings.
4. Train a node-level classifier with the PyG HeteroSAGE baseline.
5. Save artifacts and prepare a YAML-driven CLI run.

The runnable notebook version is `notebooks/graph_to_vec_walkthrough.ipynb`.

## Environment

Use the local virtual environment. No container is required.

```bash
uv venv --python /opt/homebrew/bin/python3.11 .venv
uv pip install -r requirements.txt
uv pip install -e .
uv pip install -r requirements-notebook.txt
```

Optional kernel registration for IDEs and Jupyter frontends:

```bash
.venv/bin/python -m ipykernel install --user --name graph-to-vec \
  --display-name "Python (graph-to-vec)"
```

## Data Flow

NetworkX stays first-class at the API boundary:

```python
import networkx as nx
from graph_to_vec import from_networkx

graph = nx.MultiDiGraph(id="sample-001", label="risky")
graph.add_node("user:1", type="user", spend=250.0)
graph.add_node("device:1", type="device", risk_score=0.9)
graph.add_edge("user:1", "device:1", relation="uses_device")

data = from_networkx(graph)
print(data.node_types)
print(data.edge_types)
```

Internally, training code works with `HeteroData`. The adapters preserve original
ids, node types, relation types, numeric features, labels, masks, and graph metadata.

## Graph Classification

For graph-level classification, use `Graph2VecTransformer` directly or wrap it in
`GraphClassificationPipeline`.

```python
from graph_to_vec import Graph2VecTransformer, GraphClassificationPipeline

pipeline = GraphClassificationPipeline(
    embedder=Graph2VecTransformer(
        iterations=2,
        embedding_dim=64,
        include_features=True,
        random_state=11,
    )
)

pipeline.fit(train_graphs, y_train)
predictions = pipeline.predict(test_graphs)
```

`Graph2VecTransformer` is sklearn-compatible, so it can be used with sklearn
pipelines, grid search, cross validation, and normal estimator persistence.

## Node Classification

For node-level classification, build one heterogeneous graph with node labels and
boolean train/validation/test masks on the target node type.

```python
from graph_to_vec import NodeClassifierTrainer, from_networkx

data = from_networkx(user_network)
trainer = NodeClassifierTrainer(
    target_node_type="user",
    epochs=40,
    lr=0.04,
    device="cpu",
)
trainer.fit(data)
metrics = trainer.evaluate(data, mask=data["user"].test_mask)
```

The trainer defaults to `HeteroSAGEClassifier`, which uses PyG heterogeneous message
passing and returns standard classification metrics.

## Node Embeddings

For node embeddings over typed relations, configure one or more metapaths.

```python
from graph_to_vec import MetaPath2VecNodeEmbedder

embedder = MetaPath2VecNodeEmbedder(
    metapaths=[[("user", "uses_device", "device"), ("device", "used_by", "user")]],
    walk_length=8,
    walks_per_node=4,
    embedding_dim=8,
    random_state=11,
)
embeddings = embedder.fit_transform(data)
```

The v1 implementation includes a simple typed random-walk fallback plus SVD features,
which keeps CPU examples small and portable.

## CLI Runs

The CLI uses YAML configs for repeatability:

```bash
.venv/bin/g2v embed --config configs/graph_classification.yaml
.venv/bin/g2v train --config configs/graph_classification.yaml
.venv/bin/g2v evaluate --config configs/graph_classification.yaml
.venv/bin/g2v predict --config configs/graph_classification.yaml
```

The end-to-end example writes a runnable config and CSV fixtures under
`runs/example_walkthrough/`.

```bash
.venv/bin/python examples/end_to_end_classification.py
.venv/bin/g2v train --config runs/example_walkthrough/graph_classification.yaml
```

## Artifacts

Persisted runs can include:

- `config.yaml`: normalized run config snapshot.
- `model.joblib` or `model.pt`: fitted estimator or PyG trainer artifact.
- `metrics.json`: accuracy and macro F1.
- `embeddings.csv` or `embeddings.parquet`: optional vector exports.
- `predictions.csv`: optional prediction output.

Generated outputs should stay in `runs/` or `artifacts/`, both of which are ignored.

## Matching And Querying

For entity resolution or duplicate detection, use the matching layer:

```python
from graph_to_vec import CandidatePairBuilder, MatchRanker

builder = CandidatePairBuilder(
    block_on=["email_domain"],
    compare_on=["name", "company", "zip"],
    ground_truth_col="entity_id",
)

pairs = builder.fit_transform(records)
ranker = MatchRanker(candidate_builder=builder)
ranker.fit(records, pairs=pairs)
ranker.query("crm:001", records=records, pairs=pairs, top_k=5)
```

See `docs/matching.md` and `examples/matching_workflow.py` for the full flow.
