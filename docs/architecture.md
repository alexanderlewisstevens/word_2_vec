# Architecture

`graph_to_vec` uses a two-layer design.

The front door is ergonomic graph data:

- NetworkX graphs for direct Python use.
- Knowledge-graph triples.
- Node and edge tables described by `GraphSchema`.

The internal training shape is always PyTorch Geometric `HeteroData`. Adapters preserve original node ids, node types, relation types, labels, masks, graph labels, and graph metadata wherever possible.

## Classification Paths

Graph-level classification can use:

- `Graph2VecTransformer` plus any sklearn classifier.
- `GraphClassificationPipeline` for one estimator-style object.
- `GraphClassifierTrainer` with `HeteroSAGEClassifier`.

Node-level classification can use:

- `MetaPath2VecNodeEmbedder` to produce node embeddings for downstream models.
- `NodeClassifierTrainer` with `HeteroSAGEClassifier`.
- `RGCNClassifier` for flattened relation-aware knowledge graph baselines.

## Extension Hooks

The registry reserves model names for later work:

- `hgt`
- `graphmae`
- `graphgps`

Those names intentionally fail with `NotImplementedError` until full implementations are added.
