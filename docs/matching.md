# Matching And Querying

`graph_to_vec` now includes a first-class matching path for entity resolution,
candidate review, duplicate detection, and similar "does A match B?" classification
problems.

The workflow is:

1. Generate candidate pairs with blocking and similarity features.
2. Convert each pair into an evidence graph.
3. Train a graph classifier over `match` / `not_match` labels.
4. Rank or query candidate matches by probability.
5. Optionally index graph or node embeddings for nearest-neighbor lookup.

## Candidate Pairs

Use `CandidatePairBuilder` for dedupe or record-linkage candidate generation.

```python
from graph_to_vec import CandidatePairBuilder

builder = CandidatePairBuilder(
    block_on=["email_domain"],
    compare_on=["name", "company", "zip", "tenure_days"],
    ground_truth_col="entity_id",
)

pairs = builder.fit_transform(records)
```

Important behavior:

- Passing one table creates within-table dedupe pairs.
- Passing left and right tables creates cross-table linkage pairs.
- `block_on` limits candidates with exact normalized blocking keys.
- `compare_on` creates `sim_*`, `same_*`, `left_*`, and `right_*` pair features.
- `ground_truth_col` creates a binary label when both records share the same entity id.

## Evidence Graphs

Use `MatchGraphBuilder` to turn each candidate pair into a graph sample.

```python
from graph_to_vec import MatchGraphBuilder

graphs = MatchGraphBuilder().fit_transform(pairs, left_records=records)
```

Each graph contains:

- one `candidate_match` node,
- one `left_record` node,
- one `right_record` node,
- one `evidence` node per compared field,
- typed relations such as `has_left`, `has_right`, `exact_match`,
  `strong_similarity`, and `weak_similarity`.

These graphs can be passed directly to `GraphClassificationPipeline`.

## Match Ranking

Use `MatchRanker` for the end-to-end flow.

```python
from graph_to_vec import MatchRanker

ranker = MatchRanker(candidate_builder=builder)
ranker.fit(records, pairs=pairs)

ranked = ranker.rank(records, pairs=pairs, top_k=20)
query = ranker.query("crm:001", records=records, pairs=pairs, top_k=5)
```

`rank(...)` returns candidate pairs sorted by `match_score`. `query(...)` filters
those rankings to one record id.

## Embedding Queries

`EmbeddingIndex` gives a small in-memory nearest-neighbor index over embeddings.

```python
from graph_to_vec import EmbeddingIndex

embeddings = ranker.pipeline_.transform(graphs)
index = EmbeddingIndex().fit(embeddings, ids=pair_ids)
neighbors = index.query(item_id=pair_ids[0], top_k=10)
```

This is intentionally lightweight. Larger deployments can export the same embeddings
to FAISS, pgvector, LanceDB, or another vector store.

## Full Example

```bash
.venv/bin/python examples/matching_workflow.py
```

The example covers noisy records, candidate generation, evidence-graph construction,
held-out classification, top-k match queries, embedding-neighbor queries, and
artifact persistence.
