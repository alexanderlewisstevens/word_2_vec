from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from graph_to_vec import (
    CandidatePairBuilder,
    EmbeddingIndex,
    Graph2VecTransformer,
    GraphClassificationPipeline,
    MatchGraphBuilder,
    MatchRanker,
)


def _records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "left:a1",
                "entity_id": "entity:a",
                "name": "Jordan Lee",
                "email_domain": "acme.example",
                "company": "Acme Labs",
                "zip": "80202",
                "tenure_days": 420,
            },
            {
                "id": "right:a2",
                "entity_id": "entity:a",
                "name": "J. Lee",
                "email_domain": "acme.example",
                "company": "Acme Labs",
                "zip": "80202",
                "tenure_days": 390,
            },
            {
                "id": "left:b1",
                "entity_id": "entity:b",
                "name": "Morgan Ray",
                "email_domain": "north.example",
                "company": "Northwind",
                "zip": "10001",
                "tenure_days": 110,
            },
            {
                "id": "right:b2",
                "entity_id": "entity:b",
                "name": "Morgan R.",
                "email_domain": "north.example",
                "company": "Northwind",
                "zip": "10001",
                "tenure_days": 120,
            },
            {
                "id": "left:c1",
                "entity_id": "entity:c",
                "name": "Casey Quinn",
                "email_domain": "acme.example",
                "company": "Other Co",
                "zip": "94105",
                "tenure_days": 40,
            },
            {
                "id": "right:d1",
                "entity_id": "entity:d",
                "name": "Riley Stone",
                "email_domain": "north.example",
                "company": "Different Inc",
                "zip": "60601",
                "tenure_days": 45,
            },
        ]
    )


def test_candidate_pair_builder_blocks_scores_and_labels() -> None:
    records = _records()
    builder = CandidatePairBuilder(
        block_on=["email_domain"],
        compare_on=["name", "email_domain", "company", "zip", "tenure_days"],
        ground_truth_col="entity_id",
    )

    pairs = builder.fit_transform(records)

    assert {"left_id", "right_id", "candidate_score", "label"}.issubset(pairs.columns)
    assert pairs["label"].sum() == 2
    assert pairs["same_email_domain"].all()
    assert len(pairs) < 15


def test_match_graph_builder_creates_evidence_graphs() -> None:
    records = _records()
    pairs = CandidatePairBuilder(
        block_on=["email_domain"],
        compare_on=["name", "email_domain", "company", "zip"],
        ground_truth_col="entity_id",
    ).fit_transform(records)

    graph = MatchGraphBuilder().fit_transform(pairs.head(1), left_records=records)[0]

    assert graph.graph["label"] in {0, 1}
    assert any(attrs["type"] == "candidate_match" for _, attrs in graph.nodes(data=True))
    assert any(attrs["relation"] == "exact_match" for _, _, attrs in graph.edges(data=True))


def test_match_ranker_scores_and_queries_candidate_matches() -> None:
    records = _records()
    builder = CandidatePairBuilder(
        block_on=["email_domain"],
        compare_on=["name", "email_domain", "company", "zip", "tenure_days"],
        ground_truth_col="entity_id",
    )
    pairs = builder.fit_transform(records)
    ranker = MatchRanker(
        candidate_builder=builder,
        pipeline=GraphClassificationPipeline(
            embedder=Graph2VecTransformer(iterations=2, embedding_dim=32, random_state=7),
            classifier=LogisticRegression(max_iter=1000),
        ),
    )

    ranker.fit(records, pairs=pairs)
    ranked = ranker.rank(records, pairs=pairs)
    query = ranker.query("left:a1", records=records, pairs=pairs, top_k=3)

    assert ranked["match_score"].between(0.0, 1.0).all()
    assert ranked.iloc[0]["match_score"] >= ranked.iloc[-1]["match_score"]
    assert not query.empty
    assert set(query["left_id"]) | set(query["right_id"])


def test_embedding_index_queries_by_id_and_vector() -> None:
    embeddings = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    index = EmbeddingIndex().fit(embeddings, ids=["a", "b", "c"])

    by_id = index.query(item_id="a", top_k=1)
    by_vector = index.query(vector=[1.0, 0.0], top_k=2, exclude_self=False)

    assert by_id.iloc[0]["id"] == "b"
    assert by_vector.iloc[0]["id"] == "a"
