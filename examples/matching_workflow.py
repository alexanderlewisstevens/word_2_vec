"""Comprehensive matching and querying walkthrough.

Run it as a script:

    .venv/bin/python examples/matching_workflow.py

The ``# %%`` markers let IDEs run this file cell-by-cell.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from graph_to_vec import (
    CandidatePairBuilder,
    EmbeddingIndex,
    Graph2VecTransformer,
    GraphClassificationPipeline,
    MatchGraphBuilder,
    MatchRanker,
)
from graph_to_vec.persistence import load_joblib, save_joblib

RANDOM_STATE = 17
RUN_DIR = Path("runs/matching_walkthrough")


# %%
def make_record_table() -> pd.DataFrame:
    """Create noisy records where multiple rows can refer to the same entity."""

    rows = [
        {
            "id": "crm:001",
            "entity_id": "person:ada",
            "name": "Ada Lovelace",
            "email_domain": "analytical.example",
            "company": "Analytical Engines",
            "city": "London",
            "zip": "EC1A",
            "tenure_days": 930,
            "annual_value": 1200.0,
        },
        {
            "id": "support:771",
            "entity_id": "person:ada",
            "name": "A. Lovelace",
            "email_domain": "analytical.example",
            "company": "Analytical Engine Co",
            "city": "London",
            "zip": "EC1A",
            "tenure_days": 910,
            "annual_value": 1180.0,
        },
        {
            "id": "billing:088",
            "entity_id": "person:ada",
            "name": "Ada L",
            "email_domain": "analytical.example",
            "company": "Analytical Engines",
            "city": "London",
            "zip": "EC1A",
            "tenure_days": 940,
            "annual_value": 1215.0,
        },
        {
            "id": "crm:014",
            "entity_id": "person:grace",
            "name": "Grace Hopper",
            "email_domain": "navy.example",
            "company": "Compiler Works",
            "city": "Arlington",
            "zip": "22201",
            "tenure_days": 760,
            "annual_value": 980.0,
        },
        {
            "id": "support:332",
            "entity_id": "person:grace",
            "name": "G. Hopper",
            "email_domain": "navy.example",
            "company": "Compiler Works",
            "city": "Arlington",
            "zip": "22201",
            "tenure_days": 745,
            "annual_value": 1005.0,
        },
        {
            "id": "crm:028",
            "entity_id": "person:katherine",
            "name": "Katherine Johnson",
            "email_domain": "orbital.example",
            "company": "Orbital Math",
            "city": "Hampton",
            "zip": "23666",
            "tenure_days": 680,
            "annual_value": 760.0,
        },
        {
            "id": "support:118",
            "entity_id": "person:katherine",
            "name": "K. Johnson",
            "email_domain": "orbital.example",
            "company": "Orbital Mathematics",
            "city": "Hampton",
            "zip": "23666",
            "tenure_days": 675,
            "annual_value": 755.0,
        },
        {
            "id": "crm:077",
            "entity_id": "person:alan",
            "name": "Alan Turing",
            "email_domain": "computing.example",
            "company": "Computing Lab",
            "city": "Manchester",
            "zip": "M13",
            "tenure_days": 540,
            "annual_value": 640.0,
        },
        {
            "id": "support:209",
            "entity_id": "person:alan",
            "name": "A Turing",
            "email_domain": "computing.example",
            "company": "Computing Laboratory",
            "city": "Manchester",
            "zip": "M13",
            "tenure_days": 535,
            "annual_value": 650.0,
        },
        {
            "id": "billing:510",
            "entity_id": "person:alan",
            "name": "Alan Mathison Turing",
            "email_domain": "computing.example",
            "company": "Computing Lab",
            "city": "Manchester",
            "zip": "M13",
            "tenure_days": 545,
            "annual_value": 645.0,
        },
        {
            "id": "crm:900",
            "entity_id": "person:decoy-1",
            "name": "Ada Lawrence",
            "email_domain": "analytical.example",
            "company": "Different Analytics",
            "city": "London",
            "zip": "EC2A",
            "tenure_days": 80,
            "annual_value": 210.0,
        },
        {
            "id": "support:901",
            "entity_id": "person:decoy-2",
            "name": "Grace Harper",
            "email_domain": "navy.example",
            "company": "Harbor Systems",
            "city": "Arlington",
            "zip": "22202",
            "tenure_days": 60,
            "annual_value": 190.0,
        },
        {
            "id": "crm:902",
            "entity_id": "person:decoy-3",
            "name": "Katherine Johnston",
            "email_domain": "orbital.example",
            "company": "Orbit Sales",
            "city": "Hampton",
            "zip": "23665",
            "tenure_days": 95,
            "annual_value": 205.0,
        },
        {
            "id": "support:903",
            "entity_id": "person:decoy-4",
            "name": "Alana Turner",
            "email_domain": "computing.example",
            "company": "Compute Desk",
            "city": "Manchester",
            "zip": "M14",
            "tenure_days": 70,
            "annual_value": 180.0,
        },
    ]
    return pd.DataFrame(rows)


# %%
def build_labeled_candidate_pairs(records: pd.DataFrame) -> pd.DataFrame:
    builder = CandidatePairBuilder(
        block_on=["email_domain"],
        compare_on=[
            "name",
            "email_domain",
            "company",
            "city",
            "zip",
            "tenure_days",
            "annual_value",
        ],
        ground_truth_col="entity_id",
    )
    return builder.fit_transform(records)


def train_match_ranker(
    records: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[MatchRanker, pd.DataFrame, pd.DataFrame]:
    train_pairs, test_pairs = train_test_split(
        pairs,
        test_size=0.35,
        random_state=RANDOM_STATE,
        stratify=pairs["label"],
    )
    ranker = MatchRanker(
        candidate_builder=CandidatePairBuilder(
            block_on=["email_domain"],
            compare_on=[
                "name",
                "email_domain",
                "company",
                "city",
                "zip",
                "tenure_days",
                "annual_value",
            ],
            ground_truth_col="entity_id",
        ),
        graph_builder=MatchGraphBuilder(),
        pipeline=GraphClassificationPipeline(
            embedder=Graph2VecTransformer(
                iterations=2,
                embedding_dim=96,
                include_features=True,
                random_state=RANDOM_STATE,
            ),
            classifier=LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
    )
    ranker.fit(records, pairs=train_pairs)
    return ranker, train_pairs, test_pairs


# %%
def evaluate_ranker(ranker: MatchRanker, records: pd.DataFrame, test_pairs: pd.DataFrame) -> None:
    predictions = ranker.predict(records, pairs=test_pairs)
    print("\nHeld-out candidate classification")
    print(classification_report(test_pairs["label"], predictions, zero_division=0))

    ranked = ranker.rank(records, pairs=test_pairs)
    print("Ranked held-out candidates")
    print(
        ranked[
            [
                "rank",
                "left_id",
                "right_id",
                "label",
                "prediction",
                "match_score",
                "candidate_score",
            ]
        ].to_string(index=False)
    )


def query_for_record(ranker: MatchRanker, records: pd.DataFrame, record_id: str) -> pd.DataFrame:
    all_pairs = build_labeled_candidate_pairs(records)
    query = ranker.query(record_id, records=records, pairs=all_pairs, top_k=5)
    return query[
        [
            "left_id",
            "right_id",
            "label",
            "prediction",
            "match_score",
            "candidate_score",
        ]
    ]


def build_candidate_embedding_index(
    ranker: MatchRanker,
    records: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    graphs = ranker.graph_builder_.transform(pairs, left_records=records)
    embeddings = ranker.pipeline_.transform(graphs)
    ids = [f"{row.left_id}|{row.right_id}" for row in pairs.itertuples(index=False)]
    metadata = pairs[["left_id", "right_id", "label", "candidate_score"]].reset_index(drop=True)
    index = EmbeddingIndex().fit(embeddings, ids=ids, metadata=metadata)
    return index.query(item_id=ids[0], top_k=5)


# %%
def main() -> None:
    records = make_record_table()
    pairs = build_labeled_candidate_pairs(records)

    print("Records")
    print(records[["id", "entity_id", "name", "email_domain", "company"]].to_string(index=False))
    print("\nCandidate pairs")
    print(
        pairs[["left_id", "right_id", "label", "candidate_score"]]
        .sort_values("candidate_score", ascending=False)
        .to_string(index=False)
    )

    ranker, _, test_pairs = train_match_ranker(records, pairs)
    evaluate_ranker(ranker, records, test_pairs)

    print("\nTop matches for crm:001")
    print(query_for_record(ranker, records, "crm:001").to_string(index=False))

    print("\nEmbedding-neighbor query for one candidate graph")
    print(build_candidate_embedding_index(ranker, records, pairs).to_string(index=False))

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = RUN_DIR / "match_ranker.joblib"
    save_joblib(ranker, artifact_path)
    reloaded = load_joblib(artifact_path)
    assert reloaded.rank(records, pairs=pairs).head(3)["left_id"].tolist()
    print(f"\nSaved and reloaded match ranker: {artifact_path}")


if __name__ == "__main__":
    main()
