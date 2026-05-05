"""End-to-end graph-to-vec classification walkthrough.

Run it as a script:

    .venv/bin/python examples/end_to_end_classification.py

The ``# %%`` markers also make this pleasant to run cell-by-cell in IDEs that support
Python notebook cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from graph_to_vec import (
    Graph2VecTransformer,
    GraphClassificationPipeline,
    GraphSchema,
    MetaPath2VecNodeEmbedder,
    NodeClassifierTrainer,
    from_networkx,
    from_tables,
    to_networkx,
)
from graph_to_vec.persistence import load_joblib, save_joblib

RANDOM_STATE = 11
RUN_DIR = Path("runs/example_walkthrough")


# %%
def make_transaction_graph(index: int) -> nx.MultiDiGraph:
    """Create one heterogeneous graph-level classification sample."""

    risky = index % 2 == 1
    label = "risky" if risky else "normal"
    graph = nx.MultiDiGraph(id=f"txn-{index:03d}", label=label)

    customer = f"customer:{index:03d}"
    account = f"account:{index:03d}"
    device = "device:shared-risk" if risky else f"device:{index:03d}"
    merchant = f"merchant:{index % 6:02d}"

    graph.add_node(
        customer,
        type="customer",
        age_days=20.0 + index,
        prior_disputes=2.0 if risky else 0.0,
    )
    graph.add_node(
        account,
        type="account",
        balance=40.0 if risky else 220.0,
        velocity=6.0 if risky else 1.0,
    )
    graph.add_node(
        device,
        type="device",
        risk_score=0.95 if risky else 0.10,
    )
    graph.add_node(
        merchant,
        type="merchant",
        category=float(index % 6),
        chargeback_rate=0.32 if risky else 0.02,
    )

    graph.add_edge(customer, account, relation="owns")
    graph.add_edge(account, customer, relation="owned_by")
    graph.add_edge(customer, device, relation="uses_device")
    graph.add_edge(device, customer, relation="used_by")
    graph.add_edge(account, merchant, relation="chargeback" if risky else "payment")
    graph.add_edge(merchant, account, relation="paid_by")
    return graph


def make_transaction_dataset(count: int = 60) -> tuple[list[nx.MultiDiGraph], list[str]]:
    graphs = [make_transaction_graph(index) for index in range(count)]
    labels = [graph.graph["label"] for graph in graphs]
    return graphs, labels


# %%
def show_adapter_round_trip(graph: nx.MultiDiGraph) -> None:
    """NetworkX is ergonomic, HeteroData is the training representation."""

    data = from_networkx(graph)
    print("Adapter output")
    print("  node types:", data.node_types)
    print("  edge types:", data.edge_types)
    print("  customer ids:", data["customer"].original_id)

    roundtrip = to_networkx(data)
    print("  round-tripped nodes:", roundtrip.number_of_nodes())
    print("  round-tripped edges:", roundtrip.number_of_edges())


# %%
def run_graph_classification() -> GraphClassificationPipeline:
    graphs, labels = make_transaction_dataset()
    train_graphs, test_graphs, y_train, y_test = train_test_split(
        graphs,
        labels,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    pipeline = GraphClassificationPipeline(
        embedder=Graph2VecTransformer(
            iterations=2,
            embedding_dim=64,
            include_features=True,
            random_state=RANDOM_STATE,
        )
    )
    pipeline.fit(train_graphs, y_train)
    predictions = pipeline.predict(test_graphs)

    print("\nGraph classification")
    print(classification_report(y_test, predictions, zero_division=0))

    preview = pd.DataFrame(
        {
            "graph_id": [graph.graph["id"] for graph in test_graphs[:8]],
            "true": y_test[:8],
            "predicted": predictions[:8],
        }
    )
    print(preview.to_string(index=False))

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = RUN_DIR / "graph_pipeline.joblib"
    save_joblib(pipeline, artifact_path)
    reloaded = load_joblib(artifact_path)
    assert reloaded.predict(test_graphs).tolist() == predictions.tolist()
    print(f"Saved and reloaded sklearn pipeline: {artifact_path}")

    return pipeline


# %%
def graphs_to_tables(graphs: list[nx.MultiDiGraph]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten NetworkX samples into node/edge tables for CLI-style workflows."""

    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for graph in graphs:
        graph_id = graph.graph["id"]
        graph_label = graph.graph["label"]
        for node_id, attrs in graph.nodes(data=True):
            row = {
                "graph_id": graph_id,
                "graph_label": graph_label,
                "id": node_id,
                "type": attrs["type"],
            }
            for key, value in attrs.items():
                if key != "type" and isinstance(value, (int, float, np.integer, np.floating)):
                    row[key] = float(value)
            node_rows.append(row)

        for source, target, edge_key, attrs in graph.edges(keys=True, data=True):
            edge_rows.append(
                {
                    "graph_id": graph_id,
                    "source": source,
                    "target": target,
                    "edge_id": edge_key,
                    "relation": attrs["relation"],
                }
            )

    return pd.DataFrame(node_rows).fillna(0.0), pd.DataFrame(edge_rows)


def write_cli_fixture(graphs: list[nx.MultiDiGraph]) -> Path:
    """Write CSVs plus a YAML config that can be used with ``g2v train``."""

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    nodes, edges = graphs_to_tables(graphs)
    nodes_path = RUN_DIR / "transaction_nodes.csv"
    edges_path = RUN_DIR / "transaction_edges.csv"
    config_path = RUN_DIR / "graph_classification.yaml"

    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)

    config = {
        "input": {
            "kind": "tables",
            "nodes_path": str(nodes_path),
            "edges_path": str(edges_path),
            "schema": {
                "graph_id_col": "graph_id",
                "graph_label_col": "graph_label",
            },
        },
        "task": {"level": "graph"},
        "model": {
            "name": "graph2vec",
            "iterations": 2,
            "embedding_dim": 64,
            "random_state": RANDOM_STATE,
        },
        "output_dir": str(RUN_DIR / "cli_run"),
        "artifacts": {
            "model_path": str(RUN_DIR / "cli_run" / "model.joblib"),
            "embeddings_path": str(RUN_DIR / "cli_run" / "embeddings.csv"),
            "predictions_path": str(RUN_DIR / "cli_run" / "predictions.csv"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def demonstrate_table_adapter(graphs: list[nx.MultiDiGraph]) -> None:
    nodes, edges = graphs_to_tables(graphs)
    schema = GraphSchema(graph_id_col="graph_id", graph_label_col="graph_label")
    table_graphs = from_tables(nodes, edges, schema)

    print("\nTable adapter")
    print(f"  node rows: {len(nodes)}")
    print(f"  edge rows: {len(edges)}")
    print(f"  converted graph samples: {len(table_graphs)}")
    print(f"  first graph label: {table_graphs[0].graph_label}")


# %%
def make_user_network(count: int = 36) -> nx.MultiDiGraph:
    """Create one heterogeneous graph for node-level classification."""

    graph = nx.MultiDiGraph(id="user-risk-network")
    risky_device = "device:shared-risk"
    graph.add_node(risky_device, type="device", risk_score=0.95)

    for user_idx in range(count):
        risky = user_idx % 2 == 1
        user_id = f"user:{user_idx:02d}"
        merchant_id = f"merchant:{user_idx % 5:02d}"
        device_id = risky_device if risky else f"device:{user_idx:02d}"

        graph.add_node(
            user_id,
            type="user",
            label=int(risky),
            train_mask=user_idx < 24,
            val_mask=24 <= user_idx < 30,
            test_mask=user_idx >= 30,
            spend=220.0 if risky else 25.0,
            disputes=3.0 if risky else 0.0,
        )
        if device_id not in graph:
            graph.add_node(device_id, type="device", risk_score=0.1)
        if merchant_id not in graph:
            graph.add_node(merchant_id, type="merchant", chargeback_rate=0.03)

        graph.add_edge(user_id, device_id, relation="uses_device")
        graph.add_edge(device_id, user_id, relation="used_by")
        graph.add_edge(user_id, merchant_id, relation="shops_at")
        graph.add_edge(merchant_id, user_id, relation="has_customer")

    return graph


def run_node_classification() -> None:
    data = from_networkx(make_user_network())
    trainer = NodeClassifierTrainer(
        target_node_type="user",
        epochs=60,
        lr=0.04,
        device="cpu",
    )
    trainer.fit(data)

    test_mask = data["user"].test_mask
    metrics = trainer.evaluate(data, mask=test_mask)
    predictions = trainer.predict(data, mask=test_mask)
    truth = data["user"].y[test_mask].detach().cpu().numpy()
    user_ids = np.asarray(data["user"].original_id, dtype=object)[test_mask.detach().cpu().numpy()]

    print("\nNode classification with HeteroSAGE")
    print("  test metrics:", metrics)
    print(
        pd.DataFrame({"user_id": user_ids, "true": truth, "predicted": predictions}).to_string(
            index=False
        )
    )

    metapath_embedder = MetaPath2VecNodeEmbedder(
        metapaths=[[("user", "uses_device", "device"), ("device", "used_by", "user")]],
        walk_length=8,
        walks_per_node=4,
        embedding_dim=8,
        random_state=RANDOM_STATE,
    )
    embeddings = metapath_embedder.fit_transform(data)
    print("  metapath user embedding shape:", embeddings["user"].shape)


# %%
def main() -> None:
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    graphs, _ = make_transaction_dataset()
    show_adapter_round_trip(graphs[1])
    run_graph_classification()
    demonstrate_table_adapter(graphs)
    config_path = write_cli_fixture(graphs)
    print(f"\nCLI config written: {config_path}")
    print(f"Try: .venv/bin/g2v train --config {config_path}")
    run_node_classification()


if __name__ == "__main__":
    main()
