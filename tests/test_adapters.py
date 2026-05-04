from __future__ import annotations

import networkx as nx
import pandas as pd

from graph_to_vec import GraphSchema, from_networkx, from_tables, from_triples, to_networkx


def test_networkx_adapter_preserves_types_ids_edges_labels_and_masks() -> None:
    graph = nx.MultiDiGraph(label="retained")
    graph.add_node("u0", type="user", label=1, train_mask=True, activity=1.5)
    graph.add_node("u1", type="user", label=0, train_mask=False, activity=0.2)
    graph.add_node("o0", type="org", size=10.0)
    graph.add_edge("u0", "o0", relation="works_at", weight=0.7)
    graph.add_edge("u1", "o0", relation="follows", weight=0.1)

    data = from_networkx(graph)

    assert set(data.node_types) == {"user", "org"}
    assert data["user"].original_id == ["u0", "u1"]
    assert data["user"].y.tolist() == [1, 0]
    assert data["user"].train_mask.tolist() == [True, False]
    assert ("user", "works_at", "org") in data.edge_types
    assert data[("user", "works_at", "org")].edge_index.size(1) == 1

    roundtrip = to_networkx(data)
    assert set(roundtrip.nodes) == {"u0", "u1", "o0"}
    assert roundtrip.number_of_edges() == 2
    assert roundtrip.graph["label"] == "retained"


def test_from_triples_preserves_relation_types_and_labels() -> None:
    entities = pd.DataFrame(
        [
            {"id": "alice", "type": "person"},
            {"id": "acme", "type": "company"},
        ]
    )
    triples = pd.DataFrame(
        [{"source": "alice", "relation": "works_at", "target": "acme"}]
    )
    labels = pd.DataFrame([{"id": "alice", "label": "employee"}])

    data = from_triples(entities, triples, labels=labels)

    assert set(data.node_types) == {"person", "company"}
    assert ("person", "works_at", "company") in data.edge_types
    assert data["person"].original_id == ["alice"]
    assert data["person"].y.tolist() == [0]
    assert data["person"].label_names == ["employee"]


def test_from_tables_groups_graph_samples_and_labels() -> None:
    nodes = pd.DataFrame(
        [
            {"graph_id": "g0", "graph_label": "low", "id": "u0", "type": "user", "signal": 0.1},
            {"graph_id": "g0", "graph_label": "low", "id": "i0", "type": "item", "signal": 0.3},
            {"graph_id": "g1", "graph_label": "high", "id": "u1", "type": "user", "signal": 1.0},
            {"graph_id": "g1", "graph_label": "high", "id": "i1", "type": "item", "signal": 1.3},
        ]
    )
    edges = pd.DataFrame(
        [
            {"graph_id": "g0", "source": "u0", "target": "i0", "relation": "views"},
            {"graph_id": "g1", "source": "u1", "target": "i1", "relation": "buys"},
        ]
    )
    schema = GraphSchema(graph_id_col="graph_id", graph_label_col="graph_label")

    graphs = from_tables(nodes, edges, schema)

    assert isinstance(graphs, list)
    assert [graph.graph_id for graph in graphs] == ["g0", "g1"]
    assert [graph.graph_label for graph in graphs] == ["low", "high"]
    assert ("user", "views", "item") in graphs[0].edge_types
    assert ("user", "buys", "item") in graphs[1].edge_types
