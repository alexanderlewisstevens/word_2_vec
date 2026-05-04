from __future__ import annotations

import networkx as nx
import pytest

from graph_to_vec import Graph2VecTransformer, GraphClassificationPipeline
from graph_to_vec.persistence import load_joblib, save_joblib
from graph_to_vec.registry import MODEL_REGISTRY


def _graphs():
    graphs = []
    labels = []
    for idx in range(6):
        label = idx % 2
        graph = nx.MultiDiGraph(y=label)
        graph.add_node(f"u{idx}", type="user", signal=float(label))
        graph.add_node(f"i{idx}", type="item", signal=1.0)
        graph.add_edge(f"u{idx}", f"i{idx}", relation="buys" if label else "views")
        graphs.append(graph)
        labels.append(label)
    return graphs, labels


def test_joblib_artifact_reload_reproduces_predictions(tmp_path) -> None:
    graphs, labels = _graphs()
    pipeline = GraphClassificationPipeline(
        embedder=Graph2VecTransformer(iterations=1, embedding_dim=16, random_state=7)
    )
    pipeline.fit(graphs, labels)
    before = pipeline.predict(graphs)

    path = tmp_path / "model.joblib"
    save_joblib(pipeline, path)
    loaded = load_joblib(path)

    assert loaded.predict(graphs).tolist() == before.tolist()


def test_registry_exposes_models_and_reserved_hooks() -> None:
    assert "graph2vec" in MODEL_REGISTRY.available()
    assert "heterosage" in MODEL_REGISTRY.available()
    with pytest.raises(NotImplementedError):
        MODEL_REGISTRY.create("hgt")
