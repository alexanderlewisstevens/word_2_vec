from __future__ import annotations

import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from graph_to_vec import Graph2VecTransformer, GraphClassificationPipeline


def _graphs(count: int = 8) -> tuple[list[nx.MultiDiGraph], list[int]]:
    graphs = []
    labels = []
    for idx in range(count):
        label = idx % 2
        graph = nx.MultiDiGraph(label=label, y=label)
        graph.add_node(f"u{idx}:0", type="user", signal=float(label))
        graph.add_node(f"u{idx}:1", type="user", signal=0.2)
        graph.add_node(f"i{idx}:0", type="item", signal=1.0 + label)
        graph.add_edge(f"u{idx}:0", f"i{idx}:0", relation="buys" if label else "views")
        graph.add_edge(f"u{idx}:1", f"i{idx}:0", relation="views")
        graphs.append(graph)
        labels.append(label)
    return graphs, labels


def test_graph2vec_transformer_returns_fixed_width_embeddings() -> None:
    graphs, _ = _graphs()
    embeddings = Graph2VecTransformer(
        iterations=2,
        embedding_dim=16,
        random_state=7,
    ).fit_transform(graphs)

    assert embeddings.shape == (8, 16)


def test_graph_classification_pipeline_predicts_and_scores() -> None:
    graphs, labels = _graphs()
    pipeline = GraphClassificationPipeline(
        embedder=Graph2VecTransformer(iterations=2, embedding_dim=16, random_state=7)
    )

    pipeline.fit(graphs, labels)

    assert pipeline.predict(graphs).shape == (8,)
    assert 0.0 <= pipeline.score(graphs, labels) <= 1.0


def test_graph_classification_pipeline_grid_search_compatible() -> None:
    graphs, labels = _graphs()
    estimator = GraphClassificationPipeline(
        embedder=Graph2VecTransformer(embedding_dim=16, random_state=7),
        classifier=LogisticRegression(max_iter=1000),
    )
    search = GridSearchCV(estimator, {"embedder__iterations": [1, 2]}, cv=2)

    search.fit(graphs, labels)

    assert search.best_estimator_.predict(graphs).shape == (8,)
