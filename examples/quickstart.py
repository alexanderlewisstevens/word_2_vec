"""Small graph classification example."""

from __future__ import annotations

import networkx as nx

from graph_to_vec import Graph2VecTransformer, GraphClassificationPipeline


def make_graph(label: int, index: int) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(label=label, y=label, id=f"g{index}")
    graph.add_node(f"user:{index}:0", type="user", activity=1.0 + label)
    graph.add_node(f"user:{index}:1", type="user", activity=0.4)
    graph.add_node(f"item:{index}:0", type="item", price=5.0 + label)
    graph.add_edge(f"user:{index}:0", f"item:{index}:0", relation="buys" if label else "views")
    graph.add_edge(f"user:{index}:1", f"item:{index}:0", relation="views")
    return graph


def main() -> None:
    graphs = [make_graph(idx % 2, idx) for idx in range(8)]
    labels = [idx % 2 for idx in range(8)]
    pipeline = GraphClassificationPipeline(
        embedder=Graph2VecTransformer(iterations=2, embedding_dim=32, random_state=7)
    )
    pipeline.fit(graphs, labels)
    print(pipeline.predict(graphs).tolist())


if __name__ == "__main__":
    main()
