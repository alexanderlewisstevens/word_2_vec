from __future__ import annotations

import networkx as nx

from graph_to_vec import from_networkx
from graph_to_vec.models import HeteroSAGEClassifier, RGCNClassifier, heterodata_to_rgcn_inputs
from graph_to_vec.trainers import GraphClassifierTrainer, NodeClassifierTrainer


def _node_graph():
    graph = nx.MultiDiGraph()
    for idx in range(6):
        label = idx % 2
        graph.add_node(
            f"u{idx}",
            type="user",
            label=label,
            train_mask=True,
            signal=float(label),
        )
    graph.add_node("i0", type="item", signal=0.0)
    graph.add_node("i1", type="item", signal=1.0)
    for idx in range(6):
        graph.add_edge(f"u{idx}", f"i{idx % 2}", relation="interacts")
    return from_networkx(graph)


def _graph_sample(label: int, idx: int):
    graph = nx.MultiDiGraph(y=label)
    graph.add_node(f"u{idx}", type="user", signal=float(label))
    graph.add_node(f"i{idx}", type="item", signal=1.0)
    graph.add_edge(f"u{idx}", f"i{idx}", relation="buys" if label else "views")
    return from_networkx(graph)


def test_heterosage_node_and_graph_forward_shapes() -> None:
    data = _node_graph()
    node_model = HeteroSAGEClassifier(
        metadata=data.metadata(),
        hidden_channels=8,
        out_channels=2,
        task="node",
        target_node_type="user",
    )
    graph_model = HeteroSAGEClassifier(
        metadata=data.metadata(),
        hidden_channels=8,
        out_channels=2,
        task="graph",
    )

    assert node_model(data.x_dict, data.edge_index_dict).shape == (6, 2)
    assert graph_model(data.x_dict, data.edge_index_dict).shape == (1, 2)


def test_rgcn_forward_shape_from_heterodata() -> None:
    data = _node_graph()
    inputs = heterodata_to_rgcn_inputs(data, target_node_type="user")
    model = RGCNClassifier(
        num_nodes=inputs["num_nodes"],
        num_relations=inputs["num_relations"],
        out_channels=2,
        hidden_channels=8,
    )

    logits = model(
        inputs["edge_index"],
        inputs["edge_type"],
        node_index=inputs["target_node_index"],
    )

    assert logits.shape == (6, 2)


def test_node_trainer_smoke_loss_decreases() -> None:
    data = _node_graph()
    trainer = NodeClassifierTrainer(
        target_node_type="user",
        epochs=12,
        lr=0.05,
        device="cpu",
    )

    trainer.fit(data)

    assert trainer.history_["loss"][-1] <= trainer.history_["loss"][0]


def test_graph_trainer_smoke_predicts() -> None:
    graphs = [_graph_sample(idx % 2, idx) for idx in range(6)]
    labels = [idx % 2 for idx in range(6)]
    trainer = GraphClassifierTrainer(epochs=3, lr=0.02, batch_size=3, device="cpu")

    trainer.fit(graphs, labels)

    assert trainer.predict(graphs).shape == (6,)
