"""Command line interface for repeatable graph-to-vec runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from graph_to_vec.adapters import from_tables, from_triples
from graph_to_vec.embeddings import Graph2VecTransformer, MetaPath2VecNodeEmbedder
from graph_to_vec.persistence import (
    ensure_dir,
    load_joblib,
    load_yaml,
    save_embeddings,
    save_joblib,
    save_json,
    save_yaml,
)
from graph_to_vec.pipeline import GraphClassificationPipeline
from graph_to_vec.schema import GraphSchema, RunConfig
from graph_to_vec.trainers import GraphClassifierTrainer, NodeClassifierTrainer

app = typer.Typer(no_args_is_help=True, help="Graph-to-vector classification framework.")
CONFIG_OPTION = typer.Option(..., "--config", "-c", help="YAML run config.")


def _resolve(path: str | Path, base_dir: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base_dir / path


def _load_config(path: Path) -> tuple[RunConfig, dict[str, Any]]:
    raw = load_yaml(path)
    return RunConfig(**raw), raw


def _read_csv(path: str | Path, base_dir: Path) -> pd.DataFrame:
    return pd.read_csv(_resolve(path, base_dir))


def _load_data(config: RunConfig, config_path: Path) -> Any:
    base_dir = config_path.parent
    input_config = config.input
    kind = input_config.get("kind", "tables")

    if kind == "tables":
        nodes = _read_csv(input_config["nodes_path"], base_dir)
        edges = _read_csv(input_config["edges_path"], base_dir)
        schema = GraphSchema(**input_config.get("schema", {}))
        return from_tables(nodes, edges, schema)

    if kind == "triples":
        entities = _read_csv(input_config["entities_path"], base_dir)
        triples = _read_csv(input_config["triples_path"], base_dir)
        labels = None
        if input_config.get("labels_path"):
            labels = _read_csv(input_config["labels_path"], base_dir)
        return from_triples(entities, triples, labels=labels)

    raise ValueError("input.kind must be either 'tables' or 'triples'")


def _as_graph_list(data: Any) -> list[Any]:
    return data if isinstance(data, list) else [data]


def _graph_labels(graphs: list[Any]) -> list[Any]:
    labels = []
    for idx, graph in enumerate(graphs):
        if hasattr(graph, "graph_label"):
            labels.append(graph.graph_label)
        elif hasattr(graph, "y"):
            labels.append(int(graph.y.detach().cpu().view(-1)[0].item()))
        else:
            raise ValueError(f"graph at index {idx} has no graph label")
    return labels


def _metrics(y_true: list[Any] | np.ndarray, y_pred: list[Any] | np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _node_eval_mask(store: Any) -> torch.Tensor:
    default = torch.ones_like(store.y, dtype=torch.bool)
    return getattr(store, "test_mask", getattr(store, "val_mask", default))


def _output_dir(config: RunConfig, config_path: Path) -> Path:
    return ensure_dir(_resolve(config.output_dir, config_path.parent))


def _model_artifact_path(config: RunConfig, config_path: Path, suffix: str = ".joblib") -> Path:
    output_dir = _output_dir(config, config_path)
    configured = config.artifacts.get("model_path")
    if configured:
        return _resolve(configured, config_path.parent)
    return output_dir / f"model{suffix}"


def _embedding_path(config: RunConfig, config_path: Path) -> Path:
    output_dir = _output_dir(config, config_path)
    configured = config.artifacts.get("embeddings_path")
    if configured:
        return _resolve(configured, config_path.parent)
    return output_dir / "embeddings.csv"


def _prediction_path(config: RunConfig, config_path: Path) -> Path:
    output_dir = _output_dir(config, config_path)
    configured = config.artifacts.get("predictions_path")
    if configured:
        return _resolve(configured, config_path.parent)
    return output_dir / "predictions.csv"


def _build_graph_pipeline(config: RunConfig) -> GraphClassificationPipeline:
    model_config = dict(config.model)
    model_name = model_config.pop("name", "graph2vec")
    if model_name not in {"graph2vec", "typed_wl_graph2vec"}:
        raise ValueError(
            "sklearn graph pipeline supports model.name graph2vec or typed_wl_graph2vec"
        )
    classifier_config = model_config.pop("classifier", {})
    embedder = Graph2VecTransformer(**model_config)
    classifier = LogisticRegression(max_iter=int(classifier_config.get("max_iter", 1000)))
    return GraphClassificationPipeline(embedder=embedder, classifier=classifier)


def _save_run_config(raw_config: dict[str, Any], config: RunConfig, config_path: Path) -> None:
    save_yaml(raw_config, _output_dir(config, config_path) / "config.yaml")


@app.command()
def embed(
    config: Path = CONFIG_OPTION,
) -> None:
    """Fit embeddings and write them to CSV or parquet."""

    run_config, raw_config = _load_config(config)
    data = _load_data(run_config, config)
    model_name = run_config.model.get("name", "graph2vec")

    if model_name in {"graph2vec", "typed_wl_graph2vec"}:
        graphs = _as_graph_list(data)
        model_kwargs = {key: value for key, value in run_config.model.items() if key != "name"}
        embedder = Graph2VecTransformer(**model_kwargs)
        embeddings = embedder.fit_transform(graphs)
        ids = [getattr(graph, "graph_id", idx) for idx, graph in enumerate(graphs)]
        save_embeddings(np.asarray(embeddings), _embedding_path(run_config, config), ids=ids)
        save_joblib(embedder, _model_artifact_path(run_config, config))
    elif model_name == "metapath2vec":
        model_kwargs = {key: value for key, value in run_config.model.items() if key != "name"}
        embedder = MetaPath2VecNodeEmbedder(**model_kwargs)
        embeddings = embedder.fit_transform(data)
        save_embeddings(embeddings, _embedding_path(run_config, config), ids=embedder.original_ids_)
        save_joblib(embedder, _model_artifact_path(run_config, config))
    else:
        raise ValueError(f"embedding model {model_name!r} is not supported by the CLI")

    _save_run_config(raw_config, run_config, config)
    typer.echo(f"Wrote embeddings to {_embedding_path(run_config, config)}")


@app.command()
def train(
    config: Path = CONFIG_OPTION,
) -> None:
    """Train a graph-level or node-level classifier."""

    run_config, raw_config = _load_config(config)
    data = _load_data(run_config, config)
    task_level = run_config.task.get("level", "graph")
    model_name = run_config.model.get("name", "graph2vec")

    if task_level == "graph" and model_name in {"graph2vec", "typed_wl_graph2vec"}:
        graphs = _as_graph_list(data)
        labels = _graph_labels(graphs)
        pipeline = _build_graph_pipeline(run_config)
        pipeline.fit(graphs, labels)
        predictions = pipeline.predict(graphs)
        metrics = _metrics(labels, predictions)
        save_joblib(pipeline, _model_artifact_path(run_config, config))
    elif task_level == "graph" and model_name == "heterosage":
        graphs = _as_graph_list(data)
        labels = _graph_labels(graphs)
        trainer = GraphClassifierTrainer(**run_config.train)
        trainer.fit(graphs, labels)
        metrics = trainer.evaluate(graphs, labels)
        trainer.save(_model_artifact_path(run_config, config, suffix=".pt"))
    elif task_level == "node" and model_name == "heterosage":
        target_node_type = run_config.task.get("target_node_type") or run_config.input.get(
            "schema", {}
        ).get("target_node_type", "node")
        trainer = NodeClassifierTrainer(target_node_type=target_node_type, **run_config.train)
        trainer.fit(data)
        store = data[target_node_type]
        mask = _node_eval_mask(store)
        metrics = trainer.evaluate(data, mask=mask)
        trainer.save(_model_artifact_path(run_config, config, suffix=".pt"))
    else:
        raise ValueError(f"unsupported task/model combination: {task_level}/{model_name}")

    output_dir = _output_dir(run_config, config)
    save_json(metrics, output_dir / "metrics.json")
    _save_run_config(raw_config, run_config, config)
    typer.echo(f"Wrote metrics to {output_dir / 'metrics.json'}")


@app.command()
def evaluate(
    config: Path = CONFIG_OPTION,
) -> None:
    """Evaluate a persisted classifier artifact."""

    run_config, raw_config = _load_config(config)
    data = _load_data(run_config, config)
    task_level = run_config.task.get("level", "graph")
    model_name = run_config.model.get("name", "graph2vec")

    if task_level == "graph" and model_name in {"graph2vec", "typed_wl_graph2vec"}:
        graphs = _as_graph_list(data)
        labels = _graph_labels(graphs)
        pipeline = load_joblib(_model_artifact_path(run_config, config))
        metrics = _metrics(labels, pipeline.predict(graphs))
    elif task_level == "graph" and model_name == "heterosage":
        graphs = _as_graph_list(data)
        labels = _graph_labels(graphs)
        trainer = GraphClassifierTrainer.load(
            _model_artifact_path(run_config, config, suffix=".pt")
        )
        metrics = trainer.evaluate(graphs, labels)
    elif task_level == "node" and model_name == "heterosage":
        target_node_type = run_config.task.get("target_node_type") or run_config.input.get(
            "schema", {}
        ).get("target_node_type", "node")
        trainer = NodeClassifierTrainer.load(
            _model_artifact_path(run_config, config, suffix=".pt")
        )
        store = data[target_node_type]
        mask = _node_eval_mask(store)
        metrics = trainer.evaluate(data, mask=mask)
    else:
        raise ValueError(f"unsupported task/model combination: {task_level}/{model_name}")

    output_dir = _output_dir(run_config, config)
    save_json(metrics, output_dir / "metrics.json")
    _save_run_config(raw_config, run_config, config)
    typer.echo(f"Wrote metrics to {output_dir / 'metrics.json'}")


@app.command()
def predict(
    config: Path = CONFIG_OPTION,
) -> None:
    """Write predictions from a persisted classifier artifact."""

    run_config, _ = _load_config(config)
    data = _load_data(run_config, config)
    task_level = run_config.task.get("level", "graph")
    model_name = run_config.model.get("name", "graph2vec")

    if task_level == "graph" and model_name in {"graph2vec", "typed_wl_graph2vec"}:
        graphs = _as_graph_list(data)
        model = load_joblib(_model_artifact_path(run_config, config))
        predictions = model.predict(graphs)
        ids = [getattr(graph, "graph_id", idx) for idx, graph in enumerate(graphs)]
        output = pd.DataFrame({"graph_id": ids, "prediction": predictions})
    elif task_level == "graph" and model_name == "heterosage":
        graphs = _as_graph_list(data)
        trainer = GraphClassifierTrainer.load(
            _model_artifact_path(run_config, config, suffix=".pt")
        )
        predictions = trainer.predict(graphs)
        ids = [getattr(graph, "graph_id", idx) for idx, graph in enumerate(graphs)]
        output = pd.DataFrame({"graph_id": ids, "prediction": predictions})
    elif task_level == "node" and model_name == "heterosage":
        target_node_type = run_config.task.get("target_node_type") or run_config.input.get(
            "schema", {}
        ).get("target_node_type", "node")
        trainer = NodeClassifierTrainer.load(
            _model_artifact_path(run_config, config, suffix=".pt")
        )
        predictions = trainer.predict(data)
        ids = getattr(data[target_node_type], "original_id", list(range(len(predictions))))
        output = pd.DataFrame({"node_id": ids, "prediction": predictions})
    else:
        raise ValueError(f"unsupported task/model combination: {task_level}/{model_name}")

    path = _prediction_path(run_config, config)
    ensure_dir(path.parent)
    output.to_csv(path, index=False)
    typer.echo(f"Wrote predictions to {path}")
