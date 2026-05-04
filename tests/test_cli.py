from __future__ import annotations

import pandas as pd
import yaml
from typer.testing import CliRunner

from graph_to_vec.cli import app


def _write_cli_fixture(tmp_path):
    nodes = []
    edges = []
    for idx in range(6):
        label = "high" if idx % 2 else "low"
        nodes.extend(
            [
                {
                    "graph_id": f"g{idx}",
                    "graph_label": label,
                    "id": f"u{idx}",
                    "type": "user",
                    "signal": float(idx % 2),
                },
                {
                    "graph_id": f"g{idx}",
                    "graph_label": label,
                    "id": f"i{idx}",
                    "type": "item",
                    "signal": 1.0,
                },
            ]
        )
        edges.append(
            {
                "graph_id": f"g{idx}",
                "source": f"u{idx}",
                "target": f"i{idx}",
                "relation": "buys" if idx % 2 else "views",
            }
        )

    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"
    pd.DataFrame(nodes).to_csv(nodes_path, index=False)
    pd.DataFrame(edges).to_csv(edges_path, index=False)
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
            "iterations": 1,
            "embedding_dim": 16,
            "random_state": 7,
        },
        "output_dir": str(tmp_path / "run"),
        "artifacts": {
            "model_path": str(tmp_path / "run" / "model.joblib"),
            "embeddings_path": str(tmp_path / "run" / "embeddings.csv"),
            "predictions_path": str(tmp_path / "run" / "predictions.csv"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_cli_train_evaluate_predict_and_embed(tmp_path) -> None:
    config_path = _write_cli_fixture(tmp_path)
    runner = CliRunner()

    train_result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert train_result.exit_code == 0, train_result.output
    assert (tmp_path / "run" / "metrics.json").exists()
    assert (tmp_path / "run" / "model.joblib").exists()

    evaluate_result = runner.invoke(app, ["evaluate", "--config", str(config_path)])
    assert evaluate_result.exit_code == 0, evaluate_result.output

    predict_result = runner.invoke(app, ["predict", "--config", str(config_path)])
    assert predict_result.exit_code == 0, predict_result.output
    assert (tmp_path / "run" / "predictions.csv").exists()

    embed_result = runner.invoke(app, ["embed", "--config", str(config_path)])
    assert embed_result.exit_code == 0, embed_result.output
    assert (tmp_path / "run" / "embeddings.csv").exists()
