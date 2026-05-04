"""Public schema objects used by adapters, tasks, and configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GraphRecord(BaseModel):
    """One graph-level sample with optional label and metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: Any
    label: Any | None = None
    graph_id: str | int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeLabelSet(BaseModel):
    """Node classification labels and optional split masks."""

    target_node_type: str = "node"
    labels: dict[str | int, Any]
    train_mask: dict[str | int, bool] | None = None
    val_mask: dict[str | int, bool] | None = None
    test_mask: dict[str | int, bool] | None = None


class TableFeatureSpec(BaseModel):
    """Feature columns for one node or edge table."""

    columns: list[str] = Field(default_factory=list)
    fill_value: float = 0.0


class GraphSchema(BaseModel):
    """Schema for converting node and edge tables into heterogeneous graphs."""

    model_config = ConfigDict(extra="allow")

    node_id_col: str = "id"
    node_type_col: str = "type"
    edge_source_col: str = "source"
    edge_target_col: str = "target"
    edge_type_col: str = "relation"
    edge_source_type_col: str | None = "source_type"
    edge_target_type_col: str | None = "target_type"
    graph_id_col: str | None = None
    graph_label_col: str | None = None
    node_label_col: str | None = "label"
    train_mask_col: str | None = "train_mask"
    val_mask_col: str | None = "val_mask"
    test_mask_col: str | None = "test_mask"
    node_feature_cols: dict[str, list[str]] = Field(default_factory=dict)
    edge_feature_cols: dict[str, list[str]] = Field(default_factory=dict)
    node_types: list[str] = Field(default_factory=list)
    relation_triples: list[tuple[str, str, str]] = Field(default_factory=list)
    task_level: Literal["graph", "node"] = "graph"
    target_node_type: str | None = None

    @field_validator("relation_triples", mode="before")
    @classmethod
    def _normalize_relation_triples(cls, value: Any) -> Any:
        if value is None:
            return []
        triples = []
        for item in value:
            if isinstance(item, str):
                parts = tuple(part.strip() for part in item.split(","))
            else:
                parts = tuple(item)
            if len(parts) != 3:
                raise ValueError(
                    "relation_triples entries must have source type, relation, target type"
                )
            triples.append(parts)
        return triples


class RunConfig(BaseModel):
    """Lightweight YAML config model used by the CLI."""

    model_config = ConfigDict(extra="allow")

    input: dict[str, Any]
    output_dir: Path = Path("runs/default")
    model: dict[str, Any] = Field(default_factory=lambda: {"name": "graph2vec"})
    task: dict[str, Any] = Field(default_factory=lambda: {"level": "graph"})
    train: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
