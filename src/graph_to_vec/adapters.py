"""Adapters between common graph inputs and PyG ``HeteroData``."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from graph_to_vec.schema import GraphSchema

NODE_RESERVED_ATTRS = {
    "type",
    "label",
    "y",
    "x",
    "train_mask",
    "val_mask",
    "test_mask",
}
EDGE_RESERVED_ATTRS = {"relation", "type", "edge_type", "edge_label", "y", "edge_attr"}
NUMERIC_SCALARS = (int, float, np.integer, np.floating)


@dataclass(frozen=True)
class AdapterMetadata:
    """Fitted adapter metadata stored on converted ``HeteroData`` objects."""

    node_id_maps: dict[str, dict[Any, int]]
    node_type_attr: str = "type"
    edge_type_attr: str = "relation"


def _as_feature_vector(value: Any) -> list[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().flatten().float().tolist()
    if isinstance(value, np.ndarray):
        return value.astype(float).flatten().tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(item) for item in value]
    return [float(value)]


def _feature_matrix_from_attrs(attrs: list[Mapping[str, Any]]) -> torch.Tensor:
    explicit = ["x" in item for item in attrs]
    if any(explicit):
        rows = [_as_feature_vector(item.get("x", [])) for item in attrs]
        width = max((len(row) for row in rows), default=0)
        if width == 0:
            return torch.ones((len(attrs), 1), dtype=torch.float32)
        padded = [row + [0.0] * (width - len(row)) for row in rows]
        return torch.tensor(padded, dtype=torch.float32)

    feature_keys = sorted(
        key
        for item in attrs
        for key, value in item.items()
        if key not in NODE_RESERVED_ATTRS and isinstance(value, NUMERIC_SCALARS)
    )
    if not feature_keys:
        return torch.ones((len(attrs), 1), dtype=torch.float32)

    rows = []
    for item in attrs:
        rows.append([float(item.get(key, 0.0)) for key in feature_keys])
    return torch.tensor(rows, dtype=torch.float32)


def _encode_labels(values: list[Any]) -> tuple[torch.Tensor, list[Any] | None]:
    if not values:
        return torch.empty(0, dtype=torch.long), None
    numeric = True
    encoded: list[int] = []
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            encoded.append(-1)
            continue
        try:
            encoded.append(int(value))
        except (TypeError, ValueError):
            numeric = False
            break
    if numeric:
        return torch.tensor(encoded, dtype=torch.long), None

    codes, uniques = pd.factorize(pd.Series(values, dtype="object"), sort=True)
    return torch.tensor(codes.astype(np.int64), dtype=torch.long), uniques.tolist()


def _mask_from_attrs(attrs: list[Mapping[str, Any]], key: str) -> torch.Tensor | None:
    if not any(key in item for item in attrs):
        return None
    return torch.tensor([bool(item.get(key, False)) for item in attrs], dtype=torch.bool)


def _edge_iterator(graph: nx.Graph) -> Iterable[tuple[Any, Any, Any, Mapping[str, Any]]]:
    if graph.is_multigraph():
        yield from graph.edges(keys=True, data=True)
    else:
        for idx, (source, target, attrs) in enumerate(graph.edges(data=True)):
            yield source, target, idx, attrs


def from_networkx(
    graph: nx.Graph,
    node_type_attr: str = "type",
    edge_type_attr: str = "relation",
) -> HeteroData:
    """Convert a NetworkX graph into PyG ``HeteroData``.

    Node types default to ``"node"`` and edge relations default to ``"edge"`` when the
    configured attributes are absent.
    """

    node_ids_by_type: dict[str, list[Any]] = {}
    node_attrs_by_type: dict[str, list[Mapping[str, Any]]] = {}
    node_id_maps: dict[str, dict[Any, int]] = {}
    node_type_lookup: dict[Any, str] = {}

    for node_id, attrs in graph.nodes(data=True):
        node_type = str(attrs.get(node_type_attr, "node"))
        node_type_lookup[node_id] = node_type
        node_ids_by_type.setdefault(node_type, []).append(node_id)
        node_attrs_by_type.setdefault(node_type, []).append(dict(attrs))

    data = HeteroData()
    for node_type, node_ids in node_ids_by_type.items():
        attrs = node_attrs_by_type[node_type]
        id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
        node_id_maps[node_type] = id_map

        store = data[node_type]
        store.x = _feature_matrix_from_attrs(attrs)
        store.num_nodes = len(node_ids)
        store.original_id = list(node_ids)

        label_values = [
            item.get("y", item.get("label"))
            for item in attrs
            if "y" in item or "label" in item
        ]
        if label_values:
            values = [item.get("y", item.get("label")) for item in attrs]
            store.y, label_names = _encode_labels(values)
            if label_names is not None:
                store.label_names = label_names

        for mask_name in ("train_mask", "val_mask", "test_mask"):
            mask = _mask_from_attrs(attrs, mask_name)
            if mask is not None:
                setattr(store, mask_name, mask)

    edges_by_type: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    edge_ids_by_type: dict[tuple[str, str, str], list[Any]] = {}
    edge_attrs_by_type: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}

    for edge_number, (source, target, key, attrs) in enumerate(_edge_iterator(graph)):
        source_type = node_type_lookup[source]
        target_type = node_type_lookup[target]
        relation = str(attrs.get(edge_type_attr, attrs.get("type", "edge")))
        edge_type = (source_type, relation, target_type)
        source_idx = node_id_maps[source_type][source]
        target_idx = node_id_maps[target_type][target]

        edges_by_type.setdefault(edge_type, []).append((source_idx, target_idx))
        edge_ids_by_type.setdefault(edge_type, []).append(key if key is not None else edge_number)
        edge_attrs_by_type.setdefault(edge_type, []).append(dict(attrs))

    for edge_type, pairs in edges_by_type.items():
        store = data[edge_type]
        store.edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        store.original_edge_id = edge_ids_by_type[edge_type]
        edge_attrs = edge_attrs_by_type[edge_type]
        feature_keys = sorted(
            key
            for item in edge_attrs
            for key, value in item.items()
            if key not in EDGE_RESERVED_ATTRS and isinstance(value, NUMERIC_SCALARS)
        )
        if feature_keys:
            store.edge_attr = torch.tensor(
                [[float(item.get(key, 0.0)) for key in feature_keys] for item in edge_attrs],
                dtype=torch.float32,
            )

    data.adapter_metadata = AdapterMetadata(
        node_id_maps=node_id_maps,
        node_type_attr=node_type_attr,
        edge_type_attr=edge_type_attr,
    )
    data.graph_metadata = dict(graph.graph)
    if "label" in graph.graph:
        data.graph_label = graph.graph["label"]
    if "y" in graph.graph:
        value = graph.graph["y"]
        data.y = torch.tensor([int(value)], dtype=torch.long)
    return data


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(f"expected one of columns {candidates}, found {list(columns)}")


def _entities_to_dataframe(entities: Any) -> pd.DataFrame:
    if isinstance(entities, pd.DataFrame):
        frame = entities.copy()
    elif isinstance(entities, Mapping):
        frame = pd.DataFrame(
            [{"id": entity_id, "type": entity_type} for entity_id, entity_type in entities.items()]
        )
    else:
        frame = pd.DataFrame({"id": list(entities)})
    if "id" not in frame.columns:
        id_col = _find_column(frame.columns, ["entity", "node", "name"])
        frame = frame.rename(columns={id_col: "id"})
    if "type" not in frame.columns:
        frame["type"] = "entity"
    return frame


def _triples_to_dataframe(triples: Any) -> pd.DataFrame:
    if isinstance(triples, pd.DataFrame):
        frame = triples.copy()
    else:
        frame = pd.DataFrame(list(triples), columns=["source", "relation", "target"])
    source_col = _find_column(frame.columns, ["source", "src", "head", "subject"])
    target_col = _find_column(frame.columns, ["target", "dst", "tail", "object"])
    relation_col = _find_column(frame.columns, ["relation", "rel", "predicate", "type"])
    return frame.rename(
        columns={source_col: "source", target_col: "target", relation_col: "relation"}
    )


def _labels_to_mapping(labels: Any) -> dict[Any, Any]:
    if labels is None:
        return {}
    if isinstance(labels, Mapping):
        return dict(labels)
    if isinstance(labels, pd.Series):
        return labels.to_dict()
    if isinstance(labels, pd.DataFrame):
        id_col = _find_column(labels.columns, ["id", "entity", "node"])
        label_col = _find_column(labels.columns, ["label", "y", "target"])
        return dict(zip(labels[id_col], labels[label_col], strict=False))
    return dict(labels)


def from_triples(
    entities: pd.DataFrame | Mapping[Any, str] | Iterable[Any],
    triples: pd.DataFrame | Iterable[tuple[Any, str, Any]],
    labels: Mapping[Any, Any] | pd.Series | pd.DataFrame | None = None,
) -> HeteroData:
    """Convert knowledge-graph triples into heterogeneous ``HeteroData``."""

    entity_frame = _entities_to_dataframe(entities)
    triple_frame = _triples_to_dataframe(triples)
    label_map = _labels_to_mapping(labels)
    entity_types = dict(zip(entity_frame["id"], entity_frame["type"], strict=False))

    graph = nx.MultiDiGraph()
    for row in entity_frame.to_dict("records"):
        attrs = {key: value for key, value in row.items() if key not in {"id"}}
        attrs["type"] = str(attrs.get("type", "entity"))
        if row["id"] in label_map:
            attrs["label"] = label_map[row["id"]]
        graph.add_node(row["id"], **attrs)

    for row in triple_frame.to_dict("records"):
        source = row["source"]
        target = row["target"]
        if source not in graph:
            graph.add_node(
                source,
                type=str(row.get("source_type", entity_types.get(source, "entity"))),
            )
        if target not in graph:
            graph.add_node(
                target,
                type=str(row.get("target_type", entity_types.get(target, "entity"))),
            )
        graph.add_edge(source, target, relation=str(row["relation"]))

    return from_networkx(graph)


def _select_feature_columns(
    frame: pd.DataFrame,
    configured: list[str],
    reserved: set[str],
) -> list[str]:
    if configured:
        missing = [column for column in configured if column not in frame.columns]
        if missing:
            raise ValueError(f"missing configured feature columns: {missing}")
        return configured
    return [
        column
        for column in frame.columns
        if column not in reserved and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _build_graph_from_tables(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    schema: GraphSchema,
) -> HeteroData:
    graph = nx.MultiDiGraph()
    node_reserved = {
        schema.node_id_col,
        schema.node_type_col,
        schema.graph_id_col,
        schema.graph_label_col,
        schema.node_label_col,
        schema.train_mask_col,
        schema.val_mask_col,
        schema.test_mask_col,
    }
    node_reserved = {item for item in node_reserved if item is not None}

    for row in nodes.to_dict("records"):
        node_id = row[schema.node_id_col]
        node_type = str(row.get(schema.node_type_col, "node"))
        configured_features = schema.node_feature_cols.get(node_type, [])
        feature_cols = _select_feature_columns(nodes, configured_features, node_reserved)
        attrs: dict[str, Any] = {"type": node_type}
        if feature_cols:
            attrs["x"] = [row.get(column, 0.0) for column in feature_cols]
        if (
            schema.node_label_col
            and schema.node_label_col in row
            and pd.notna(row[schema.node_label_col])
        ):
            attrs["label"] = row[schema.node_label_col]
        for mask_attr, mask_col in (
            ("train_mask", schema.train_mask_col),
            ("val_mask", schema.val_mask_col),
            ("test_mask", schema.test_mask_col),
        ):
            if mask_col and mask_col in row and pd.notna(row[mask_col]):
                attrs[mask_attr] = bool(row[mask_col])
        graph.add_node(node_id, **attrs)

    node_types = nx.get_node_attributes(graph, "type")
    edge_reserved = {
        schema.edge_source_col,
        schema.edge_target_col,
        schema.edge_type_col,
        schema.edge_source_type_col,
        schema.edge_target_type_col,
        schema.graph_id_col,
    }
    edge_reserved = {item for item in edge_reserved if item is not None}

    for row in edges.to_dict("records"):
        source = row[schema.edge_source_col]
        target = row[schema.edge_target_col]
        if source not in graph or target not in graph:
            continue
        relation = str(row.get(schema.edge_type_col, "edge"))
        configured_features = schema.edge_feature_cols.get(relation, [])
        feature_cols = _select_feature_columns(edges, configured_features, edge_reserved)
        attrs: dict[str, Any] = {"relation": relation}
        for column in feature_cols:
            attrs[column] = row.get(column, 0.0)
        graph.add_edge(source, target, **attrs)
        if schema.relation_triples:
            expected = (node_types[source], relation, node_types[target])
            if expected not in schema.relation_triples:
                raise ValueError(
                    f"edge relation {expected!r} not declared in schema.relation_triples"
                )

    data = from_networkx(graph)
    if schema.graph_id_col and schema.graph_id_col in nodes.columns and len(nodes):
        data.graph_id = nodes[schema.graph_id_col].iloc[0]
    if schema.graph_label_col:
        label_value = None
        if schema.graph_label_col in nodes.columns and len(nodes):
            labels = nodes[schema.graph_label_col].dropna().unique()
            if len(labels):
                label_value = labels[0]
        if label_value is None and schema.graph_label_col in edges.columns and len(edges):
            labels = edges[schema.graph_label_col].dropna().unique()
            if len(labels):
                label_value = labels[0]
        if label_value is not None:
            data.graph_label = label_value
            try:
                data.y = torch.tensor([int(label_value)], dtype=torch.long)
            except (TypeError, ValueError):
                pass
    return data


def from_tables(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    schema: GraphSchema | Mapping[str, Any],
) -> HeteroData | list[HeteroData]:
    """Convert node and edge tables into one graph or a list of graph samples."""

    schema = schema if isinstance(schema, GraphSchema) else GraphSchema(**schema)
    if schema.graph_id_col and schema.graph_id_col in nodes.columns:
        records = []
        for graph_id, node_group in nodes.groupby(schema.graph_id_col, sort=False):
            if schema.graph_id_col in edges.columns:
                edge_group = edges[edges[schema.graph_id_col] == graph_id]
            else:
                node_ids = set(node_group[schema.node_id_col])
                edge_group = edges[
                    edges[schema.edge_source_col].isin(node_ids)
                    & edges[schema.edge_target_col].isin(node_ids)
                ]
            records.append(_build_graph_from_tables(node_group, edge_group, schema))
        return records
    return _build_graph_from_tables(nodes, edges, schema)


def to_networkx(data: HeteroData) -> nx.MultiDiGraph:
    """Convert ``HeteroData`` back to a NetworkX ``MultiDiGraph``."""

    graph = nx.MultiDiGraph()

    for node_type in data.node_types:
        store = data[node_type]
        original_ids = getattr(store, "original_id", None)
        if original_ids is None:
            original_ids = [f"{node_type}:{idx}" for idx in range(store.num_nodes)]
        for idx, node_id in enumerate(original_ids):
            attrs: dict[str, Any] = {"type": node_type}
            if hasattr(store, "x"):
                attrs["x"] = store.x[idx].detach().cpu().tolist()
            if hasattr(store, "y"):
                attrs["y"] = int(store.y[idx].detach().cpu().item())
            for mask_name in ("train_mask", "val_mask", "test_mask"):
                if hasattr(store, mask_name):
                    attrs[mask_name] = bool(getattr(store, mask_name)[idx].detach().cpu().item())
            graph.add_node(node_id, **attrs)

    for edge_type in data.edge_types:
        source_type, relation, target_type = edge_type
        store = data[edge_type]
        source_ids = getattr(data[source_type], "original_id", None)
        target_ids = getattr(data[target_type], "original_id", None)
        if source_ids is None:
            source_ids = [f"{source_type}:{idx}" for idx in range(data[source_type].num_nodes)]
        if target_ids is None:
            target_ids = [f"{target_type}:{idx}" for idx in range(data[target_type].num_nodes)]

        edge_index = store.edge_index.detach().cpu()
        for edge_pos in range(edge_index.size(1)):
            source_idx = int(edge_index[0, edge_pos].item())
            target_idx = int(edge_index[1, edge_pos].item())
            attrs: dict[str, Any] = {
                "relation": relation,
                "source_type": source_type,
                "target_type": target_type,
            }
            if hasattr(store, "edge_attr"):
                attrs["edge_attr"] = store.edge_attr[edge_pos].detach().cpu().tolist()
            graph.add_edge(source_ids[source_idx], target_ids[target_idx], **attrs)

    if hasattr(data, "graph_metadata"):
        graph.graph.update(data.graph_metadata)
    if hasattr(data, "graph_label"):
        graph.graph["label"] = data.graph_label
    if hasattr(data, "graph_id"):
        graph.graph["id"] = data.graph_id
    return graph
