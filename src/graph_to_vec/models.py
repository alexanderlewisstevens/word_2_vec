"""PyTorch Geometric model baselines."""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, RGCNConv, SAGEConv, global_mean_pool
from torch_geometric.nn import Linear as PygLinear


def _module_key(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", value)


class HeteroSAGEClassifier(nn.Module):
    """Heterogeneous GraphSAGE for node or graph classification."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]] | None = None,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        task: str = "node",
        target_node_type: str | None = None,
        aggr: str = "sum",
    ) -> None:
        super().__init__()
        if task not in {"node", "graph"}:
            raise ValueError("task must be 'node' or 'graph'")
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.task = task
        self.target_node_type = target_node_type
        self.aggr = aggr

        self.input_proj = nn.ModuleDict()
        self.convs = nn.ModuleList()
        self.node_heads = nn.ModuleDict()
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

        if metadata is not None:
            self._ensure_modules(metadata)

    def _ensure_modules(self, metadata: tuple[list[str], list[tuple[str, str, str]]]) -> None:
        node_types, edge_types = metadata
        for node_type in node_types:
            key = _module_key(node_type)
            if key not in self.input_proj:
                self.input_proj[key] = PygLinear(-1, self.hidden_channels)
            if key not in self.node_heads:
                self.node_heads[key] = nn.Linear(self.hidden_channels, self.out_channels)

        if edge_types and not self.convs:
            for _ in range(self.num_layers):
                self.convs.append(
                    HeteroConv(
                        {
                            edge_type: SAGEConv(
                                (-1, -1),
                                self.hidden_channels,
                                normalize=True,
                            )
                            for edge_type in edge_types
                        },
                        aggr=self.aggr,
                    )
                )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        batch_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
        self._ensure_modules(metadata)

        h_dict = {
            node_type: F.relu(self.input_proj[_module_key(node_type)](x.float()))
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            conv_out = conv(h_dict, edge_index_dict)
            next_h = {}
            for node_type, previous in h_dict.items():
                updated = conv_out.get(node_type, previous)
                next_h[node_type] = F.dropout(
                    F.relu(updated),
                    p=self.dropout,
                    training=self.training,
                )
            h_dict = next_h

        if self.task == "node":
            target = self.target_node_type or next(iter(h_dict))
            return self.node_heads[_module_key(target)](h_dict[target])

        pooled = []
        num_graphs = None
        if batch_dict:
            num_graphs = max(
                (int(batch.max().item()) + 1 for batch in batch_dict.values() if batch.numel()),
                default=1,
            )
        for node_type, hidden in h_dict.items():
            batch = batch_dict.get(node_type) if batch_dict else None
            if batch is None:
                pooled.append(hidden.mean(dim=0, keepdim=True))
            else:
                pooled.append(global_mean_pool(hidden, batch, size=num_graphs))
        graph_embedding = torch.stack(pooled, dim=0).sum(dim=0)
        return self.graph_head(graph_embedding)


class RGCNClassifier(nn.Module):
    """R-GCN entity classification baseline over flattened relation-typed graphs."""

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = max(1, num_relations)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.embedding = nn.Embedding(num_nodes, hidden_channels)
        self.input_proj = PygLinear(-1, hidden_channels)
        self.convs = nn.ModuleList(
            RGCNConv(hidden_channels, hidden_channels, self.num_relations)
            for _ in range(num_layers)
        )
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        x: torch.Tensor | None = None,
        node_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x is None:
            hidden = self.embedding.weight
        else:
            hidden = F.relu(self.input_proj(x.float()))
        for conv in self.convs:
            hidden = conv(hidden, edge_index, edge_type)
            hidden = F.dropout(F.relu(hidden), p=self.dropout, training=self.training)
        logits = self.head(hidden)
        return logits if node_index is None else logits[node_index]


def heterodata_to_rgcn_inputs(
    data: HeteroData,
    target_node_type: str | None = None,
) -> dict[str, Any]:
    """Flatten ``HeteroData`` edge stores into homogeneous R-GCN tensors."""

    node_offsets: dict[str, int] = {}
    node_slices: dict[str, slice] = {}
    offset = 0
    for node_type in data.node_types:
        count = int(data[node_type].num_nodes)
        node_offsets[node_type] = offset
        node_slices[node_type] = slice(offset, offset + count)
        offset += count

    edge_indices = []
    edge_types = []
    relation_to_id = {edge_type: idx for idx, edge_type in enumerate(data.edge_types)}
    for edge_type, relation_id in relation_to_id.items():
        source_type, _, target_type = edge_type
        edge_index = data[edge_type].edge_index.detach().clone()
        edge_index[0] += node_offsets[source_type]
        edge_index[1] += node_offsets[target_type]
        edge_indices.append(edge_index)
        edge_types.append(torch.full((edge_index.size(1),), relation_id, dtype=torch.long))

    if edge_indices:
        edge_index_tensor = torch.cat(edge_indices, dim=1).long()
        edge_type_tensor = torch.cat(edge_types, dim=0).long()
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_type_tensor = torch.empty((0,), dtype=torch.long)

    target_node_index = None
    if target_node_type is not None:
        target_slice = node_slices[target_node_type]
        target_node_index = torch.arange(target_slice.start, target_slice.stop, dtype=torch.long)

    return {
        "num_nodes": offset,
        "num_relations": max(1, len(relation_to_id)),
        "edge_index": edge_index_tensor,
        "edge_type": edge_type_tensor,
        "node_slices": node_slices,
        "node_offsets": node_offsets,
        "relation_to_id": relation_to_id,
        "target_node_index": target_node_index,
    }
