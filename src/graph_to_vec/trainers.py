"""Training helpers for PyG classification baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from graph_to_vec.models import HeteroSAGEClassifier


def _device(value: str | torch.device | None = None) -> torch.device:
    if value is not None:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _num_classes(labels: torch.Tensor) -> int:
    valid = labels[labels >= 0]
    if valid.numel() == 0:
        raise ValueError("labels must contain at least one non-negative class")
    return int(valid.max().item()) + 1


def _default_mask(labels: torch.Tensor) -> torch.Tensor:
    return labels >= 0


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _union_metadata(graphs: list[HeteroData]) -> tuple[list[str], list[tuple[str, str, str]]]:
    node_types: list[str] = []
    edge_types: list[tuple[str, str, str]] = []
    for graph in graphs:
        for node_type in graph.node_types:
            if node_type not in node_types:
                node_types.append(node_type)
        for edge_type in graph.edge_types:
            if edge_type not in edge_types:
                edge_types.append(edge_type)
    return node_types, edge_types


def _tensor_batchable_copy(data: HeteroData) -> HeteroData:
    copy = data.clone()
    for store in copy.stores:
        for key, value in list(store.items()):
            if key == "num_nodes":
                continue
            if not torch.is_tensor(value):
                del store[key]
    return copy


class NodeClassifierTrainer:
    """Small-batch trainer for node classification on one ``HeteroData`` graph."""

    def __init__(
        self,
        model: HeteroSAGEClassifier | None = None,
        target_node_type: str = "node",
        lr: float = 0.01,
        weight_decay: float = 0.0,
        epochs: int = 25,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.target_node_type = target_node_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device

    def fit(
        self,
        data: HeteroData,
        labels: torch.Tensor | None = None,
        train_mask: torch.Tensor | None = None,
        val_mask: torch.Tensor | None = None,
    ) -> NodeClassifierTrainer:
        device = _device(self.device)
        labels = labels if labels is not None else data[self.target_node_type].y
        train_mask = (
            train_mask
            if train_mask is not None
            else getattr(data[self.target_node_type], "train_mask", _default_mask(labels))
        )
        out_channels = _num_classes(labels)

        self.model_ = self.model or HeteroSAGEClassifier(
            metadata=data.metadata(),
            out_channels=out_channels,
            task="node",
            target_node_type=self.target_node_type,
        )
        if hasattr(self.model_, "_ensure_modules"):
            self.model_._ensure_modules(data.metadata())
        self.model_.to(device)
        data = data.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device) if val_mask is not None else None

        with torch.no_grad():
            self.model_(data.x_dict, data.edge_index_dict)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.history_ = {"loss": [], "val_accuracy": []}
        for _ in range(self.epochs):
            self.model_.train()
            optimizer.zero_grad()
            logits = self.model_(data.x_dict, data.edge_index_dict)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()
            self.history_["loss"].append(float(loss.detach().cpu().item()))

            if val_mask is not None and bool(val_mask.any()):
                self.model_.eval()
                with torch.no_grad():
                    pred = logits[val_mask].argmax(dim=-1).detach().cpu().numpy()
                    truth = labels[val_mask].detach().cpu().numpy()
                    self.history_["val_accuracy"].append(float(accuracy_score(truth, pred)))

        self.classes_ = np.arange(out_channels)
        self.target_node_type_ = self.target_node_type
        return self

    def predict(self, data: HeteroData, mask: torch.Tensor | None = None) -> np.ndarray:
        self.model_.eval()
        device = next(self.model_.parameters()).device
        data = data.to(device)
        with torch.no_grad():
            logits = self.model_(data.x_dict, data.edge_index_dict)
            if mask is not None:
                logits = logits[mask.to(device)]
            return logits.argmax(dim=-1).detach().cpu().numpy()

    def predict_proba(self, data: HeteroData, mask: torch.Tensor | None = None) -> np.ndarray:
        self.model_.eval()
        device = next(self.model_.parameters()).device
        data = data.to(device)
        with torch.no_grad():
            logits = self.model_(data.x_dict, data.edge_index_dict)
            if mask is not None:
                logits = logits[mask.to(device)]
            return logits.softmax(dim=-1).detach().cpu().numpy()

    def evaluate(
        self,
        data: HeteroData,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        labels = labels if labels is not None else data[self.target_node_type].y
        mask = mask if mask is not None else _default_mask(labels)
        pred = self.predict(data, mask=mask)
        truth = labels[mask].detach().cpu().numpy()
        return _classification_metrics(truth, pred)

    def save(self, path: str | Path) -> None:
        payload = {
            "model": self.model_,
            "target_node_type": self.target_node_type_,
            "history": self.history_,
            "classes": self.classes_,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> NodeClassifierTrainer:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        trainer = cls(model=payload["model"], target_node_type=payload["target_node_type"])
        trainer.model_ = payload["model"]
        trainer.history_ = payload.get("history", {})
        trainer.classes_ = payload.get("classes")
        trainer.target_node_type_ = payload["target_node_type"]
        return trainer


class GraphClassifierTrainer:
    """Mini-batch trainer for graph classification over ``HeteroData`` samples."""

    def __init__(
        self,
        model: HeteroSAGEClassifier | None = None,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        epochs: int = 25,
        batch_size: int = 16,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def _prepare_dataset(
        self,
        graphs: list[HeteroData],
        labels: list[Any] | np.ndarray | None = None,
        fit_encoder: bool = True,
    ) -> list[HeteroData]:
        if labels is None:
            labels = [
                int(graph.y.detach().cpu().view(-1)[0].item())
                if hasattr(graph, "y")
                else graph.graph_label
                for graph in graphs
            ]
        if fit_encoder:
            self.label_encoder_ = LabelEncoder()
            encoded = self.label_encoder_.fit_transform(labels)
        else:
            encoded = self.label_encoder_.transform(labels)

        dataset = []
        for graph, label in zip(graphs, encoded, strict=False):
            item = _tensor_batchable_copy(graph)
            item.y = torch.tensor([int(label)], dtype=torch.long)
            dataset.append(item)
        return dataset

    def fit(
        self,
        graphs: list[HeteroData],
        labels: list[Any] | np.ndarray | None = None,
    ) -> GraphClassifierTrainer:
        if not graphs:
            raise ValueError("graphs must be non-empty")
        dataset = self._prepare_dataset(graphs, labels, fit_encoder=True)
        metadata = _union_metadata(dataset)
        out_channels = len(self.label_encoder_.classes_)
        self.model_ = self.model or HeteroSAGEClassifier(
            metadata=metadata,
            out_channels=out_channels,
            task="graph",
        )
        if hasattr(self.model_, "_ensure_modules"):
            self.model_._ensure_modules(metadata)
        device = _device(self.device)
        self.model_.to(device)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        first_batch = next(iter(DataLoader(dataset, batch_size=min(len(dataset), self.batch_size))))
        first_batch = first_batch.to(device)
        with torch.no_grad():
            self.model_(first_batch.x_dict, first_batch.edge_index_dict, first_batch.batch_dict)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.history_ = {"loss": []}
        for _ in range(self.epochs):
            self.model_.train()
            epoch_losses = []
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = self.model_(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
                loss = F.cross_entropy(logits, batch.y.view(-1))
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            self.history_["loss"].append(float(np.mean(epoch_losses)))
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, graphs: list[HeteroData]) -> np.ndarray:
        dataset = self._prepare_dataset(
            graphs,
            labels=[self.classes_[0] for _ in graphs],
            fit_encoder=False,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        device = next(self.model_.parameters()).device
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = self.model_(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
                preds.extend(logits.argmax(dim=-1).detach().cpu().numpy().tolist())
        return self.label_encoder_.inverse_transform(np.asarray(preds, dtype=int))

    def evaluate(
        self,
        graphs: list[HeteroData],
        labels: list[Any] | np.ndarray | None = None,
    ) -> dict[str, float]:
        if labels is None:
            labels = [
                int(graph.y.detach().cpu().view(-1)[0].item())
                if hasattr(graph, "y")
                else graph.graph_label
                for graph in graphs
            ]
        pred = self.predict(graphs)
        return _classification_metrics(np.asarray(labels), np.asarray(pred))

    def save(self, path: str | Path) -> None:
        payload = {
            "model": self.model_,
            "history": self.history_,
            "classes": self.classes_,
            "label_encoder": self.label_encoder_,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> GraphClassifierTrainer:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        trainer = cls(model=payload["model"])
        trainer.model_ = payload["model"]
        trainer.history_ = payload.get("history", {})
        trainer.classes_ = payload.get("classes")
        trainer.label_encoder_ = payload.get("label_encoder")
        return trainer
