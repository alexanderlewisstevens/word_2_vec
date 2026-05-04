"""Model registry and reserved extension hooks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from graph_to_vec.embeddings import Graph2VecTransformer, MetaPath2VecNodeEmbedder, TypedWLGraph2Vec
from graph_to_vec.models import HeteroSAGEClassifier, RGCNClassifier


class ModelRegistry:
    """Small callable registry used by the CLI and extension points."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Any]] = {}
        self._reserved: dict[str, str] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        self._factories[name] = factory
        self._reserved.pop(name, None)

    def reserve(self, name: str, message: str) -> None:
        self._reserved[name] = message

    def create(self, name: str, **kwargs: Any) -> Any:
        if name in self._reserved:
            raise NotImplementedError(self._reserved[name])
        if name not in self._factories:
            known = sorted([*self._factories.keys(), *self._reserved.keys()])
            raise KeyError(f"unknown model {name!r}; known models: {known}")
        return self._factories[name](**kwargs)

    def available(self) -> list[str]:
        return sorted(self._factories)

    def reserved(self) -> dict[str, str]:
        return dict(self._reserved)


MODEL_REGISTRY = ModelRegistry()
MODEL_REGISTRY.register("graph2vec", Graph2VecTransformer)
MODEL_REGISTRY.register("typed_wl_graph2vec", TypedWLGraph2Vec)
MODEL_REGISTRY.register("metapath2vec", MetaPath2VecNodeEmbedder)
MODEL_REGISTRY.register("heterosage", HeteroSAGEClassifier)
MODEL_REGISTRY.register("rgcn", RGCNClassifier)
MODEL_REGISTRY.reserve("hgt", "HGT is reserved for a later phase.")
MODEL_REGISTRY.reserve("graphmae", "GraphMAE-style self-supervised pretraining is reserved.")
MODEL_REGISTRY.reserve("graphgps", "GraphGPS-style graph transformers are reserved.")
