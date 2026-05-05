"""Classical graph and node embedding baselines."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.preprocessing import normalize
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils.validation import check_is_fitted
from torch_geometric.data import HeteroData

from graph_to_vec.adapters import from_networkx
from graph_to_vec.schema import GraphRecord


def _stable_digest(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()


def _ensure_heterodata(graph: HeteroData | nx.Graph | GraphRecord) -> HeteroData:
    if isinstance(graph, HeteroData):
        return graph
    if isinstance(graph, GraphRecord):
        return _ensure_heterodata(graph.graph)
    if isinstance(graph, nx.Graph):
        data = from_networkx(graph)
        if isinstance(graph, nx.Graph) and graph.graph:
            data.graph_metadata = dict(graph.graph)
        return data
    raise TypeError(f"unsupported graph input type: {type(graph)!r}")


def _node_feature_token(store: Any, idx: int, precision: int = 4) -> str:
    if not hasattr(store, "x"):
        return ""
    values = store.x[idx].detach().cpu().flatten().tolist()
    rounded = ",".join(f"{float(value):.{precision}g}" for value in values)
    return f"|x={rounded}"


def _wl_counter(
    data: HeteroData,
    iterations: int,
    include_features: bool,
    feature_precision: int,
) -> Counter[str]:
    labels: dict[tuple[str, int], str] = {}
    counter: Counter[str] = Counter()

    for node_type in data.node_types:
        store = data[node_type]
        for idx in range(store.num_nodes):
            feature_part = (
                _node_feature_token(store, idx, feature_precision) if include_features else ""
            )
            label = f"{node_type}{feature_part}"
            labels[(node_type, idx)] = _stable_digest(label)
            counter[f"wl0:{node_type}:{labels[(node_type, idx)]}"] += 1

    for edge_type in data.edge_types:
        source_type, relation, target_type = edge_type
        edge_index = data[edge_type].edge_index
        counter[f"edge:{source_type}:{relation}:{target_type}"] += int(edge_index.size(1))

    for iteration in range(1, iterations + 1):
        neighbors: dict[tuple[str, int], list[str]] = defaultdict(list)
        for source_type, relation, target_type in data.edge_types:
            edge_index = data[(source_type, relation, target_type)].edge_index.detach().cpu()
            for edge_pos in range(edge_index.size(1)):
                source_idx = int(edge_index[0, edge_pos].item())
                target_idx = int(edge_index[1, edge_pos].item())
                source_key = (source_type, source_idx)
                target_key = (target_type, target_idx)
                neighbors[source_key].append(
                    f"out:{relation}:{target_type}:{labels.get(target_key, '')}"
                )
                neighbors[target_key].append(
                    f"in:{relation}:{source_type}:{labels.get(source_key, '')}"
                )

        next_labels: dict[tuple[str, int], str] = {}
        for node_type in data.node_types:
            store = data[node_type]
            for idx in range(store.num_nodes):
                key = (node_type, idx)
                payload = labels[key] + "|" + "|".join(sorted(neighbors.get(key, [])))
                label = _stable_digest(payload)
                next_labels[key] = label
                counter[f"wl{iteration}:{node_type}:{label}"] += 1
        labels = next_labels

    return counter


class Graph2VecTransformer(BaseEstimator, TransformerMixin):
    """Typed Weisfeiler-Lehman graph embeddings with sklearn compatibility.

    This is a pragmatic Graph2Vec-style baseline: it extracts heterogeneous WL subtree tokens and
    turns them into graph vectors via hashing, vocabulary counts, or random projection.
    """

    def __init__(
        self,
        iterations: int = 2,
        embedding_dim: int | None = 128,
        include_features: bool = True,
        projection: str = "hash",
        normalize_output: bool = True,
        min_df: int = 1,
        max_features: int | None = None,
        feature_precision: int = 4,
        random_state: int | None = None,
    ) -> None:
        self.iterations = iterations
        self.embedding_dim = embedding_dim
        self.include_features = include_features
        self.projection = projection
        self.normalize_output = normalize_output
        self.min_df = min_df
        self.max_features = max_features
        self.feature_precision = feature_precision
        self.random_state = random_state

    def _counters(
        self,
        graphs: Iterable[HeteroData | nx.Graph | GraphRecord],
    ) -> list[Counter[str]]:
        return [
            _wl_counter(
                _ensure_heterodata(graph),
                iterations=self.iterations,
                include_features=self.include_features,
                feature_precision=self.feature_precision,
            )
            for graph in graphs
        ]

    def fit(
        self,
        X: Iterable[HeteroData | nx.Graph | GraphRecord],
        y: Any = None,
    ) -> Graph2VecTransformer:
        counters = self._counters(X)
        self.projection_ = self.projection

        if self.projection == "hash":
            if self.embedding_dim is None:
                raise ValueError("embedding_dim is required when projection='hash'")
            self.hasher_ = FeatureHasher(
                n_features=self.embedding_dim,
                input_type="dict",
                alternate_sign=False,
            )
        elif self.projection in {"vocabulary", "random_projection"}:
            self.vectorizer_ = DictVectorizer(sparse=True)
            matrix = self.vectorizer_.fit_transform(counters)
            if self.min_df > 1:
                document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
                keep = document_frequency >= self.min_df
                self._kept_features_ = keep
                matrix = matrix[:, keep]
            else:
                self._kept_features_ = None
            if self.max_features is not None and matrix.shape[1] > self.max_features:
                sums = np.asarray(matrix.sum(axis=0)).ravel()
                keep_indices = np.argsort(sums)[-self.max_features :]
                self._max_feature_indices_ = np.sort(keep_indices)
                matrix = matrix[:, self._max_feature_indices_]
            else:
                self._max_feature_indices_ = None
            if self.projection == "random_projection" and self.embedding_dim is not None:
                self.projector_ = SparseRandomProjection(
                    n_components=self.embedding_dim,
                    random_state=self.random_state,
                )
                self.projector_.fit(matrix)
        else:
            raise ValueError(
                "projection must be one of 'hash', 'vocabulary', or 'random_projection'"
            )
        self.n_features_in_ = None
        self.is_fitted_ = True
        return self

    def transform(
        self,
        X: Iterable[HeteroData | nx.Graph | GraphRecord],
    ) -> np.ndarray | sp.csr_matrix:
        check_is_fitted(self, "is_fitted_")
        counters = self._counters(X)

        if self.projection_ == "hash":
            matrix = self.hasher_.transform(counters)
        else:
            matrix = self.vectorizer_.transform(counters)
            if self._kept_features_ is not None:
                matrix = matrix[:, self._kept_features_]
            if self._max_feature_indices_ is not None:
                matrix = matrix[:, self._max_feature_indices_]
            if self.projection_ == "random_projection" and hasattr(self, "projector_"):
                matrix = self.projector_.transform(matrix)

        if self.normalize_output:
            matrix = normalize(matrix, norm="l2", copy=False)
        if self.projection_ == "vocabulary":
            return matrix
        return np.asarray(matrix.toarray() if sp.issparse(matrix) else matrix, dtype=np.float32)

    def fit_transform(
        self,
        X: Iterable[HeteroData | nx.Graph | GraphRecord],
        y: Any = None,
        **fit_params: Any,
    ) -> np.ndarray | sp.csr_matrix:
        return self.fit(X, y).transform(X)

    def infer(self, X: Iterable[HeteroData | nx.Graph | GraphRecord]) -> np.ndarray | sp.csr_matrix:
        """Alias for ``transform`` for users coming from embedding APIs."""

        return self.transform(X)

    def get_feature_names_out(self) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        if self.projection_ == "hash":
            return np.asarray(
                [f"hash_{idx}" for idx in range(self.embedding_dim or 0)],
                dtype=object,
            )
        names = self.vectorizer_.get_feature_names_out()
        if self._kept_features_ is not None:
            names = names[self._kept_features_]
        if self._max_feature_indices_ is not None:
            names = names[self._max_feature_indices_]
        if self.projection_ == "random_projection" and self.embedding_dim is not None:
            return np.asarray([f"rp_{idx}" for idx in range(self.embedding_dim)], dtype=object)
        return names


class TypedWLGraph2Vec(Graph2VecTransformer):
    """Named alias for the heterogeneous WL/Graph2Vec baseline."""


def _normalize_metapath(
    metapath: Sequence[str | tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    normalized = []
    for edge_type in metapath:
        if isinstance(edge_type, str):
            parts = tuple(part.strip() for part in edge_type.split(","))
        else:
            parts = tuple(edge_type)
        if len(parts) != 3:
            raise ValueError("metapath entries must be edge type triples")
        normalized.append(parts)  # type: ignore[arg-type]
    return normalized


class MetaPath2VecNodeEmbedder(BaseEstimator, TransformerMixin):
    """Simple metapath/random-walk node embeddings for heterogeneous graphs.

    The implementation intentionally avoids a heavyweight training loop in V1. It samples typed
    walks, builds a skip-gram-style co-occurrence matrix, and projects it with SVD.
    """

    def __init__(
        self,
        metapaths: Sequence[Sequence[str | tuple[str, str, str]]] | None = None,
        walk_length: int = 12,
        walks_per_node: int = 8,
        context_size: int = 3,
        embedding_dim: int = 64,
        random_state: int | None = None,
    ) -> None:
        self.metapaths = metapaths
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.random_state = random_state

    def _build_adjacency(
        self, data: HeteroData
    ) -> dict[tuple[str, str, str], dict[int, list[int]]]:
        adjacency: dict[tuple[str, str, str], dict[int, list[int]]] = {}
        for edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index.detach().cpu()
            by_source: dict[int, list[int]] = defaultdict(list)
            for edge_pos in range(edge_index.size(1)):
                by_source[int(edge_index[0, edge_pos].item())].append(
                    int(edge_index[1, edge_pos].item())
                )
            adjacency[edge_type] = by_source
        return adjacency

    def _sample_walks(self, data: HeteroData) -> list[list[tuple[str, int]]]:
        rng = random.Random(self.random_state)
        adjacency = self._build_adjacency(data)
        edge_types_by_source: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for edge_type in data.edge_types:
            edge_types_by_source[edge_type[0]].append(edge_type)

        metapaths = (
            [_normalize_metapath(path) for path in self.metapaths]
            if self.metapaths
            else None
        )
        walks: list[list[tuple[str, int]]] = []

        for start_type in data.node_types:
            for start_idx in range(data[start_type].num_nodes):
                for walk_number in range(self.walks_per_node):
                    current = (start_type, start_idx)
                    walk = [current]
                    if metapaths:
                        path = metapaths[walk_number % len(metapaths)]
                        steps = [path[idx % len(path)] for idx in range(self.walk_length)]
                    else:
                        steps = []

                    for step_idx in range(self.walk_length):
                        if metapaths:
                            edge_type = steps[step_idx]
                            if edge_type[0] != current[0]:
                                break
                        else:
                            candidates = edge_types_by_source.get(current[0], [])
                            if not candidates:
                                break
                            edge_type = rng.choice(candidates)
                        targets = adjacency.get(edge_type, {}).get(current[1], [])
                        if not targets:
                            break
                        current = (edge_type[2], rng.choice(targets))
                        walk.append(current)
                    if len(walk) > 1:
                        walks.append(walk)
        return walks

    def fit(self, X: HeteroData | nx.Graph, y: Any = None) -> MetaPath2VecNodeEmbedder:
        data = _ensure_heterodata(X)
        walks = self._sample_walks(data)
        tokens = [
            (node_type, idx)
            for node_type in data.node_types
            for idx in range(data[node_type].num_nodes)
        ]
        token_to_index = {token: idx for idx, token in enumerate(tokens)}
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        counts: Counter[tuple[int, int]] = Counter()

        for walk in walks:
            encoded = [token_to_index[token] for token in walk if token in token_to_index]
            for pos, center in enumerate(encoded):
                start = max(0, pos - self.context_size)
                end = min(len(encoded), pos + self.context_size + 1)
                for context_pos in range(start, end):
                    if context_pos == pos:
                        continue
                    counts[(center, encoded[context_pos])] += 1

        for (row, col), value in counts.items():
            rows.append(row)
            cols.append(col)
            values.append(float(value))

        if tokens and counts:
            matrix = sp.csr_matrix((values, (rows, cols)), shape=(len(tokens), len(tokens)))
            matrix = normalize(matrix, norm="l2", copy=False)
            n_components = min(self.embedding_dim, max(1, min(matrix.shape) - 1))
            if n_components >= 1 and matrix.shape[1] > 1:
                from sklearn.decomposition import TruncatedSVD

                projector = TruncatedSVD(n_components=n_components, random_state=self.random_state)
                embedding = projector.fit_transform(matrix)
            else:
                embedding = matrix.toarray()
        else:
            rng = np.random.default_rng(self.random_state)
            embedding = rng.normal(0.0, 0.01, size=(len(tokens), self.embedding_dim))

        if embedding.shape[1] < self.embedding_dim:
            padding = np.zeros((embedding.shape[0], self.embedding_dim - embedding.shape[1]))
            embedding = np.hstack([embedding, padding])
        elif embedding.shape[1] > self.embedding_dim:
            embedding = embedding[:, : self.embedding_dim]

        self.embeddings_: dict[str, np.ndarray] = {}
        self.original_ids_: dict[str, list[Any]] = {}
        offset = 0
        for node_type in data.node_types:
            count = data[node_type].num_nodes
            self.embeddings_[node_type] = np.asarray(
                embedding[offset : offset + count],
                dtype=np.float32,
            )
            self.original_ids_[node_type] = list(
                getattr(data[node_type], "original_id", range(count))
            )
            offset += count
        self.is_fitted_ = True
        return self

    def transform(self, X: HeteroData | nx.Graph | None = None) -> dict[str, np.ndarray]:
        if not hasattr(self, "is_fitted_"):
            raise NotFittedError("MetaPath2VecNodeEmbedder is not fitted")
        if X is not None:
            return self.fit(X).embeddings_
        return self.embeddings_

    def fit_transform(
        self,
        X: HeteroData | nx.Graph,
        y: Any = None,
        **fit_params: Any,
    ) -> dict[str, np.ndarray]:
        return self.fit(X, y).embeddings_
