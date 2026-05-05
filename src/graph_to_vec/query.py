"""Lightweight nearest-neighbor querying over graph or node embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted


class EmbeddingIndex(BaseEstimator):
    """In-memory vector index for small and medium embedding workflows.

    This intentionally uses sklearn instead of adding a vector database dependency. Larger
    deployments can export the same embeddings to FAISS, pgvector, LanceDB, or similar systems.
    """

    def __init__(
        self,
        metric: str = "cosine",
        algorithm: str = "auto",
        normalize_embeddings: bool = True,
    ) -> None:
        self.metric = metric
        self.algorithm = algorithm
        self.normalize_embeddings = normalize_embeddings

    def fit(
        self,
        embeddings: np.ndarray,
        ids: list[Any] | np.ndarray | None = None,
        metadata: pd.DataFrame | list[dict[str, Any]] | None = None,
    ) -> EmbeddingIndex:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a 2D matrix")
        if ids is None:
            ids = list(range(matrix.shape[0]))
        if len(ids) != matrix.shape[0]:
            raise ValueError("ids length must match embeddings rows")

        self.ids_ = list(ids)
        self.id_to_index_ = {item_id: idx for idx, item_id in enumerate(self.ids_)}
        self.embeddings_ = normalize(matrix, norm="l2") if self.normalize_embeddings else matrix
        self.neighbors_ = NearestNeighbors(metric=self.metric, algorithm=self.algorithm)
        self.neighbors_.fit(self.embeddings_)

        if metadata is None:
            self.metadata_ = pd.DataFrame(index=range(matrix.shape[0]))
        else:
            self.metadata_ = pd.DataFrame(metadata).reset_index(drop=True)
            if len(self.metadata_) != matrix.shape[0]:
                raise ValueError("metadata length must match embeddings rows")
        return self

    def query(
        self,
        vector: np.ndarray | None = None,
        item_id: Any | None = None,
        top_k: int = 10,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        """Return nearest neighbors for a vector or a fitted item id."""

        check_is_fitted(self, "neighbors_")
        if vector is None and item_id is None:
            raise ValueError("provide either vector or item_id")
        if vector is not None and item_id is not None:
            raise ValueError("provide vector or item_id, not both")
        if item_id is not None:
            if item_id not in self.id_to_index_:
                raise KeyError(f"unknown item_id {item_id!r}")
            query_index = self.id_to_index_[item_id]
            query_vector = self.embeddings_[query_index : query_index + 1]
        else:
            query_index = None
            query_vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
            if self.normalize_embeddings:
                query_vector = normalize(query_vector, norm="l2")

        requested = min(
            len(self.ids_),
            top_k + (1 if exclude_self and query_index is not None else 0),
        )
        distances, indices = self.neighbors_.kneighbors(query_vector, n_neighbors=requested)

        rows = []
        for distance, idx in zip(distances[0], indices[0], strict=False):
            if exclude_self and query_index is not None and int(idx) == query_index:
                continue
            score = (
                1.0 - float(distance)
                if self.metric == "cosine"
                else 1.0 / (1.0 + float(distance))
            )
            row = {
                "rank": len(rows) + 1,
                "id": self.ids_[int(idx)],
                "distance": float(distance),
                "score": score,
            }
            if not self.metadata_.empty:
                row.update(self.metadata_.iloc[int(idx)].to_dict())
            rows.append(row)
            if len(rows) >= top_k:
                break
        return pd.DataFrame(rows)
