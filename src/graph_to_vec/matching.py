"""Candidate matching and ranking utilities built on graph classification."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

from graph_to_vec.embeddings import Graph2VecTransformer
from graph_to_vec.pipeline import GraphClassificationPipeline


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value)


def _normalize_value(value: Any) -> str:
    if _is_missing(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _stable_id(value: Any) -> str:
    return hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).hexdigest()


def _token_set(value: Any) -> set[str]:
    normalized = _normalize_value(value)
    return {token for token in normalized.replace("@", " ").replace(".", " ").split() if token}


def _value_similarity(left: Any, right: Any) -> float:
    if _is_missing(left) or _is_missing(right):
        return 0.0
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
        right,
        (int, float, np.integer, np.floating),
    ):
        left_value = float(left)
        right_value = float(right)
        scale = max(abs(left_value), abs(right_value), 1.0)
        return max(0.0, 1.0 - abs(left_value - right_value) / scale)

    left_norm = _normalize_value(left)
    right_norm = _normalize_value(right)
    if left_norm and left_norm == right_norm:
        return 1.0
    left_tokens = _token_set(left_norm)
    right_tokens = _token_set(right_norm)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _block_key(row: pd.Series, columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(_normalize_value(row[column]) for column in columns)


def _safe_column_name(prefix: str, column: str) -> str:
    safe = "".join(char if char.isalnum() or char == "_" else "_" for char in column)
    return f"{prefix}_{safe}"


class CandidatePairBuilder(BaseEstimator):
    """Generate candidate record pairs with simple blocking and similarity features.

    The builder supports two common workflows:

    - dedupe: pass one record table and get unique within-table pairs.
    - linkage: pass left and right record tables and get cross-table pairs.
    """

    def __init__(
        self,
        id_col: str = "id",
        block_on: Sequence[str] | None = None,
        compare_on: Sequence[str] | None = None,
        ground_truth_col: str | None = None,
        label_col: str = "label",
        score_col: str = "candidate_score",
        include_self: bool = False,
        top_k_per_record: int | None = None,
        max_pairs: int | None = None,
    ) -> None:
        self.id_col = id_col
        self.block_on = block_on
        self.compare_on = compare_on
        self.ground_truth_col = ground_truth_col
        self.label_col = label_col
        self.score_col = score_col
        self.include_self = include_self
        self.top_k_per_record = top_k_per_record
        self.max_pairs = max_pairs

    def fit(
        self,
        left_records: pd.DataFrame,
        y: Any = None,
        right_records: pd.DataFrame | None = None,
    ) -> CandidatePairBuilder:
        left = pd.DataFrame(left_records)
        right = left if right_records is None else pd.DataFrame(right_records)
        common = [column for column in left.columns if column in right.columns]
        excluded = {self.id_col}
        if self.ground_truth_col:
            excluded.add(self.ground_truth_col)
        self.compare_columns_ = list(
            self.compare_on or [column for column in common if column not in excluded]
        )
        self.block_columns_ = list(self.block_on or [])
        missing = [
            column
            for column in [self.id_col, *self.compare_columns_, *self.block_columns_]
            if column not in left.columns or column not in right.columns
        ]
        if missing:
            raise ValueError(
                f"missing required columns in both record tables: {sorted(set(missing))}"
            )
        return self

    def transform(
        self,
        left_records: pd.DataFrame,
        right_records: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not hasattr(self, "compare_columns_"):
            self.fit(left_records, right_records=right_records)

        left = pd.DataFrame(left_records).reset_index(drop=True)
        right = (
            left
            if right_records is None
            else pd.DataFrame(right_records).reset_index(drop=True)
        )
        within_table = right_records is None
        right_groups: dict[tuple[str, ...], list[int]] = defaultdict(list)

        if self.block_columns_:
            for right_idx, row in right.iterrows():
                right_groups[_block_key(row, self.block_columns_)].append(int(right_idx))
        else:
            right_groups[()] = list(range(len(right)))

        rows = []
        for left_idx, left_row in left.iterrows():
            key = _block_key(left_row, self.block_columns_) if self.block_columns_ else ()
            candidates = right_groups.get(key, [])
            scored_rows = []

            for right_idx in candidates:
                if within_table:
                    if not self.include_self and right_idx <= left_idx:
                        continue
                    if self.include_self is False and left_idx == right_idx:
                        continue
                right_row = right.iloc[right_idx]

                pair: dict[str, Any] = {
                    "left_id": left_row[self.id_col],
                    "right_id": right_row[self.id_col],
                }
                similarities = []
                for column in self.compare_columns_:
                    similarity = _value_similarity(left_row[column], right_row[column])
                    similarities.append(similarity)
                    pair[_safe_column_name("left", column)] = left_row[column]
                    pair[_safe_column_name("right", column)] = right_row[column]
                    pair[_safe_column_name("sim", column)] = similarity
                    left_norm = _normalize_value(left_row[column])
                    right_norm = _normalize_value(right_row[column])
                    pair[_safe_column_name("same", column)] = bool(
                        left_norm and left_norm == right_norm
                    )
                pair[self.score_col] = float(np.mean(similarities)) if similarities else 0.0

                if self.ground_truth_col:
                    left_truth = _normalize_value(left_row[self.ground_truth_col])
                    right_truth = _normalize_value(right_row[self.ground_truth_col])
                    pair[self.label_col] = int(bool(left_truth and left_truth == right_truth))
                scored_rows.append(pair)

            scored_rows.sort(key=lambda item: item[self.score_col], reverse=True)
            if self.top_k_per_record is not None:
                scored_rows = scored_rows[: self.top_k_per_record]
            rows.extend(scored_rows)
            if self.max_pairs is not None and len(rows) >= self.max_pairs:
                rows = rows[: self.max_pairs]
                break

        return pd.DataFrame(rows)

    def fit_transform(
        self,
        left_records: pd.DataFrame,
        y: Any = None,
        right_records: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return self.fit(left_records, y=y, right_records=right_records).transform(
            left_records,
            right_records=right_records,
        )


class MatchGraphBuilder(BaseEstimator):
    """Convert candidate pairs into evidence graphs for graph classification."""

    def __init__(
        self,
        id_col: str = "id",
        left_id_col: str = "left_id",
        right_id_col: str = "right_id",
        label_col: str = "label",
        score_col: str = "candidate_score",
        compare_on: Sequence[str] | None = None,
        candidate_node_type: str = "candidate_match",
        left_node_type: str = "left_record",
        right_node_type: str = "right_record",
        evidence_node_type: str = "evidence",
    ) -> None:
        self.id_col = id_col
        self.left_id_col = left_id_col
        self.right_id_col = right_id_col
        self.label_col = label_col
        self.score_col = score_col
        self.compare_on = compare_on
        self.candidate_node_type = candidate_node_type
        self.left_node_type = left_node_type
        self.right_node_type = right_node_type
        self.evidence_node_type = evidence_node_type

    def fit(
        self,
        pairs: pd.DataFrame,
        y: Any = None,
        left_records: pd.DataFrame | None = None,
        right_records: pd.DataFrame | None = None,
    ) -> MatchGraphBuilder:
        pair_frame = pd.DataFrame(pairs)
        inferred = []
        for column in pair_frame.columns:
            if column.startswith("sim_"):
                inferred.append(column.removeprefix("sim_"))
        self.compare_columns_ = list(self.compare_on or inferred)
        return self

    def transform(
        self,
        pairs: pd.DataFrame,
        left_records: pd.DataFrame,
        right_records: pd.DataFrame | None = None,
    ) -> list[nx.MultiDiGraph]:
        if not hasattr(self, "compare_columns_"):
            self.fit(pairs, left_records=left_records, right_records=right_records)

        pair_frame = pd.DataFrame(pairs).reset_index(drop=True)
        left = pd.DataFrame(left_records)
        right = left if right_records is None else pd.DataFrame(right_records)
        left_lookup = left.set_index(self.id_col, drop=False).to_dict("index")
        right_lookup = right.set_index(self.id_col, drop=False).to_dict("index")

        graphs = []
        for _, pair in pair_frame.iterrows():
            left_id = pair[self.left_id_col]
            right_id = pair[self.right_id_col]
            left_row = left_lookup[left_id]
            right_row = right_lookup[right_id]
            graph_id = f"match:{left_id}:{right_id}"
            graph = nx.MultiDiGraph(id=graph_id)
            if self.label_col in pair:
                graph.graph["label"] = pair[self.label_col]
                graph.graph["y"] = int(pair[self.label_col])
            graph.graph["left_id"] = left_id
            graph.graph["right_id"] = right_id

            candidate_id = f"candidate:{left_id}|{right_id}"
            graph.add_node(
                candidate_id,
                type=self.candidate_node_type,
                candidate_score=float(pair.get(self.score_col, 0.0)),
            )
            graph.add_node(
                f"left:{left_id}",
                type=self.left_node_type,
                **self._numeric_features(left_row),
            )
            graph.add_node(
                f"right:{right_id}",
                type=self.right_node_type,
                **self._numeric_features(right_row),
            )
            graph.add_edge(candidate_id, f"left:{left_id}", relation="has_left")
            graph.add_edge(candidate_id, f"right:{right_id}", relation="has_right")
            graph.add_edge(f"left:{left_id}", candidate_id, relation="left_of")
            graph.add_edge(f"right:{right_id}", candidate_id, relation="right_of")

            for column in self.compare_columns_:
                sim_col = _safe_column_name("sim", column)
                default_similarity = _value_similarity(left_row.get(column), right_row.get(column))
                similarity = float(pair.get(sim_col, default_similarity))
                left_norm = _normalize_value(left_row.get(column))
                right_norm = _normalize_value(right_row.get(column))
                same = bool(
                    left_norm
                    and left_norm == right_norm
                )
                value_digest = _stable_id((left_row.get(column), right_row.get(column)))
                evidence_id = f"evidence:{column}:{value_digest}"
                relation = (
                    "exact_match"
                    if same
                    else "strong_similarity"
                    if similarity >= 0.5
                    else "weak_similarity"
                )
                graph.add_node(
                    evidence_id,
                    type=self.evidence_node_type,
                    similarity=similarity,
                    exact_match=float(same),
                    value_present=float(
                        bool(left_norm)
                        and bool(right_norm)
                    ),
                )
                graph.add_edge(candidate_id, evidence_id, relation=relation)
                graph.add_edge(evidence_id, candidate_id, relation="supports_candidate")
                graph.add_edge(f"left:{left_id}", evidence_id, relation=f"left_{column}")
                graph.add_edge(f"right:{right_id}", evidence_id, relation=f"right_{column}")

            graphs.append(graph)
        return graphs

    def fit_transform(
        self,
        pairs: pd.DataFrame,
        y: Any = None,
        left_records: pd.DataFrame | None = None,
        right_records: pd.DataFrame | None = None,
    ) -> list[nx.MultiDiGraph]:
        if left_records is None:
            raise ValueError("left_records is required")
        return self.fit(
            pairs,
            y=y,
            left_records=left_records,
            right_records=right_records,
        ).transform(pairs, left_records=left_records, right_records=right_records)

    def _numeric_features(self, row: dict[str, Any]) -> dict[str, float]:
        features = {}
        for key, value in row.items():
            if key == self.id_col:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)) and not _is_missing(value):
                features[str(key)] = float(value)
        return features


class MatchRanker(BaseEstimator, ClassifierMixin):
    """Train, score, and query candidate matches as graph classification samples."""

    def __init__(
        self,
        candidate_builder: CandidatePairBuilder | None = None,
        graph_builder: MatchGraphBuilder | None = None,
        pipeline: GraphClassificationPipeline | None = None,
        positive_label: Any = 1,
    ) -> None:
        self.candidate_builder = candidate_builder
        self.graph_builder = graph_builder
        self.pipeline = pipeline
        self.positive_label = positive_label

    def fit(
        self,
        records: pd.DataFrame,
        y: Sequence[Any] | None = None,
        pairs: pd.DataFrame | None = None,
        right_records: pd.DataFrame | None = None,
    ) -> MatchRanker:
        self.candidate_builder_ = (
            clone(self.candidate_builder) if self.candidate_builder else CandidatePairBuilder()
        )
        self.graph_builder_ = (
            clone(self.graph_builder) if self.graph_builder else MatchGraphBuilder()
        )
        self.pipeline_ = (
            clone(self.pipeline)
            if self.pipeline
            else GraphClassificationPipeline(
                embedder=Graph2VecTransformer(
                    iterations=2,
                    embedding_dim=128,
                    include_features=True,
                    random_state=13,
                ),
                classifier=LogisticRegression(max_iter=1000, class_weight="balanced"),
            )
        )

        pair_frame = (
            pd.DataFrame(pairs).copy()
            if pairs is not None
            else self.candidate_builder_.fit_transform(records, right_records=right_records)
        )
        if y is not None:
            pair_frame[self.graph_builder_.label_col] = list(y)
        if self.graph_builder_.label_col not in pair_frame.columns:
            raise ValueError(
                f"training pairs must include {self.graph_builder_.label_col!r} "
                "or y must be provided"
            )

        graphs = self.graph_builder_.fit_transform(
            pair_frame,
            left_records=records,
            right_records=right_records,
        )
        labels = pair_frame[self.graph_builder_.label_col].tolist()
        self.pipeline_.fit(graphs, labels)
        self.classes_ = self.pipeline_.classes_
        self.training_pairs_ = pair_frame
        self.records_ = pd.DataFrame(records).copy()
        self.right_records_ = None if right_records is None else pd.DataFrame(right_records).copy()
        return self

    def _build_graphs(
        self,
        records: pd.DataFrame | None = None,
        pairs: pd.DataFrame | None = None,
        right_records: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, list[nx.MultiDiGraph]]:
        if not hasattr(self, "pipeline_"):
            raise ValueError("MatchRanker is not fitted")
        records = self.records_ if records is None else pd.DataFrame(records)
        right_records = (
            self.right_records_ if right_records is None else pd.DataFrame(right_records)
        )
        pair_frame = (
            pd.DataFrame(pairs).copy()
            if pairs is not None
            else self.candidate_builder_.transform(records, right_records=right_records)
        )
        graphs = self.graph_builder_.transform(
            pair_frame,
            left_records=records,
            right_records=right_records,
        )
        return pair_frame, graphs

    def predict(
        self,
        records: pd.DataFrame | None = None,
        pairs: pd.DataFrame | None = None,
        right_records: pd.DataFrame | None = None,
    ) -> np.ndarray:
        _, graphs = self._build_graphs(records=records, pairs=pairs, right_records=right_records)
        return self.pipeline_.predict(graphs)

    def predict_proba(
        self,
        records: pd.DataFrame | None = None,
        pairs: pd.DataFrame | None = None,
        right_records: pd.DataFrame | None = None,
    ) -> np.ndarray:
        _, graphs = self._build_graphs(records=records, pairs=pairs, right_records=right_records)
        return self.pipeline_.predict_proba(graphs)

    def rank(
        self,
        records: pd.DataFrame | None = None,
        pairs: pd.DataFrame | None = None,
        right_records: pd.DataFrame | None = None,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        pair_frame, graphs = self._build_graphs(
            records=records,
            pairs=pairs,
            right_records=right_records,
        )
        probabilities = self.pipeline_.predict_proba(graphs)
        predictions = self.pipeline_.predict(graphs)
        classes = list(self.pipeline_.classes_)
        class_index = classes.index(self.positive_label) if self.positive_label in classes else -1

        output = pair_frame.copy()
        output["prediction"] = predictions
        output["match_score"] = probabilities[:, class_index]
        output = output.sort_values("match_score", ascending=False).reset_index(drop=True)
        output.insert(0, "rank", np.arange(1, len(output) + 1))
        if top_k is not None:
            output = output.head(top_k)
        return output

    def query(
        self,
        record_id: Any,
        records: pd.DataFrame | None = None,
        pairs: pd.DataFrame | None = None,
        right_records: pd.DataFrame | None = None,
        side: str = "either",
        top_k: int = 10,
    ) -> pd.DataFrame:
        ranked = self.rank(
            records=records,
            pairs=pairs,
            right_records=right_records,
            top_k=None,
        )
        if side == "left":
            ranked = ranked[ranked[self.graph_builder_.left_id_col] == record_id]
        elif side == "right":
            ranked = ranked[ranked[self.graph_builder_.right_id_col] == record_id]
        elif side == "either":
            ranked = ranked[
                (ranked[self.graph_builder_.left_id_col] == record_id)
                | (ranked[self.graph_builder_.right_id_col] == record_id)
            ]
        else:
            raise ValueError("side must be 'left', 'right', or 'either'")
        return ranked.head(top_k).reset_index(drop=True)
