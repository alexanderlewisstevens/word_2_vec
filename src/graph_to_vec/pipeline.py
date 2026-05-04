"""sklearn-compatible graph classification pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

from graph_to_vec.embeddings import Graph2VecTransformer


class GraphClassificationPipeline(BaseEstimator, ClassifierMixin):
    """Embed graphs, then train a conventional sklearn classifier."""

    def __init__(
        self,
        embedder: Any | None = None,
        classifier: Any | None = None,
    ) -> None:
        self.embedder = embedder
        self.classifier = classifier

    def fit(self, X: Any, y: Any) -> GraphClassificationPipeline:
        self.embedder_ = (
            clone(self.embedder) if self.embedder is not None else Graph2VecTransformer()
        )
        self.classifier_ = (
            clone(self.classifier)
            if self.classifier is not None
            else LogisticRegression(max_iter=1000)
        )
        embeddings = self.embedder_.fit_transform(X, y)
        self.classifier_.fit(embeddings, y)
        self.classes_ = getattr(self.classifier_, "classes_", np.unique(y))
        return self

    def transform(self, X: Any) -> Any:
        check_is_fitted(self, "classifier_")
        return self.embedder_.transform(X)

    def infer(self, X: Any) -> Any:
        return self.transform(X)

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict(self.embedder_.transform(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classifier_")
        if not hasattr(self.classifier_, "predict_proba"):
            raise AttributeError("wrapped classifier does not expose predict_proba")
        return self.classifier_.predict_proba(self.embedder_.transform(X))

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classifier_")
        if not hasattr(self.classifier_, "decision_function"):
            raise AttributeError("wrapped classifier does not expose decision_function")
        return self.classifier_.decision_function(self.embedder_.transform(X))

    def score(self, X: Any, y: Any) -> float:
        return float(accuracy_score(y, self.predict(X)))
