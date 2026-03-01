from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .data_loading import SupervisedDataset, SurfaceDatasetBuilder, WalkForwardSplit


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_t - y_p)))


# def qlike(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
#     y_t = np.clip(np.asarray(y_true, dtype=float), eps, None)
#     y_p = np.clip(np.asarray(y_pred, dtype=float), eps, None)
#     return float(np.mean(np.log(y_p) + (y_t / y_p)))


DEFAULT_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse": rmse,
    "mae": mae,
    # "qlike": qlike,
}


class ModelInterface(ABC):
    """Generic model interface for training and inference."""

    supports_batch_prediction: bool = True

    @abstractmethod
    def clone(self) -> "ModelInterface":
        """Return a fresh model instance with the same configuration."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Batch prediction. Override if supported."""
        raise NotImplementedError

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        """Single-sample prediction. Override if needed."""
        raise NotImplementedError


class FeatureExtractorInterface(ABC):
    """Generic feature extractor interface."""

    @abstractmethod
    def clone(self) -> "FeatureExtractorInterface":
        """Return a fresh extractor instance with the same configuration."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit extractor on train features."""

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features."""

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y=y)
        return self.transform(X)


class IdentityFeatureExtractor(FeatureExtractorInterface):
    """No-op feature extractor."""

    def clone(self) -> "IdentityFeatureExtractor":
        return IdentityFeatureExtractor()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        _ = y

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)


class PCAFeatureExtractor(FeatureExtractorInterface):
    """PCA feature extractor that can be reused with any model."""

    def __init__(
        self,
        n_components: Optional[int] = None,
        whiten: bool = False,
        svd_solver: str = "auto",
        random_state: Optional[int] = None,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.random_state = random_state
        self._pca: Optional[PCA] = None

    def clone(self) -> "PCAFeatureExtractor":
        return PCAFeatureExtractor(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        _ = y
        self._pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
        )
        self._pca.fit(np.asarray(X))

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise ValueError("PCAFeatureExtractor is not fitted. Call fit() first.")
        return np.asarray(self._pca.transform(np.asarray(X)))


class SklearnModelAdapter(ModelInterface):
    """
    Adapter for sklearn-like models.

    model_factory: callable returning a new estimator instance.
    """

    def __init__(self, model_factory: Callable[[], object]) -> None:
        self._model_factory = model_factory
        self._model = model_factory()

    def clone(self) -> "SklearnModelAdapter":
        return SklearnModelAdapter(model_factory=self._model_factory)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(X))

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(np.asarray(x).reshape(1, -1))).reshape(-1)


@dataclass(frozen=True)
class PipelineResult:
    dataset: SupervisedDataset
    cv_splits: List[WalkForwardSplit]
    cv_fold_metrics: pd.DataFrame
    cv_summary: pd.DataFrame
    holdout_metrics: Dict[str, float]
    holdout_true: np.ndarray
    holdout_direct_predictions: np.ndarray
    holdout_recursive_predictions: np.ndarray
    final_model: ModelInterface
    final_feature_extractor: FeatureExtractorInterface


class TrainingDriver:
    """Generic training driver over walk-forward CV + holdout evaluation."""

    def __init__(
        self,
        dataset_builder: SurfaceDatasetBuilder,
        lookback_window: int,
        holdout_size: int,
        num_splits: int,
        cross_val_frac: float,
        include_static_features: bool = True,
        dataset_factory: Optional[
            Callable[[SurfaceDatasetBuilder, int, bool], SupervisedDataset]
        ] = None,
        feature_extractor: Optional[FeatureExtractorInterface] = None,
        metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    ) -> None:
        self.dataset_builder = dataset_builder
        self.lookback_window = int(lookback_window)
        self.holdout_size = int(holdout_size)
        self.num_splits = int(num_splits)
        self.cross_val_frac = float(cross_val_frac)
        self.include_static_features = include_static_features
        self.dataset_factory = dataset_factory
        self.feature_extractor = (
            feature_extractor if feature_extractor is not None else IdentityFeatureExtractor()
        )
        self.metrics = metrics if metrics is not None else deepcopy(DEFAULT_METRICS)

    def _predict(self, model: ModelInterface, X: np.ndarray) -> np.ndarray:
        if model.supports_batch_prediction:
            try:
                return np.asarray(model.predict(X))
            except NotImplementedError:
                pass

        preds = [np.asarray(model.predict_one(x)) for x in X]
        return np.asarray(preds)

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for metric_name, metric_fn in self.metrics.items():
            out[f"{prefix}_{metric_name}"] = float(metric_fn(y_true, y_pred))
        return out

    def _train_on_indices(
        self,
        model: ModelInterface,
        dataset: SupervisedDataset,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        train_prefix: str,
        test_prefix: str,
    ) -> Dict[str, float]:
        X_train = dataset.X[train_indices]
        y_train = dataset.y[train_indices]
        X_test = dataset.X[test_indices]
        y_test = dataset.y[test_indices]

        model.fit(X_train, y_train)

        y_train_hat = self._predict(model, X_train)
        y_test_hat = self._predict(model, X_test)

        if y_train_hat.ndim > 1:
            y_train_hat = y_train_hat.reshape(-1)
        if y_test_hat.ndim > 1:
            y_test_hat = y_test_hat.reshape(-1)

        metrics = {}
        metrics.update(self._score(y_train, y_train_hat, train_prefix))
        metrics.update(self._score(y_test, y_test_hat, test_prefix))
        return metrics

    def _build_day_feature_matrix(
        self,
        values: np.ndarray,
        day_idx: int,
    ) -> np.ndarray:
        lookback = self.lookback_window
        n_points = self.dataset_builder.surface.num_points
        tenors = self.dataset_builder.surface.tenors
        maturities = self.dataset_builder.surface.maturities

        history = values[day_idx - lookback : day_idx, :]
        if not self.include_static_features:
            return history.T.astype(np.float32)

        X = np.empty((n_points, lookback + 4), dtype=np.float32)
        for col_idx in range(n_points):
            hist = history[:, col_idx].astype(np.float32)
            tenor = float(tenors[col_idx])
            maturity = float(maturities[col_idx])
            X[col_idx, :] = np.concatenate(
                (
                    hist,
                    np.array(
                        [float(day_idx), tenor, maturity, tenor * maturity],
                        dtype=np.float32,
                    ),
                )
            )
        return X

    def _recursive_holdout_predictions(
        self,
        model: ModelInterface,
        feature_extractor: FeatureExtractorInterface,
        holdout_day_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        surface_values_true = self.dataset_builder.surface.values
        rolling_values = np.array(surface_values_true, copy=True)

        y_true_days: List[np.ndarray] = []
        y_pred_days: List[np.ndarray] = []

        for day_idx in holdout_day_indices:
            X_day_raw = self._build_day_feature_matrix(rolling_values, int(day_idx))
            X_day = feature_extractor.transform(X_day_raw)
            y_day_pred = self._predict(model, X_day).reshape(-1)
            y_day_true = surface_values_true[int(day_idx), :].reshape(-1)

            if y_day_pred.shape[0] != y_day_true.shape[0]:
                raise ValueError(
                    f"Prediction size mismatch on day {day_idx}: "
                    f"pred={y_day_pred.shape[0]}, true={y_day_true.shape[0]}"
                )

            y_true_days.append(y_day_true)
            y_pred_days.append(y_day_pred)
            rolling_values[int(day_idx), :] = y_day_pred

        return np.concatenate(y_true_days), np.concatenate(y_pred_days)

    def run(self, model: ModelInterface) -> PipelineResult:
        if self.dataset_factory is not None:
            dataset = self.dataset_factory(
                self.dataset_builder,
                self.lookback_window,
                self.include_static_features,
            )
        else:
            dataset = self.dataset_builder.create_supervised_dataset(
                lookback_window=self.lookback_window,
                include_static_features=self.include_static_features,
            )

        cv_splits = list(
            self.dataset_builder.iter_walk_forward_splits(
                dataset=dataset,
                holdout_size=self.holdout_size,
                num_splits=self.num_splits,
                cross_val_frac=self.cross_val_frac,
            )
        )
        if not cv_splits:
            raise ValueError("No CV splits were generated.")

        fold_rows = []
        for split in cv_splits:
            fold_model = model.clone()
            fold_feature_extractor = self.feature_extractor.clone()
            X_train_raw = dataset.X[split.train_indices]
            y_train = dataset.y[split.train_indices]
            X_val_raw = dataset.X[split.val_indices]
            y_val = dataset.y[split.val_indices]

            X_train = fold_feature_extractor.fit_transform(X_train_raw, y=y_train)
            X_val = fold_feature_extractor.transform(X_val_raw)

            fold_model.fit(X_train, y_train)
            y_train_hat = self._predict(fold_model, X_train).reshape(-1)
            y_val_hat = self._predict(fold_model, X_val).reshape(-1)

            fold_metrics = {}
            fold_metrics.update(self._score(y_train, y_train_hat, "train"))
            fold_metrics.update(self._score(y_val, y_val_hat, "val"))
            fold_metrics["split_id"] = int(split.split_id)
            fold_metrics["n_train"] = int(split.train_indices.size)
            fold_metrics["n_val"] = int(split.val_indices.size)
            fold_rows.append(fold_metrics)

        cv_fold_metrics = pd.DataFrame(fold_rows).sort_values("split_id").reset_index(drop=True)

        summary_rows = []
        for col in cv_fold_metrics.columns:
            if col in ("split_id", "n_train", "n_val"):
                continue
            summary_rows.append(
                {
                    "metric": col,
                    "mean": float(cv_fold_metrics[col].mean()),
                    "std": float(cv_fold_metrics[col].std(ddof=0)),
                }
            )
        cv_summary = pd.DataFrame(summary_rows)

        trainval_idx, holdout_idx = self.dataset_builder.get_holdout_indices(
            dataset=dataset,
            holdout_size=self.holdout_size,
        )

        final_feature_extractor = self.feature_extractor.clone()
        final_model = model.clone()
        X_trainval_raw = dataset.X[trainval_idx]
        y_trainval = dataset.y[trainval_idx]
        X_holdout_raw = dataset.X[holdout_idx]
        y_holdout = dataset.y[holdout_idx]

        X_trainval = final_feature_extractor.fit_transform(X_trainval_raw, y=y_trainval)
        X_holdout = final_feature_extractor.transform(X_holdout_raw)
        final_model.fit(X_trainval, y_trainval)
        y_trainval_hat = self._predict(final_model, X_trainval).reshape(-1)
        y_holdout_hat = self._predict(final_model, X_holdout).reshape(-1)

        holdout_metrics = {}
        holdout_metrics.update(self._score(y_trainval, y_trainval_hat, "trainval"))
        holdout_metrics.update(self._score(y_holdout, y_holdout_hat, "holdout"))
        holdout_metrics["n_trainval"] = float(trainval_idx.size)
        holdout_metrics["n_holdout"] = float(holdout_idx.size)
        holdout_days = np.unique(dataset.sample_day_idx[holdout_idx])
        holdout_metrics["n_holdout_days"] = float(holdout_days.size)

        holdout_true = dataset.y[holdout_idx].reshape(-1)
        holdout_direct_predictions = y_holdout_hat.reshape(-1)

        y_true_recursive, y_pred_recursive = self._recursive_holdout_predictions(
            model=final_model,
            feature_extractor=final_feature_extractor,
            holdout_day_indices=holdout_days,
        )
        holdout_recursive_predictions = y_pred_recursive.reshape(-1)
        for metric_name, metric_fn in self.metrics.items():
            holdout_metrics[f"holdout_recursive_{metric_name}"] = float(
                metric_fn(y_true_recursive, y_pred_recursive)
            )

        return PipelineResult(
            dataset=dataset,
            cv_splits=cv_splits,
            cv_fold_metrics=cv_fold_metrics,
            cv_summary=cv_summary,
            holdout_metrics=holdout_metrics,
            holdout_true=holdout_true,
            holdout_direct_predictions=holdout_direct_predictions,
            holdout_recursive_predictions=holdout_recursive_predictions,
            final_model=final_model,
            final_feature_extractor=final_feature_extractor,
        )
