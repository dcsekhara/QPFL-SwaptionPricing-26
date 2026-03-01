from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List

from sklearn.linear_model import LinearRegression, Ridge

from .training_pipeline import (
    FeatureExtractorInterface,
    IdentityFeatureExtractor,
    ModelInterface,
    PCAFeatureExtractor,
    SklearnModelAdapter,
)
from .torch_models import TorchCNNRegressor, is_torch_available
from .quantum_reservoir import QuantumReservoirRegressor


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: ModelInterface


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    feature_extractor: FeatureExtractorInterface


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    model: ModelInterface
    feature_extractor: FeatureExtractorInterface


class LinearRegressionModel(SklearnModelAdapter):
    """No-regularization linear regression wrapper."""

    def __init__(self, **kwargs) -> None:
        self._kwargs = dict(kwargs)
        super().__init__(lambda: LinearRegression(**self._kwargs))

    def clone(self) -> "LinearRegressionModel":
        return LinearRegressionModel(**self._kwargs)


class RidgeRegressionModel(SklearnModelAdapter):
    """Ridge regression wrapper."""

    def __init__(self, alpha: float = 1.0, **kwargs) -> None:
        self._alpha = float(alpha)
        self._kwargs = dict(kwargs)
        super().__init__(lambda: Ridge(alpha=self._alpha, **self._kwargs))

    def clone(self) -> "RidgeRegressionModel":
        return RidgeRegressionModel(alpha=self._alpha, **self._kwargs)


class XGBoostModel(SklearnModelAdapter):
    """
    XGBoost regressor wrapper.
    Requires package: xgboost
    """

    def __init__(self, **kwargs) -> None:
        self._kwargs = dict(kwargs)
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBoostModel. Install with: pip install xgboost"
            ) from exc
        super().__init__(lambda: XGBRegressor(**self._kwargs))

    def clone(self) -> "XGBoostModel":
        return XGBoostModel(**self._kwargs)


def _build_xgboost_model_specs_from_grid(
    grid: Dict[str, List[Any]],
    random_state: int,
    n_jobs: int,
) -> List[ModelSpec]:
    shared = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
    }

    keys = list(grid.keys())
    specs: List[ModelSpec] = []
    for values in product(*(grid[k] for k in keys)):
        params = {k: v for k, v in zip(keys, values)}
        params.update(shared)
        name = (
            "xgb_"
            f"ne{params['n_estimators']}_"
            f"lr{params['learning_rate']}_"
            f"md{params['max_depth']}_"
            f"mcw{params['min_child_weight']}_"
            f"ss{params['subsample']}_"
            f"cs{params['colsample_bytree']}_"
            f"ra{params['reg_alpha']}_"
            f"rl{params['reg_lambda']}"
        )
        specs.append(ModelSpec(name=name, model=XGBoostModel(**params)))
    return specs


def baseline_model_specs(ridge_alphas: List[float]) -> List[ModelSpec]:
    specs: List[ModelSpec] = [
        # ModelSpec(name="linear", model=LinearRegressionModel())
    ]
    for alpha in ridge_alphas:
        specs.append(
            ModelSpec(
                name=f"ridge_alpha={alpha}",
                model=RidgeRegressionModel(alpha=float(alpha)),
            )
        )
    return specs


def xgboost_exhaustive_model_specs(
    random_state: int = 42,
    n_jobs: int = -1,
) -> List[ModelSpec]:
    """
    Exhaustive Cartesian product over a strong XGBoost search space.
    This is intentionally large; use with care for runtime.
    """
    grid: Dict[str, List[Any]] = {
        "n_estimators": [300, 700],
        "learning_rate": [0.03, 0.1],
        "max_depth": [4, 8],
        "min_child_weight": [1, 6],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "reg_alpha": [0.0, 0.01],
        "reg_lambda": [1.0, 5.0],
    }

    return _build_xgboost_model_specs_from_grid(
        grid=grid,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def xgboost_model_specs(
    search_mode: str = "standard",
    random_state: int = 42,
    n_jobs: int = -1,
) -> List[ModelSpec]:
    """
    Build XGBoost specs by budget:
    - quick: small sanity-check grid
    - standard: moderate grid
    - exhaustive: large Cartesian grid
    """
    mode = str(search_mode).strip().lower()
    if mode == "quick":
        grid: Dict[str, List[Any]] = {
            "n_estimators": [300],
            "learning_rate": [0.1],
            "max_depth": [4, 8],
            "min_child_weight": [1],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "reg_alpha": [0.0],
            "reg_lambda": [1.0],
        }
        return _build_xgboost_model_specs_from_grid(grid, random_state, n_jobs)

    if mode == "standard":
        grid = {
            "n_estimators": [300, 700],
            "learning_rate": [0.03, 0.1],
            "max_depth": [4, 8],
            "min_child_weight": [1, 6],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "reg_alpha": [0.0, 0.01],
            "reg_lambda": [1.0, 5.0],
        }
        return _build_xgboost_model_specs_from_grid(grid, random_state, n_jobs)

    if mode == "exhaustive":
        return xgboost_exhaustive_model_specs(
            random_state=random_state,
            n_jobs=n_jobs,
        )

    raise ValueError("search_mode must be one of: quick, standard, exhaustive")


def cnn_model_specs(
    epochs: int = 120,
    batch_size: int = 64,
    lr: float = 1e-3,
    conv_channels_grid: List[int] | None = None,
    kernel_size_grid: List[int] | None = None,
    hidden_dim_grid: List[int] | None = None,
    weight_decay_grid: List[float] | None = None,
    random_state: int = 42,
) -> List[ModelSpec]:
    if conv_channels_grid is None:
        conv_channels_grid = [64]
    if kernel_size_grid is None:
        kernel_size_grid = [3]
    if hidden_dim_grid is None:
        hidden_dim_grid = [256]
    if weight_decay_grid is None:
        weight_decay_grid = [0.0]

    specs: List[ModelSpec] = []
    for conv_channels in conv_channels_grid:
        for kernel_size in kernel_size_grid:
            for hidden_dim in hidden_dim_grid:
                for weight_decay in weight_decay_grid:
                    name = (
                        "cnn_"
                        f"cc{conv_channels}_"
                        f"ks{kernel_size}_"
                        f"hd{hidden_dim}_"
                        f"ep{epochs}_"
                        f"wd{weight_decay}"
                    )
                    specs.append(
                        ModelSpec(
                            name=name,
                            model=TorchCNNRegressor(
                                conv_channels=conv_channels,
                                kernel_size=kernel_size,
                                hidden_dim=hidden_dim,
                                epochs=epochs,
                                batch_size=batch_size,
                                lr=lr,
                                weight_decay=weight_decay,
                                random_state=random_state,
                                verbose=False,
                            ),
                        )
                    )
    return specs


def quantum_reservoir_model_specs(
    lookback_window: int,
    n_values: int | None = None,
    ridge_alpha_grid: List[float] | None = None,
    hidden_modes_grid: List[int] | None = None,
    depth_grid: List[int] | None = None,
    alpha_grid: List[float] | None = None,
    state_nonlinearity_grid: List[float] | None = None,
    strict_backend: bool = False,
    seed: int = 7,
    include_tail_features: bool = True,
) -> List[ModelSpec]:
    if ridge_alpha_grid is None:
        ridge_alpha_grid = [1e-3, 1e-2]
    if hidden_modes_grid is None:
        hidden_modes_grid = [6, 10]
    if depth_grid is None:
        depth_grid = [8, 12]
    if alpha_grid is None:
        alpha_grid = [0.8, 1.2]
    if state_nonlinearity_grid is None:
        state_nonlinearity_grid = [1.0]

    specs: List[ModelSpec] = []
    for ridge_alpha in ridge_alpha_grid:
        for hidden_modes in hidden_modes_grid:
            for depth in depth_grid:
                for alpha in alpha_grid:
                    for state_nonlinearity in state_nonlinearity_grid:
                        name = (
                            "qres_"
                            f"nv{n_values if n_values is not None else 'auto'}_"
                            f"hm{hidden_modes}_"
                            f"d{depth}_"
                            f"a{alpha}_"
                            f"sn{state_nonlinearity}_"
                            f"ra{ridge_alpha}"
                        )
                        specs.append(
                            ModelSpec(
                                name=name,
                                model=QuantumReservoirRegressor(
                                    lookback_window=lookback_window,
                                    n_values=n_values,
                                    hidden_modes=hidden_modes,
                                    depth=depth,
                                    alpha=alpha,
                                    state_nonlinearity=state_nonlinearity,
                                    seed=seed,
                                    strict_backend=strict_backend,
                                    ridge_alpha=ridge_alpha,
                                    include_tail_features=include_tail_features,
                                ),
                            )
                        )
    return specs


def baseline_feature_specs(pca_components: List[int]) -> List[FeatureSpec]:
    specs: List[FeatureSpec] = [
        FeatureSpec(name="raw", feature_extractor=IdentityFeatureExtractor()),
    ]
    for n_components in pca_components:
        specs.append(
            FeatureSpec(
                name=f"pca_{n_components}",
                feature_extractor=PCAFeatureExtractor(n_components=int(n_components), random_state=42),
            )
        )
    return specs


def build_experiment_specs(
    model_specs: List[ModelSpec],
    feature_specs: List[FeatureSpec],
) -> List[ExperimentSpec]:
    experiments: List[ExperimentSpec] = []
    for model_spec in model_specs:
        for feature_spec in feature_specs:
            experiments.append(
                ExperimentSpec(
                    name=f"{model_spec.name}__{feature_spec.name}",
                    model=model_spec.model.clone(),
                    feature_extractor=feature_spec.feature_extractor.clone(),
                )
            )
    return experiments
