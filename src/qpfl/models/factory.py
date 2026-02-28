from copy import deepcopy

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor

from .deep_learning import TorchCNNRegressor, TorchLSTMRegressor, TorchMLPRegressor


def build_model_from_config(config):
    cfg = deepcopy(config)
    name = str(cfg.pop("name")).lower()

    if name == "linear":
        return LinearRegression(**cfg)
    if name == "ridge":
        return Ridge(**cfg)
    if name == "mlp":
        return MLPRegressor(**cfg)
    if name == "xgboost":
        try:
            from sklearn.multioutput import MultiOutputRegressor
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for model 'xgboost'. Install xgboost in your environment."
            ) from exc
        return MultiOutputRegressor(XGBRegressor(**cfg))
    if name == "cnn":
        return TorchCNNRegressor(**cfg)
    if name == "lstm":
        return TorchLSTMRegressor(**cfg)
    if name == "mlp_torch":
        return TorchMLPRegressor(**cfg)

    raise ValueError(
        "Unknown model name: "
        f"{name}. Supported: linear, ridge, mlp, xgboost, cnn, lstm, mlp_torch"
    )


def default_model_configs():
    return [
        {"name": "linear"},
        {"name": "ridge", "alpha": 1e-3},
        {"name": "ridge", "alpha": 1e-2},
        {"name": "ridge", "alpha": 1e-1},
        {"name": "ridge", "alpha": 1.0},
        {"name": "ridge", "alpha": 10.0},
        {"name": "ridge", "alpha": 100.0},
    ]

