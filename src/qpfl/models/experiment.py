from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

from .baselines import (
    build_lookback_dataset,
    generate_time_series_splits,
    make_surface_wide,
)
from .factory import build_model_from_config, default_model_configs


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def config_to_name(config):
    name = config.get("name", "model")
    items = [f"{k}={v}" for k, v in sorted(config.items()) if k != "name"]
    return f"{name}({', '.join(items)})" if items else str(name)


def cross_validate_model_configs(
    X,
    y,
    splits,
    model_configs=None,
):
    if model_configs is None:
        model_configs = default_model_configs()

    rows = []
    for i, cfg in enumerate(model_configs):
        cfg = deepcopy(cfg)
        model_label = config_to_name(cfg)
        tr_rmse_all, va_rmse_all = [], []
        tr_mae_all, va_mae_all = [], []

        for tr_idx, va_idx in splits:
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

            model = build_model_from_config(cfg)
            model.fit(X_tr, y_tr)
            y_tr_hat = model.predict(X_tr)
            y_va_hat = model.predict(X_va)

            tr_rmse_all.append(rmse(y_tr, y_tr_hat))
            va_rmse_all.append(rmse(y_va, y_va_hat))
            tr_mae_all.append(mae(y_tr, y_tr_hat))
            va_mae_all.append(mae(y_va, y_va_hat))

        rows.append(
            {
                "config_id": i,
                "model_name": model_label,
                "model_type": cfg["name"],
                "model_config": cfg,
                "train_mae_mean": float(np.mean(tr_mae_all)),
                "val_mae_mean": float(np.mean(va_mae_all)),
                "train_rmse_mean": float(np.mean(tr_rmse_all)),
                "val_rmse_mean": float(np.mean(va_rmse_all)),
                "n_folds": len(splits),
            }
        )

    return pd.DataFrame(rows)


def final_holdout_eval_from_configs(
    X,
    y,
    cv_results,
    holdout_size=20,
):
    if holdout_size <= 0 or holdout_size >= len(X):
        raise ValueError("holdout_size must be between 1 and len(X)-1.")

    n_train = len(X) - holdout_size
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]

    best = cv_results.sort_values("val_rmse_mean", ascending=True).iloc[0]
    model = build_model_from_config(best["model_config"])
    model.fit(X_tr, y_tr)
    y_tr_hat = model.predict(X_tr)
    y_te_hat = model.predict(X_te)

    summary = {
        "best_model": best["model_name"],
        "best_model_type": best["model_type"],
        "best_config_id": int(best["config_id"]),
        "train_rmse": float(rmse(y_tr, y_tr_hat)),
        "test_rmse": float(rmse(y_te, y_te_hat)),
    }
    return summary, model


def plot_train_val_by_config(cv_results, metric="rmse"):
    if metric == "rmse":
        tr_col, va_col, y_title = "train_rmse_mean", "val_rmse_mean", "RMSE"
    elif metric == "mae":
        tr_col, va_col, y_title = "train_mae_mean", "val_mae_mean", "MAE"
    else:
        raise ValueError("metric must be 'rmse' or 'mae'")

    sub = cv_results.sort_values("val_rmse_mean").reset_index(drop=True)
    x = [f"{row.config_id}:{row.model_type}" for row in sub.itertuples(index=False)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=sub[tr_col], name=f"train {metric}"))
    fig.add_trace(go.Bar(x=x, y=sub[va_col], name=f"val {metric}"))
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        title=f"Train vs Validation {y_title} by model config",
        xaxis_title="config_id:model_type (sorted by validation RMSE)",
        yaxis_title=y_title,
        width=1200,
        height=550,
    )
    return fig


def run_surface_experiment(
    long_df,
    lookback_k=5,
    cv_method="walk_forward",
    holdout_size=20,
    n_splits=5,
    train_frac=0.7,
    model_configs=None,
):
    surface_wide = make_surface_wide(long_df)
    X, y, y_dates = build_lookback_dataset(surface_wide, lookback_k=lookback_k)

    X_trainval, y_trainval = X[:-holdout_size], y[:-holdout_size]
    splits = generate_time_series_splits(
        n_samples=len(X_trainval),
        method=cv_method,
        k=n_splits,
        train_frac=train_frac,
    )

    cv_results = cross_validate_model_configs(
        X_trainval,
        y_trainval,
        splits=splits,
        model_configs=model_configs,
    )

    best_summary, best_model = final_holdout_eval_from_configs(
        X,
        y,
        cv_results,
        holdout_size=holdout_size,
    )

    fig_rmse = plot_train_val_by_config(cv_results, metric="rmse")
    fig_mae = plot_train_val_by_config(cv_results, metric="mae")

    return {
        "surface_wide": surface_wide,
        "X": X,
        "y": y,
        "y_dates": y_dates,
        "cv_results": cv_results,
        "best_summary": best_summary,
        "best_model": best_model,
        "fig_rmse": fig_rmse,
        "fig_mae": fig_mae,
    }

