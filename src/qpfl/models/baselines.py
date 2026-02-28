import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


def make_surface_wide(
    long_df,
    date_col="Date",
    tenor_col="Tenor",
    maturity_col="Maturity",
    price_col="price",
):
    df = long_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, tenor_col, maturity_col, price_col])

    wide = df.pivot_table(
        index=date_col,
        columns=[tenor_col, maturity_col],
        values=price_col,
        aggfunc="mean",
    )
    wide = wide.sort_index().sort_index(axis=1)
    wide.columns = [f"T{int(t)}_M{int(m)}m" for t, m in wide.columns]
    return wide


def build_lookback_dataset(surface_wide, lookback_k=5):
    # shape: [n_dates, n_points]
    V = surface_wide.to_numpy(dtype=float)
    D = surface_wide.index.to_numpy()

    if len(V) <= lookback_k:
        raise ValueError("Not enough dates for selected lookback_k.")

    X, y, y_dates = [], [], []
    for t in range(lookback_k - 1, len(V) - 1):
        X.append(V[t - lookback_k + 1 : t + 1].reshape(-1))
        y.append(V[t + 1])
        y_dates.append(D[t + 1])

    return np.asarray(X), np.asarray(y), np.asarray(y_dates)


def generate_time_series_splits(
    n_samples: int,
    method: str = "walk_forward",  # "walk_forward"|"rolling"|"expanding"
    k: int = 5,
    train_frac: float = 0.7,
):
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    if k < 1:
        raise ValueError("k must be >= 1")
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be in (0, 1)")
    if method not in ("walk_forward", "rolling", "expanding"):
        raise ValueError("method must be one of: 'walk_forward', 'rolling', 'expanding'")

    edges = np.linspace(0, n_samples, k + 1, dtype=int)
    splits = []
    fixed_train_len = None

    for i in range(k):
        b_start, b_end = edges[i], edges[i + 1]
        block_len = b_end - b_start
        if block_len < 2:
            assert (
                False
            ), f"Block {i} is too small for splitting (length={block_len}). Adjust k or ensure n_samples is large enough."

        tr_len_block = int(np.floor(block_len * train_frac))
        tr_len_block = max(1, min(tr_len_block, block_len - 1))
        cut = b_start + tr_len_block

        if method == "expanding":
            tr_idx = np.arange(0, cut)
        else:
            if fixed_train_len is None:
                fixed_train_len = tr_len_block
            tr_start = max(0, cut - fixed_train_len)
            tr_idx = np.arange(tr_start, cut)

        te_idx = np.arange(cut, b_end)
        if len(te_idx) == 0:
            assert (
                False
            ), f"No test samples in block {i} after splitting. Adjust train_frac or ensure block sizes are large enough."

        splits.append((tr_idx, te_idx))

    if not splits:
        raise ValueError("No valid splits generated. Adjust k/train_frac.")

    return splits


def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def _build_model(model_name, alpha=None, random_state=42, lasso_max_iter=20000):
    if model_name == "linear":
        return LinearRegression()
    if model_name == "ridge":
        return Ridge(alpha=float(alpha), random_state=random_state)
    # if model_name == "lasso":
    #     return MultiTaskLasso(alpha=float(alpha), max_iter=lasso_max_iter, random_state=random_state)
    raise ValueError(f"Unknown model_name: {model_name}")


def cross_validate_models(
    X,
    y,
    splits,
    ridge_alphas=None,
    lasso_alphas=None,
    random_state=42,
    lasso_max_iter=20000,
):
    if ridge_alphas is None:
        ridge_alphas = np.logspace(-3, 2, 8)
    # if lasso_alphas is None:
    #     lasso_alphas = np.logspace(-5, -2, 6)

    param_grid = {
        "linear": [None],
        "ridge": list(ridge_alphas),
        # "lasso": list(lasso_alphas),
    }

    rows = []
    for model_name, alphas in param_grid.items():
        for alpha in alphas:
            tr_rmse_all, va_rmse_all = [], []
            tr_mae_all, va_mae_all = [], []

            for tr_idx, va_idx in splits:
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_va, y_va = X[va_idx], y[va_idx]

                model = _build_model(
                    model_name,
                    alpha=alpha,
                    random_state=random_state,
                    lasso_max_iter=lasso_max_iter,
                )

                model.fit(X_tr, y_tr)
                y_tr_hat = model.predict(X_tr)
                y_va_hat = model.predict(X_va)

                tr_rmse_all.append(_rmse(y_tr, y_tr_hat))
                va_rmse_all.append(_rmse(y_va, y_va_hat))
                tr_mae_all.append(_mae(y_tr, y_tr_hat))
                va_mae_all.append(_mae(y_va, y_va_hat))

            rows.append(
                {
                    "model": model_name,
                    "alpha": alpha,
                    "train_mae_mean": float(np.mean(tr_mae_all)),
                    "val_mae_mean": float(np.mean(va_mae_all)),
                    "train_rmse_mean": float(np.mean(tr_rmse_all)),
                    "val_rmse_mean": float(np.mean(va_rmse_all)),
                    "n_folds": len(splits),
                }
            )

    return pd.DataFrame(rows)


def plot_train_val_vs_alpha(cv_results, metric="rmse"):
    if metric == "rmse":
        tr_col, va_col, y_title = "train_rmse_mean", "val_rmse_mean", "RMSE"
    elif metric == "mae":
        tr_col, va_col, y_title = "train_mae_mean", "val_mae_mean", "MAE"
    else:
        raise ValueError("metric must be 'rmse' or 'mae'")

    fig = go.Figure()
    for model_name in cv_results["model"].unique():
        sub = cv_results[cv_results["model"] == model_name].copy()

        if model_name == "linear":
            fig.add_trace(
                go.Scatter(
                    x=[1.0],
                    y=sub[tr_col],
                    mode="markers",
                    name="linear train",
                    marker=dict(size=10),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[1.0],
                    y=sub[va_col],
                    mode="markers",
                    name="linear val",
                    marker=dict(size=10, symbol="diamond"),
                )
            )
        else:
            sub = sub.sort_values("alpha")
            fig.add_trace(
                go.Scatter(
                    x=sub["alpha"],
                    y=sub[tr_col],
                    mode="lines+markers",
                    name=f"{model_name} train",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=sub["alpha"],
                    y=sub[va_col],
                    mode="lines+markers",
                    name=f"{model_name} val",
                )
            )

    fig.update_layout(
        template="plotly_white",
        title=f"Train vs Validation {y_title} vs alpha",
        xaxis_title="alpha (log scale; linear shown at alpha=1 marker)",
        yaxis_title=y_title,
        width=1100,
        height=550,
    )
    fig.update_xaxes(type="log")
    return fig


def final_holdout_eval(
    X,
    y,
    cv_results,
    holdout_size=20,
    random_state=42,
    lasso_max_iter=20000,
):
    if holdout_size <= 0 or holdout_size >= len(X):
        raise ValueError("holdout_size must be between 1 and len(X)-1.")

    n_train = len(X) - holdout_size
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]

    best = cv_results.sort_values("val_rmse_mean", ascending=True).iloc[0]

    model = _build_model(
        best["model"],
        alpha=best["alpha"],
        random_state=random_state,
        lasso_max_iter=lasso_max_iter,
    )
    model.fit(X_tr, y_tr)
    y_tr_hat = model.predict(X_tr)
    y_te_hat = model.predict(X_te)

    summary = {
        "best_model": best["model"],
        "best_alpha": best["alpha"],
        "train_rmse": float(_rmse(y_tr, y_tr_hat)),
        "test_rmse": float(_rmse(y_te, y_te_hat)),
    }
    return summary, model, None


def run_surface_linear_experiment(
    long_df,
    lookback_k=5,
    cv_method="walk_forward",  # expanding | walk_forward | tscv
    holdout_size=20,
    n_splits=5,
    train_frac=0.7,
    ridge_alphas=None,
    lasso_alphas=None,
    random_state=42,
    lasso_max_iter=20000,
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

    cv_results = cross_validate_models(
        X_trainval,
        y_trainval,
        splits=splits,
        ridge_alphas=ridge_alphas,
        lasso_alphas=lasso_alphas,
        random_state=random_state,
        lasso_max_iter=lasso_max_iter,
    )

    best_summary, best_model, best_scaler = final_holdout_eval(
        X,
        y,
        cv_results,
        holdout_size=holdout_size,
        random_state=random_state,
        lasso_max_iter=lasso_max_iter,
    )

    fig_rmse = plot_train_val_vs_alpha(cv_results, metric="rmse")
    fig_mae = plot_train_val_vs_alpha(cv_results, metric="mae")

    return {
        "surface_wide": surface_wide,
        "X": X,
        "y": y,
        "y_dates": y_dates,
        "cv_results": cv_results,
        "best_summary": best_summary,
        "best_model": best_model,
        "best_scaler": best_scaler,
        "fig_rmse": fig_rmse,
        "fig_mae": fig_mae,
    }

