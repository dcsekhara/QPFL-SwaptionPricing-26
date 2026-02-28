# %%
import pandas as pd
import numpy as np
from pathlib import Path

# %%
from unicodedata import numeric


def read_data(
    path,
    sheet_name=0,
    date_col="Date",
    output="long",
    value_name="value",
    engine="openpyxl",
):
    """Read Excel data and parse surface columns into Tenor and Maturity (months)."""
    import re

    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet_name, engine=engine)
    assert date_col in df.columns, f"Expected date column '{date_col}' not found in the data."

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y")

    pattern = re.compile(
        r"^\s*Tenor\s*:\s*(\d+)\s*;\s*Maturity\s*:\s*([0-9]*\.?[0-9]+)\s*$"
    )
    parsed_cols = {}
    parse_ignore_cols = [date_col]
    for col in df.columns:
        if col in parse_ignore_cols:
            continue
        match = pattern.match(str(col))
        if match:
            tenor = int(match.group(1))
            maturity_float = float(match.group(2))
            maturity_months = maturity_float * 12
            int_maturity_months = int(round(maturity_months))
            assert(abs(maturity_months - int_maturity_months) < 1e-6), f"Maturity {maturity_float} years does not convert to an integer number of months."
            maturity_months = int_maturity_months
            parsed_cols[col] = (tenor, maturity_months)
        else:
            assert False, "Unable to parse the column name: " + str(col)

    if not parsed_cols:
        raise ValueError("No columns matched the 'Tenor : x; Maturity : y' format.")

    if output == "wide":
        raise NotImplementedError("Wide format is not implemented yet.")
        renamed = {
            col: f"Tenor_{tenor}_Maturity_{maturity}m"
            for col, (tenor, maturity) in parsed_cols.items()
        }
        return df.rename(columns=renamed)

    if output == "long":
        id_vars = [date_col]
        value_vars = list(parsed_cols.keys())
        long_df = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="swaption_details",
            value_name=value_name,
        )
        long_df[["Tenor", "Maturity"]] = long_df["swaption_details"].map(parsed_cols).apply(pd.Series)
        long_df = long_df.drop(columns=["swaption_details"])
        ordered_cols = id_vars + ["Tenor", "Maturity", value_name]
        long_df = long_df[ordered_cols]
        # Verify that date column Tenor and Maturity columns don't have missing values
        assert long_df[date_col].notna().all(), "Date column contains missing values."
        assert long_df["Tenor"].notna().all(), "Tenor column contains missing values."
        assert long_df["Maturity"].notna().all(), "Maturity column contains missing values."

        # Verify that the price, Tenor and Maturity columns have the expected data types
        assert np.issubdtype(long_df[date_col].dtype, np.datetime64), "Date column is not of datetime type."
        assert np.issubdtype(long_df["Tenor"].dtype, np.integer), "Tenor column is not of integer type."
        assert np.issubdtype(long_df["Maturity"].dtype, np.integer), "Maturity column is not of integer type."
        # The price_var should be numeric (float or integer)
        assert np.issubdtype(long_df[value_name].dtype, np.number), f"{value_name} column is not of numeric type."
        return long_df[ordered_cols]

    raise ValueError("output must be either 'long' or 'wide'")


# %%
df_long = read_data("DATASETS/train.xlsx", output="long", value_name="price")
df_long

# %%
# def train_test_split(df, date_col="Date", test_size=0.2):
#     """
#     Split by unique dates so all rows for a given date stay together.

#     Works for long data (multiple Tenor/Maturity rows per date) and wide data
#     (one row per date).
#     """
#     if date_col not in df.columns:
#         raise ValueError(f"date_col '{date_col}' not found in DataFrame columns.")

#     if not (0 < test_size < 1):
#         raise ValueError("test_size must be a float in (0, 1).")

#     work_df = df.copy()

#     unique_dates = np.array(sorted(work_df[date_col].unique()))
#     n_dates = len(unique_dates)

#     if n_dates < 2:
#         raise ValueError("Need at least 2 unique dates to create train/test split.")

#     n_test_dates = int(np.ceil(n_dates * test_size))
#     n_test_dates = min(n_test_dates, n_dates - 1)

#     test_dates = set(unique_dates[-n_test_dates:])

#     sort_cols = [date_col, "Maturity", "Tenor"]
#     train_df = work_df[~work_df[date_col].isin(test_dates)].sort_values(sort_cols).reset_index(drop=True)
#     test_df = work_df[work_df[date_col].isin(test_dates)].sort_values(sort_cols).reset_index(drop=True)

#     return train_df, test_df


# %%
full_df = read_data("DATASETS/train.xlsx", output="long", value_name="price")
full_df

# %%
# test_frac = 0.3
# train_df, test_df = train_test_split(full_df, test_size=test_frac)

# %%
# train_df

# %%
# test_df

# %%
full_df["Maturity"].unique(), full_df["Tenor"].unique()

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_tenor_maturity_lines(
    long_df,
    n_days=15,
    seed=42,
    max_tenors=10,
    date_col="Date",
    tenor_col="Tenor",
    maturity_col="Maturity",
    price_col="price",
):
    """
    For random days, plot Price vs Maturity with multiple traces (one trace per Tenor).
    Same tenor keeps the same color across all subplots.
    """
    df = long_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, tenor_col, maturity_col, price_col])

    unique_dates = np.array(sorted(df[date_col].unique()))
    if len(unique_dates) == 0:
        raise ValueError("No valid dates available after cleaning.")

    n_days = min(n_days, len(unique_dates))
    rng = np.random.default_rng(seed)
    sampled_dates = np.sort(rng.choice(unique_dates, size=n_days, replace=False))

    n_cols = 3
    n_rows = int(np.ceil(n_days / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in sampled_dates],
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    tenor_values = sorted(df[tenor_col].unique())[:max_tenors]

    # Fixed tenor -> color map
    palette = px.colors.qualitative.Dark24
    tenor_colors = {t: palette[i % len(palette)] for i, t in enumerate(tenor_values)}

    for i, d in enumerate(sampled_dates):
        row = i // n_cols + 1
        col = i % n_cols + 1
        day_df = df[df[date_col] == d]

        for tenor in tenor_values:
            tdf = day_df[day_df[tenor_col] == tenor].sort_values(maturity_col)
            if tdf.empty:
                continue

            color = tenor_colors[tenor]

            fig.add_trace(
                go.Scatter(
                    x=tdf[maturity_col],
                    y=tdf[price_col],
                    mode="lines+markers",
                    name=f"Tenor {tenor}Y",
                    legendgroup=f"tenor_{tenor}",
                    showlegend=(i == 0),
                    line=dict(width=2, color=color),
                    marker=dict(size=5, color=color),
                    hovertemplate=(
                        "Maturity: %{x}m<br>"
                        "Price: %{y:.5f}<br>"
                        f"Tenor: {tenor}Y<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="Maturity (months)", row=row, col=col)
        fig.update_yaxes(title_text="Price", row=row, col=col)

    fig.update_layout(
        height=320 * n_rows,
        width=1200,
        title=f"Price vs Maturity with Multiple Tenor Traces (Random {n_days} Days)",
        template="plotly_white",
        legend_title="Tenor",
    )
    return fig, sampled_dates


# Usage
fig, sampled_dates = plot_tenor_maturity_lines(full_df, n_days=15, seed=7, max_tenors=10)
fig.show()
print("Sampled dates:", [pd.Timestamp(d).strftime("%Y-%m-%d") for d in sampled_dates])


# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_maturity_tenor_lines(
    long_df,
    n_days=15,
    seed=42,
    max_maturities=12,
    date_col="Date",
    tenor_col="Tenor",
    maturity_col="Maturity",
    price_col="price",
):
    """
    For random days, plot Price vs Tenor with multiple traces (one trace per Maturity).
    Same maturity keeps the same color across all subplots.
    """
    df = long_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, tenor_col, maturity_col, price_col])

    unique_dates = np.array(sorted(df[date_col].unique()))
    if len(unique_dates) == 0:
        raise ValueError("No valid dates available after cleaning.")

    n_days = min(n_days, len(unique_dates))
    rng = np.random.default_rng(seed)
    sampled_dates = np.sort(rng.choice(unique_dates, size=n_days, replace=False))

    n_cols = 3
    n_rows = int(np.ceil(n_days / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in sampled_dates],
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    maturity_values = sorted(df[maturity_col].unique())[:max_maturities]

    # Fixed maturity -> color map
    palette = px.colors.qualitative.Dark24
    maturity_colors = {m: palette[i % len(palette)] for i, m in enumerate(maturity_values)}

    for i, d in enumerate(sampled_dates):
        row = i // n_cols + 1
        col = i % n_cols + 1
        day_df = df[df[date_col] == d]

        for maturity in maturity_values:
            mdf = day_df[day_df[maturity_col] == maturity].sort_values(tenor_col)
            if mdf.empty:
                continue

            color = maturity_colors[maturity]

            fig.add_trace(
                go.Scatter(
                    x=mdf[tenor_col],
                    y=mdf[price_col],
                    mode="lines+markers",
                    name=f"Maturity {maturity}m",
                    legendgroup=f"maturity_{maturity}",
                    showlegend=(i == 0),
                    line=dict(width=2, color=color),
                    marker=dict(size=5, color=color),
                    hovertemplate=(
                        "Tenor: %{x}Y<br>"
                        "Price: %{y:.5f}<br>"
                        f"Maturity: {maturity}m<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="Tenor (years)", row=row, col=col)
        fig.update_yaxes(title_text="Price", row=row, col=col)

    fig.update_layout(
        height=320 * n_rows,
        width=1200,
        title=f"Price vs Tenor with Multiple Maturity Traces (Random {n_days} Days)",
        template="plotly_white",
        legend_title="Maturity",
    )
    return fig, sampled_dates


# Usage
fig, sampled_dates = plot_maturity_tenor_lines(full_df, n_days=15, seed=7, max_maturities=12)
fig.show()
print("Sampled dates:", [pd.Timestamp(d).strftime("%Y-%m-%d") for d in sampled_dates])


# %%
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_price_vs_date_tenor_maturity_lines(
    long_df,
    date_col="Date",
    tenor_col="Tenor",
    maturity_col="Maturity",
    price_col="price",
    tenors=None,
    maturities=None,
    max_traces=None,
):
    """
    Plot Price vs Date (all dates), with each trace = one unique (Tenor, Maturity) pair.
    """
    df = long_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, tenor_col, maturity_col, price_col])

    if tenors is not None:
        df = df[df[tenor_col].isin(tenors)]
    if maturities is not None:
        df = df[df[maturity_col].isin(maturities)]

    if df.empty:
        raise ValueError("No data left after filtering.")

    grouped = (
        df.groupby([tenor_col, maturity_col, date_col], as_index=False)[price_col]
        .mean()
        .sort_values(date_col)
    )

    pairs = list(
        grouped[[tenor_col, maturity_col]]
        .drop_duplicates()
        .sort_values([tenor_col, maturity_col])
        .itertuples(index=False, name=None)
    )

    if max_traces is not None:
        pairs = pairs[:max_traces]

    # Fixed color per (Tenor, Maturity) line
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

    fig = go.Figure()

    for i, (tenor, maturity) in enumerate(pairs):
        ts = grouped[
            (grouped[tenor_col] == tenor) & (grouped[maturity_col] == maturity)
        ].sort_values(date_col)

        color = palette[i % len(palette)]

        fig.add_trace(
            go.Scatter(
                x=ts[date_col],
                y=ts[price_col],
                mode="lines",
                name=f"T{tenor}Y_M{maturity}m",
                legendgroup=f"t{tenor}_m{maturity}",
                line=dict(width=1.8, color=color),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Price: %{y:.5f}<br>"
                    f"Tenor: {tenor}Y<br>"
                    f"Maturity: {maturity}m<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Price vs Date - Tenor_Maturity Lines",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=700,
        width=1300,
        legend_title="Tenor_Maturity",
    )
    return fig


# Usage
fig = plot_price_vs_date_tenor_maturity_lines(full_df)  # or long_df
fig.show()

# Optional subset to avoid overcrowding:
# fig = plot_price_vs_date_tenor_maturity_lines(
#     full_df,
#     tenors=[1, 2, 5, 10],
#     maturities=[1, 6, 12, 24, 60, 120],
#     max_traces=30,
# )
# fig.show()


# %% [markdown]
# ### Now go to prediction tasks

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression, Ridge, MultiTaskLasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# %%
# ----------------------------
# 1) Data prep: long -> wide surface per date
# ----------------------------
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

# %%
# ----------------------------
# 2) Build lookback dataset
# X_t = [surface(t-k+1), ..., surface(t)] -> y_t = surface(t+1)
# ----------------------------
def build_lookback_dataset(surface_wide, lookback_k=5):
    V = surface_wide.to_numpy(dtype=float)   # shape: [n_dates, n_points]
    D = surface_wide.index.to_numpy()        # dates

    if len(V) <= lookback_k:
        raise ValueError("Not enough dates for selected lookback_k.")

    X, y, y_dates = [], [], []
    for t in range(lookback_k - 1, len(V) - 1):
        X.append(V[t - lookback_k + 1 : t + 1].reshape(-1))
        y.append(V[t + 1])
        y_dates.append(D[t + 1])

    return np.asarray(X), np.asarray(y), np.asarray(y_dates)

# %%
# ----------------------------
# 3) Time-series CV split generators
# methods: expanding, walk_forward (rolling), tscv
# ----------------------------
import numpy as np

def generate_time_series_splits(
    n_samples: int,
    method: str = "walk_forward",   # "walk_forward"|"rolling"|"expanding"
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

    # 1) First divide full index range into k chronological blocks
    edges = np.linspace(0, n_samples, k + 1, dtype=int)

    splits = []
    fixed_train_len = None  # used for walk_forward/rolling

    for i in range(k):
        b_start, b_end = edges[i], edges[i + 1]
        block_len = b_end - b_start
        if block_len < 2:
            assert False, f"Block {i} is too small for splitting (length={block_len}). Adjust k or ensure n_samples is large enough."

        # 2) Then split within block by train_frac
        tr_len_block = int(np.floor(block_len * train_frac))
        tr_len_block = max(1, min(tr_len_block, block_len - 1))
        cut = b_start + tr_len_block

        if method == "expanding":
            tr_idx = np.arange(0, cut)          # grows with time
        else:  # walk_forward / rolling
            if fixed_train_len is None:
                fixed_train_len = tr_len_block  # keep train window fixed
            tr_start = max(0, cut - fixed_train_len)
            tr_idx = np.arange(tr_start, cut)

        te_idx = np.arange(cut, b_end)
        if len(te_idx) == 0:
            assert False, f"No test samples in block {i} after splitting. Adjust train_frac or ensure block sizes are large enough."

        splits.append((tr_idx, te_idx))

    if not splits:
        raise ValueError("No valid splits generated. Adjust k/train_frac.")

    return splits

# %%
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
    #     # L1 for multi-output (joint)
    #     return MultiTaskLasso(alpha=float(alpha), max_iter=lasso_max_iter, random_state=random_state)
    raise ValueError(f"Unknown model_name: {model_name}")

# %%
# ----------------------------
# 4) Cross-validation over models/params
# ----------------------------
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
            # tr_r2_all, va_r2_all = [], []
            tr_rmse_all, va_rmse_all = [], []
            tr_mae_all, va_mae_all = [], []

            for tr_idx, va_idx in splits:
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_va, y_va = X[va_idx], y[va_idx]

                # scaler = StandardScaler()
                # X_tr_s = scaler.fit_transform(X_tr)
                # X_va_s = scaler.transform(X_va)

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


# %%
# ----------------------------
# 5) Plot train vs validation as params vary
# ----------------------------
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
            # one-point model (no alpha)
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

# %%
# ----------------------------
# 6) Final holdout evaluation
# ----------------------------
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

    # scaler = StandardScaler()
    # X_tr_s = scaler.fit_transform(X_tr)
    # X_te_s = scaler.transform(X_te)

    model.fit(X_tr, y_tr)
    y_tr_hat = model.predict(X_tr)
    y_te_hat = model.predict(X_te)

    summary = {
        "best_model": best["model"],
        "best_alpha": best["alpha"],
        "train_rmse": float(_rmse(y_tr, y_tr_hat)),
        "test_rmse": float(_rmse(y_te, y_te_hat)),
    }
    return summary, model, None  # scaler not returned for now


# %%
# ----------------------------
# 7) One-shot runner
# ----------------------------
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

    # keep final holdout untouched
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
        X, y, cv_results,
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


# %%
# ============================
# USAGE
# ============================
# long_df must have: Date, Tenor, Maturity, price
# Example:
results = run_surface_linear_experiment(
    long_df=full_df,
    lookback_k=5,
    cv_method="expanding",   # "walk_forward" or "tscv" also supported
    holdout_size=20,
    n_splits=5,
    train_frac=0.7,
    ridge_alphas=np.logspace(-3, 1, 6),
    lasso_alphas=None,
    lasso_max_iter=20000,
)

print(results["cv_results"].sort_values(["model", "alpha"]))
print(results["best_summary"])
results["fig_rmse"].show()
results["fig_mae"].show()


# %%
np.linspace(0, 490, 11, dtype=int)


