import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error


@dataclass
class Config:
    train_path: str = "data/train.xlsx"
    template_path: str = "data/test_template.xlsx"
    output_dir: str = "outputs"

    # Grid search
    n_components_grid: Tuple[int, ...] = (1, 2, 3, 4, 5, 8, 10, 15, 20, 40, 50, 80, 100)
    windows: Tuple[int, ...] = (5, 10, 15, 20, 30, 40)
    alphas: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

    val_ratio: float = 0.2


def build_supervised(values: np.ndarray, window: int):
    """values: (N, d) -> X: (N-window, window*d), Y: (N-window, d)"""
    N, d = values.shape
    X = np.zeros((N - window, window * d), dtype=float)
    Y = np.zeros((N - window, d), dtype=float)
    for i, t in enumerate(range(window, N)):
        X[i] = values[t - window:t].reshape(-1)
        Y[i] = values[t]
    return X, Y


def fit_surface_scaler_pca(train_prices: np.ndarray, n_components: int):
    """Fit scaler + PCA on 224-dim surfaces. Return (scaler_surface, pca, train_pca, explained)."""
    scaler_surface = StandardScaler()
    train_scaled = scaler_surface.fit_transform(train_prices)
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_scaled)
    explained = float(np.sum(pca.explained_variance_ratio_))
    return scaler_surface, pca, train_pca, explained


def reconstruct_prices(pca_preds: np.ndarray, scaler_surface: StandardScaler, pca: PCA) -> np.ndarray:
    scaled = pca.inverse_transform(pca_preds)
    prices = scaler_surface.inverse_transform(scaled)
    return prices


def train_eval_ridge(X_tr, Y_tr, X_va, Y_va, alpha: float):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])
    model.fit(X_tr, Y_tr)
    pred = model.predict(X_va)
    mse = mean_squared_error(Y_va, pred)
    mae = mean_absolute_error(Y_va, pred)
    return model, float(mse), float(mae)


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load
    train = pd.read_excel(cfg.train_path)
    template = pd.read_excel(cfg.template_path)

    train["Date"] = pd.to_datetime(train["Date"], dayfirst=True)
    template["Date"] = pd.to_datetime(template["Date"], dayfirst=True)

    if "Type" not in template.columns:
        raise ValueError("Template must contain a 'Type' column.")

    price_cols = [c for c in train.columns if c not in ["Date", "Type"]]
    train = train.sort_values("Date").reset_index(drop=True)

    train_prices = train[price_cols].astype(float).values  # (N, 224)
    N = train_prices.shape[0]
    val_size = int(np.ceil(cfg.val_ratio * N))
    split_idx = N - val_size  # split on raw timeline

    best: Dict[str, Any] = {}

    print(f"N={N}, val_size={val_size}, split_idx={split_idx}\n")

    # GRID SEARCH
    for n_comp in cfg.n_components_grid:
        scaler_surface, pca, train_pca, explained = fit_surface_scaler_pca(train_prices, n_components=n_comp)
        print(f"=== PCA n_components={n_comp} explained={explained:.6f} ===")

        for window in cfg.windows:
            X_all, Y_all = build_supervised(train_pca, window=window)

            # map split point into supervised rows
            split_row = split_idx - window
            if split_row <= 10 or split_row >= len(X_all) - 10:
                # not enough room on either side
                continue

            X_tr, Y_tr = X_all[:split_row], Y_all[:split_row]
            X_va, Y_va = X_all[split_row:], Y_all[split_row:]

            for alpha in cfg.alphas:
                model, mse, mae = train_eval_ridge(X_tr, Y_tr, X_va, Y_va, alpha=alpha)
                print(f"n_comp={n_comp:>2} window={window:>2} alpha={alpha:>7} -> MSE={mse:.6e} MAE={mae:.6e}")

                if not best or mse < best["mse"]:
                    best = {
                        "n_components": n_comp,
                        "explained": explained,
                        "window": window,
                        "alpha": alpha,
                        "mse": mse,
                        "mae": mae,
                        # store fitted PCA objects so we can reuse them
                        "scaler_surface": scaler_surface,
                        "pca": pca,
                        "train_pca": train_pca,
                    }

        print("")

    if not best:
        raise RuntimeError("Grid search failed to produce a best model.")

    print("🏆 BEST (by MSE):", {k: best[k] for k in ["n_components","explained","window","alpha","mse","mae"]})

    # TRAIN FINAL MODEL ON FULL DATA (PCA already fit on full surfaces)
    n_comp = best["n_components"]
    window = best["window"]
    alpha = best["alpha"]
    scaler_surface = best["scaler_surface"]
    pca = best["pca"]
    train_pca = best["train_pca"]  # (N, n_comp)

    X_full, Y_full = build_supervised(train_pca, window=window)
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])
    final_model.fit(X_full, Y_full)

    # BUILD SUBMISSION
    out_df = template.copy()

    future_mask = out_df["Type"].astype(str).str.contains("Future", case=False, na=False)
    missing_mask = out_df["Type"].astype(str).str.contains("Missing", case=False, na=False)

    future_dates = out_df.loc[future_mask, "Date"].tolist()  # keep template order
    missing_dates = out_df.loc[missing_mask, "Date"].tolist()

    # FUTURE: iterative steps = number of future rows
    buffer = train_pca[-window:].copy()
    future_preds_pca = []
    for _ in range(len(future_dates)):
        yhat = final_model.predict(buffer.reshape(1, -1))[0]
        future_preds_pca.append(yhat)
        buffer = np.vstack([buffer[1:], yhat])
    future_preds_pca = np.array(future_preds_pca)
    future_prices = reconstruct_prices(future_preds_pca, scaler_surface, pca)

    # MISSING: one-step impute using preceding window rows
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(train["Date"])}
    missing_prices_map = {}
    for d in missing_dates:
        d = pd.Timestamp(d)
        if d not in date_to_idx:
            raise ValueError(f"Missing date {d.date()} not found in train.")
        idx = date_to_idx[d]
        if idx < window:
            raise ValueError(f"Not enough history to impute {d.date()} with window={window}.")
        past = train_pca[idx - window: idx]
        yhat = final_model.predict(past.reshape(1, -1))[0]
        missing_prices_map[d] = reconstruct_prices(yhat.reshape(1, -1), scaler_surface, pca)[0]

    # Fill
    fut_i = 0
    for i, row in out_df.iterrows():
        t = str(row["Type"])
        d = pd.Timestamp(row["Date"])
        if "Future" in t or "future" in t:
            out_df.loc[i, price_cols] = future_prices[fut_i]
            fut_i += 1
        elif "Missing" in t or "missing" in t:
            out_df.loc[i, price_cols] = missing_prices_map[d]

    if out_df[price_cols].isna().any().any():
        bad = out_df[out_df[price_cols].isna().any(axis=1)][["Date", "Type"]]
        raise ValueError(f"NaNs remain after filling:\n{bad}")

    # Save with informative name
    out_name = f"submission_best_PC{n_comp}_W{window}_A{alpha}.xlsx"
    out_path = os.path.join(cfg.output_dir, out_name)
    out_df.to_excel(out_path, index=False)
    print(f"✅ Saved submission to: {out_path}")


if __name__ == "__main__":
    main()