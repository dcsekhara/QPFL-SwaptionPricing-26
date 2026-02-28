from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qpfl.data import read_data
from qpfl.models import run_surface_experiment
from qpfl.visualization import (
    plot_maturity_tenor_lines,
    plot_price_vs_date_tenor_maturity_lines,
    plot_tenor_maturity_lines,
)


def main():
    full_df = read_data("DATASETS/train.xlsx", output="long", value_name="price")

    fig, sampled_dates = plot_tenor_maturity_lines(
        full_df, n_days=15, seed=7, max_tenors=10
    )
    fig.show()
    print("Sampled dates:", [pd.Timestamp(d).strftime("%Y-%m-%d") for d in sampled_dates])

    fig, sampled_dates = plot_maturity_tenor_lines(
        full_df, n_days=15, seed=7, max_maturities=12
    )
    fig.show()
    print("Sampled dates:", [pd.Timestamp(d).strftime("%Y-%m-%d") for d in sampled_dates])

    fig = plot_price_vs_date_tenor_maturity_lines(full_df)
    fig.show()

    # Local-safe default configs (should run in this environment).
    model_configs = [
        {"name": "linear"},
        {"name": "ridge", "alpha": 1e-3},
        {"name": "ridge", "alpha": 1e-2},
        {"name": "ridge", "alpha": 1e-1},
        {"name": "ridge", "alpha": 1.0},
        {"name": "ridge", "alpha": 10.0},
        {"name": "ridge", "alpha": 100.0},
        {"name": "mlp", "hidden_layer_sizes": (256, 128), "max_iter": 300, "random_state": 42},
    ]

    # Optional model configs for environments like Colab (enable as needed):
    # model_configs += [
    #     {"name": "xgboost", "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8, "random_state": 42},
    #     {"name": "mlp_torch", "hidden_dims": (512, 256), "epochs": 50, "batch_size": 64, "lr": 1e-3, "random_state": 42},
    #     {"name": "cnn", "conv_channels": 64, "kernel_size": 3, "hidden_dim": 256, "epochs": 50, "batch_size": 64, "lr": 1e-3, "random_state": 42},
    #     {"name": "lstm", "hidden_dim": 256, "num_layers": 1, "dropout": 0.0, "epochs": 50, "batch_size": 64, "lr": 1e-3, "random_state": 42},
    # ]

    results = run_surface_experiment(
        long_df=full_df,
        lookback_k=5,
        cv_method="expanding",
        holdout_size=20,
        n_splits=5,
        train_frac=0.7,
        model_configs=model_configs,
    )

    print(results["cv_results"].sort_values(["val_rmse_mean", "config_id"]))
    print(results["best_summary"])
    results["fig_rmse"].show()
    results["fig_mae"].show()


if __name__ == "__main__":
    main()
