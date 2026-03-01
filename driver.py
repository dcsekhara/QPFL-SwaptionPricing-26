from __future__ import annotations

import pandas as pd

from lib import (
    FrozenCNNLatentFeatureExtractor,
    IdentityFeatureExtractor,
    SurfaceDatasetBuilder,
    TrainingDriver,
)
from lib.model_catalog import (
    baseline_feature_specs,
    baseline_model_specs,
    build_experiment_specs,
    cnn_model_specs,
    quantum_reservoir_model_specs,
    xgboost_model_specs,
)
from lib.torch_models import is_torch_available


def summarize_run(model_name: str, result) -> dict:
    fold_df = result.cv_fold_metrics
    row = {
        "model": model_name,
    }
    for col in fold_df.columns:
        if col.startswith("train_") or col.startswith("val_"):
            row[f"cv_{col}_mean"] = float(fold_df[col].mean())
    for metric_name, metric_value in result.holdout_metrics.items():
        row[metric_name] = float(metric_value)
    return row


def main() -> None:
    builder = SurfaceDatasetBuilder.from_excel("data/train.xlsx")

    lookback_window = 5
    include_static_features = True
    xgboost_search_mode = "standard"  # quick | standard | exhaustive
    include_cnn = True
    include_quantum = False
    include_quantum_with_cnn_latent = True
    quantum_k_neighbors = 5

    driver = TrainingDriver(
        dataset_builder=builder,
        lookback_window=lookback_window,
        holdout_size=20,
        num_splits=5,
        cross_val_frac=0.2,
        include_static_features=include_static_features,
    )

    ridge_alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    model_specs = baseline_model_specs(ridge_alphas)
    try:
        model_specs.extend(
            xgboost_model_specs(
                search_mode=xgboost_search_mode,
                random_state=42,
                n_jobs=-1,
            )
        )
    except ImportError as exc:
        print(f"[WARN] Skipping XGBoost models: {exc}")

    if include_cnn:
        if is_torch_available():
            model_specs.extend(
                cnn_model_specs(
                    epochs=120,
                    batch_size=64,
                    lr=1e-3,
                    conv_channels_grid=[64],
                    kernel_size_grid=[3],
                    hidden_dim_grid=[256],
                    weight_decay_grid=[0.0],
                    random_state=42,
                )
            )
        else:
            print("[WARN] Skipping CNN models: PyTorch is not installed in this environment.")

    base_feature_dim = lookback_window + (4 if include_static_features else 0)
    pca_candidates = [3, 5, 7]
    pca_components = [k for k in pca_candidates if 1 <= k < base_feature_dim]

    # Always includes "raw" + optional PCA variants.
    feature_specs = baseline_feature_specs(pca_components=pca_components)
    experiment_specs = build_experiment_specs(model_specs, feature_specs)
    print(f"Running {len(experiment_specs)} experiments (models x feature extractors)")

    rows = []
    result_by_name = {}
    for spec in experiment_specs:
        driver.feature_extractor = spec.feature_extractor
        result = driver.run(spec.model)
        model_name = spec.name
        result_by_name[model_name] = result
        rows.append(summarize_run(model_name, result))

    if include_quantum:
        quantum_specs = quantum_reservoir_model_specs(
            lookback_window=lookback_window,
            n_values=quantum_k_neighbors,
            strict_backend=True,
            include_tail_features=False,
        )
        quantum_driver = TrainingDriver(
            dataset_builder=builder,
            lookback_window=lookback_window,
            holdout_size=20,
            num_splits=5,
            cross_val_frac=0.2,
            include_static_features=False,
            dataset_factory=lambda b, lw, include_static: b.create_neighbor_supervised_dataset(
                lookback_window=lw,
                k_neighbors=quantum_k_neighbors,
                include_center=True,
                include_static_features=False,
            ),
            feature_extractor=IdentityFeatureExtractor(),
        )
        for spec in quantum_specs:
            result = quantum_driver.run(spec.model)
            rows.append(summarize_run(f"{spec.name}__neighbor{quantum_k_neighbors}", result))

    if include_quantum_with_cnn_latent:
        cnn_rows = [r for r in rows if str(r["model"]).startswith("cnn_")]
        if not cnn_rows:
            print("[WARN] Skipping CNN-latent quantum stage: no CNN result was available.")
        else:
            best_cnn_row = sorted(cnn_rows, key=lambda r: r["holdout_rmse"])[0]
            best_cnn_name = str(best_cnn_row["model"])
            best_cnn_result = result_by_name[best_cnn_name]
            best_cnn_model = best_cnn_result.final_model

            latent_extractor = FrozenCNNLatentFeatureExtractor(
                cnn_model=best_cnn_model,
                layer_index=-2,  # hidden activation before output layer
            )

            quantum_latent_specs = quantum_reservoir_model_specs(
                lookback_window=lookback_window,
                n_values=None,
                strict_backend=False,
                include_tail_features=True,
            )
            quantum_latent_driver = TrainingDriver(
                dataset_builder=builder,
                lookback_window=lookback_window,
                holdout_size=20,
                num_splits=5,
                cross_val_frac=0.2,
                include_static_features=include_static_features,
                feature_extractor=latent_extractor,
            )
            print(
                f"Running {len(quantum_latent_specs)} quantum models on CNN latent features "
                f"(source CNN={best_cnn_name})"
            )
            for spec in quantum_latent_specs:
                result = quantum_latent_driver.run(spec.model)
                rows.append(summarize_run(f"{spec.name}__cnn_latent", result))

    results_df = pd.DataFrame(rows).sort_values("holdout_rmse").reset_index(drop=True)
    pd.set_option("display.max_columns", None)
    print(results_df)

    best = results_df.iloc[0]
    print("\nBest model:")
    best_cols = [c for c in ("model", "holdout_rmse", "holdout_mae", "holdout_qlike") if c in best.index]
    print(best[best_cols])


if __name__ == "__main__":
    main()
