from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Generator, Optional, Tuple, Union

import numpy as np
import pandas as pd


SURFACE_COL_PATTERN = re.compile(
    r"^\s*Tenor\s*:\s*(\d+)\s*;\s*Maturity\s*:\s*([0-9]*\.?[0-9]+)\s*$"
)


@dataclass(frozen=True)
class SurfaceFrame:
    """In-memory representation of surface data organized by day and surface point."""

    dates: np.ndarray
    values: np.ndarray
    tenors: np.ndarray
    maturities: np.ndarray
    column_names: Tuple[str, ...]

    @property
    def num_days(self) -> int:
        return int(self.values.shape[0])

    @property
    def num_points(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True)
class SupervisedDataset:
    """Supervised samples built from lookback windows."""

    X: np.ndarray
    y: np.ndarray
    sample_day_idx: np.ndarray
    sample_col_idx: np.ndarray
    target_dates: np.ndarray
    tenors: np.ndarray
    maturities: np.ndarray
    lookback_window: int
    num_points: int


@dataclass(frozen=True)
class WalkForwardSplit:
    split_id: int
    train_day_indices: np.ndarray
    val_day_indices: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray


class ExcelSurfaceLoader:
    """Read and parse train Excel file into a consistent surface matrix."""

    def __init__(
        self,
        file_path: Union[str, Path] = "data/train.xlsx",
        date_col: str = "Date",
        sheet_name: Union[int, str] = 0,
        engine: str = "openpyxl",
    ) -> None:
        self.file_path = Path(file_path)
        self.date_col = date_col
        self.sheet_name = sheet_name
        self.engine = engine

    def load(self) -> SurfaceFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Could not find file: {self.file_path}")

        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name, engine=self.engine)
        if self.date_col not in df.columns:
            raise ValueError(f"Expected date column '{self.date_col}' not found.")

        df[self.date_col] = pd.to_datetime(df[self.date_col], dayfirst=True)
        feature_cols = [c for c in df.columns if c != self.date_col]
        if not feature_cols:
            raise ValueError("No surface columns found in the file.")

        tenors = []
        maturities = []
        for col in feature_cols:
            match = SURFACE_COL_PATTERN.match(str(col))
            if match is None:
                raise ValueError(f"Unable to parse surface column name: {col}")

            tenor = int(match.group(1))
            maturity_years = float(match.group(2))
            maturity_months_f = maturity_years * 12.0
            maturity_months = int(round(maturity_months_f))
            if abs(maturity_months_f - maturity_months) > 1e-6:
                raise ValueError(
                    f"Maturity {maturity_years} years does not map cleanly to integer months."
                )

            tenors.append(tenor)
            maturities.append(maturity_months)

        values = df[feature_cols].to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError("Input surface matrix contains NaN values.")

        dates = df[self.date_col].to_numpy(dtype="datetime64[ns]")
        return SurfaceFrame(
            dates=dates,
            values=values,
            tenors=np.asarray(tenors, dtype=np.int32),
            maturities=np.asarray(maturities, dtype=np.int32),
            column_names=tuple(str(c) for c in feature_cols),
        )


class SurfaceDatasetBuilder:
    """Build train/validation/holdout data and neighbor-based features."""

    def __init__(self, surface: SurfaceFrame) -> None:
        self.surface = surface
        self._coord = np.column_stack((surface.tenors, surface.maturities)).astype(float)

    @classmethod
    def from_excel(cls, file_path: Union[str, Path] = "data/train.xlsx") -> "SurfaceDatasetBuilder":
        return cls(ExcelSurfaceLoader(file_path=file_path).load())

    def create_supervised_dataset(
        self,
        lookback_window: int,
        include_static_features: bool = True,
    ) -> SupervisedDataset:
        if lookback_window <= 0:
            raise ValueError("lookback_window must be > 0")

        n_days = self.surface.num_days
        n_points = self.surface.num_points
        if lookback_window >= n_days:
            raise ValueError(
                f"lookback_window ({lookback_window}) must be smaller than num_days ({n_days})."
            )

        n_target_days = n_days - lookback_window
        n_samples = n_target_days * n_points

        static_dim = 4 if include_static_features else 0
        X = np.empty((n_samples, lookback_window + static_dim), dtype=np.float32)
        y = np.empty(n_samples, dtype=np.float32)
        sample_day_idx = np.empty(n_samples, dtype=np.int32)
        sample_col_idx = np.empty(n_samples, dtype=np.int32)
        target_dates = np.empty(n_samples, dtype="datetime64[ns]")
        tenors = np.empty(n_samples, dtype=np.int32)
        maturities = np.empty(n_samples, dtype=np.int32)

        write_idx = 0
        for day_idx in range(lookback_window, n_days):
            history = self.surface.values[day_idx - lookback_window : day_idx, :]
            target = self.surface.values[day_idx, :]

            for col_idx in range(n_points):
                history_feat = history[:, col_idx]
                if include_static_features:
                    tenor = self.surface.tenors[col_idx]
                    maturity = self.surface.maturities[col_idx]
                    X[write_idx, :] = np.concatenate(
                        (
                            history_feat,
                            np.array(
                                [
                                    float(day_idx),
                                    float(tenor),
                                    float(maturity),
                                    float(tenor * maturity),
                                ],
                                dtype=np.float32,
                            ),
                        )
                    )
                else:
                    X[write_idx, :] = history_feat

                y[write_idx] = target[col_idx]
                sample_day_idx[write_idx] = day_idx
                sample_col_idx[write_idx] = col_idx
                target_dates[write_idx] = self.surface.dates[day_idx]
                tenors[write_idx] = self.surface.tenors[col_idx]
                maturities[write_idx] = self.surface.maturities[col_idx]
                write_idx += 1

        return SupervisedDataset(
            X=X,
            y=y,
            sample_day_idx=sample_day_idx,
            sample_col_idx=sample_col_idx,
            target_dates=target_dates,
            tenors=tenors,
            maturities=maturities,
            lookback_window=lookback_window,
            num_points=n_points,
        )

    def _neighbor_cols_for_point(self, col_idx: int, k_neighbors: int, include_center: bool) -> np.ndarray:
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be > 0")
        if k_neighbors > self.surface.num_points:
            raise ValueError(
                f"k_neighbors ({k_neighbors}) cannot exceed num_points ({self.surface.num_points})."
            )

        center = self._coord[col_idx]
        deltas = self._coord - center
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        order = np.argsort(distances)
        if include_center:
            return order[:k_neighbors]
        return order[1 : 1 + k_neighbors]

    def create_neighbor_supervised_dataset(
        self,
        lookback_window: int,
        k_neighbors: int,
        include_center: bool = True,
        include_static_features: bool = False,
    ) -> SupervisedDataset:
        """
        Create supervised dataset using k nearest tenor/maturity neighbors per point.
        Feature layout per sample:
        [lookback_window * k_neighbors (+ optional 4 static features)]
        """
        if lookback_window <= 0:
            raise ValueError("lookback_window must be > 0")

        n_days = self.surface.num_days
        n_points = self.surface.num_points
        if lookback_window >= n_days:
            raise ValueError(
                f"lookback_window ({lookback_window}) must be smaller than num_days ({n_days})."
            )

        n_target_days = n_days - lookback_window
        n_samples = n_target_days * n_points

        static_dim = 4 if include_static_features else 0
        X = np.empty((n_samples, lookback_window * k_neighbors + static_dim), dtype=np.float32)
        y = np.empty(n_samples, dtype=np.float32)
        sample_day_idx = np.empty(n_samples, dtype=np.int32)
        sample_col_idx = np.empty(n_samples, dtype=np.int32)
        target_dates = np.empty(n_samples, dtype="datetime64[ns]")
        tenors = np.empty(n_samples, dtype=np.int32)
        maturities = np.empty(n_samples, dtype=np.int32)

        neighbors_by_col = [
            self._neighbor_cols_for_point(col_idx=i, k_neighbors=k_neighbors, include_center=include_center)
            for i in range(n_points)
        ]

        write_idx = 0
        for day_idx in range(lookback_window, n_days):
            history = self.surface.values[day_idx - lookback_window : day_idx, :]
            target = self.surface.values[day_idx, :]

            for col_idx in range(n_points):
                nbr_cols = neighbors_by_col[col_idx]
                history_neighbors = history[:, nbr_cols].reshape(-1).astype(np.float32)
                if include_static_features:
                    tenor = self.surface.tenors[col_idx]
                    maturity = self.surface.maturities[col_idx]
                    X[write_idx, :] = np.concatenate(
                        (
                            history_neighbors,
                            np.array(
                                [
                                    float(day_idx),
                                    float(tenor),
                                    float(maturity),
                                    float(tenor * maturity),
                                ],
                                dtype=np.float32,
                            ),
                        )
                    )
                else:
                    X[write_idx, :] = history_neighbors

                y[write_idx] = target[col_idx]
                sample_day_idx[write_idx] = day_idx
                sample_col_idx[write_idx] = col_idx
                target_dates[write_idx] = self.surface.dates[day_idx]
                tenors[write_idx] = self.surface.tenors[col_idx]
                maturities[write_idx] = self.surface.maturities[col_idx]
                write_idx += 1

        return SupervisedDataset(
            X=X,
            y=y,
            sample_day_idx=sample_day_idx,
            sample_col_idx=sample_col_idx,
            target_dates=target_dates,
            tenors=tenors,
            maturities=maturities,
            lookback_window=lookback_window,
            num_points=n_points,
        )

    def get_holdout_indices(
        self,
        dataset: SupervisedDataset,
        holdout_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if holdout_size <= 0:
            raise ValueError("holdout_size must be > 0")

        unique_days = np.unique(dataset.sample_day_idx)
        if holdout_size >= unique_days.size:
            raise ValueError(
                f"holdout_size ({holdout_size}) must be smaller than available target days ({unique_days.size})."
            )

        holdout_days = unique_days[-holdout_size:]
        trainval_days = unique_days[:-holdout_size]

        holdout_idx = np.where(np.isin(dataset.sample_day_idx, holdout_days))[0]
        trainval_idx = np.where(np.isin(dataset.sample_day_idx, trainval_days))[0]
        return trainval_idx, holdout_idx

    def iter_walk_forward_splits(
        self,
        dataset: SupervisedDataset,
        holdout_size: int,
        num_splits: int,
        cross_val_frac: float,
    ) -> Generator[WalkForwardSplit, None, None]:
        if num_splits <= 0:
            raise ValueError("num_splits must be > 0")
        if not (0.0 < cross_val_frac < 1.0):
            raise ValueError("cross_val_frac must be in (0, 1)")

        trainval_idx, _ = self.get_holdout_indices(dataset, holdout_size=holdout_size)
        trainval_days = np.unique(dataset.sample_day_idx[trainval_idx])

        if trainval_days.size < 3:
            raise ValueError("Not enough days for walk-forward splitting.")

        prev_horizon_end = 0
        for split_id in range(num_splits):
            # Expanding horizon end point.
            horizon_end = int(np.floor((split_id + 1) * trainval_days.size / num_splits))
            horizon_end = max(2, min(horizon_end, trainval_days.size))
            if horizon_end <= prev_horizon_end:
                continue
            prev_horizon_end = horizon_end

            current_days = trainval_days[:horizon_end]
            val_size = max(1, int(np.floor(current_days.size * cross_val_frac)))
            val_size = min(val_size, current_days.size - 1)

            train_days = current_days[:-val_size]
            val_days = current_days[-val_size:]

            train_indices = np.where(np.isin(dataset.sample_day_idx, train_days))[0]
            val_indices = np.where(np.isin(dataset.sample_day_idx, val_days))[0]

            if train_indices.size == 0 or val_indices.size == 0:
                continue

            yield WalkForwardSplit(
                split_id=split_id,
                train_day_indices=train_days,
                val_day_indices=val_days,
                train_indices=train_indices,
                val_indices=val_indices,
            )

    def get_split_arrays(
        self,
        dataset: SupervisedDataset,
        split: WalkForwardSplit,
    ) -> Dict[str, np.ndarray]:
        return {
            "X_train": dataset.X[split.train_indices],
            "y_train": dataset.y[split.train_indices],
            "X_val": dataset.X[split.val_indices],
            "y_val": dataset.y[split.val_indices],
        }

    def get_holdout_arrays(
        self,
        dataset: SupervisedDataset,
        holdout_size: int,
    ) -> Dict[str, np.ndarray]:
        _, holdout_idx = self.get_holdout_indices(dataset, holdout_size=holdout_size)
        return {
            "X_holdout": dataset.X[holdout_idx],
            "y_holdout": dataset.y[holdout_idx],
            "holdout_indices": holdout_idx,
        }

    def _resolve_col_index(
        self,
        col: Optional[int] = None,
        tenor: Optional[int] = None,
        maturity: Optional[int] = None,
    ) -> int:
        if col is not None:
            if not (0 <= col < self.surface.num_points):
                raise IndexError(f"col out of range: {col}")
            return int(col)

        if tenor is None or maturity is None:
            raise ValueError("Provide either col or (tenor, maturity)")

        hits = np.where(
            (self.surface.tenors == int(tenor)) & (self.surface.maturities == int(maturity))
        )[0]
        if hits.size == 0:
            raise ValueError(f"No surface point found for tenor={tenor}, maturity={maturity}")
        return int(hits[0])

    def neighbor_feature_vector(
        self,
        row: int,
        lookback_window: int,
        k_neighbors: int,
        col: Optional[int] = None,
        tenor: Optional[int] = None,
        maturity: Optional[int] = None,
        include_center: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Build neighbor-based feature vector for a point at a target row.

        row: target day index (the day to predict).
        lookback_window: number of previous days to include.
        k_neighbors: number of nearest neighbors in tenor/maturity space.
        col or (tenor, maturity): anchor surface point.
        """
        if row <= 0 or row >= self.surface.num_days:
            raise IndexError(f"row must be in [1, {self.surface.num_days - 1}] but got {row}")
        if lookback_window <= 0:
            raise ValueError("lookback_window must be > 0")
        if row - lookback_window < 0:
            raise ValueError(
                f"Not enough history for row={row} and lookback_window={lookback_window}."
            )
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be > 0")
        if k_neighbors > self.surface.num_points:
            raise ValueError(
                f"k_neighbors ({k_neighbors}) cannot exceed num_points ({self.surface.num_points})."
            )

        center_col = self._resolve_col_index(col=col, tenor=tenor, maturity=maturity)

        center = self._coord[center_col]
        deltas = self._coord - center
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))

        order = np.argsort(distances)
        if include_center:
            neighbors = order[:k_neighbors]
        else:
            neighbors = order[1 : 1 + k_neighbors]

        history = self.surface.values[row - lookback_window : row, :]
        neighbor_values = history[:, neighbors]

        feature_vector = neighbor_values.reshape(-1).astype(np.float32)
        return {
            "feature_vector": feature_vector,
            "neighbor_col_indices": neighbors.astype(np.int32),
            "neighbor_tenors": self.surface.tenors[neighbors].astype(np.int32),
            "neighbor_maturities": self.surface.maturities[neighbors].astype(np.int32),
            "target_value": np.array([self.surface.values[row, center_col]], dtype=np.float32),
            "target_date": np.array([self.surface.dates[row]], dtype="datetime64[ns]"),
            "row": np.array([row], dtype=np.int32),
            "center_col": np.array([center_col], dtype=np.int32),
        }
