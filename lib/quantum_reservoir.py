from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge

from .training_pipeline import ModelInterface

import perceval as pcvl

from merlin.builder import CircuitBuilder

@dataclass
class ReservoirConfig:
    n_values: int = 5
    data_modes: Optional[int] = None
    hidden_modes: int = 6
    lookback_window: int = 3
    depth: int = 12
    alpha: float = 1.0
    state_nonlinearity: float = 1.0
    seed: int = 7
    strict_backend: bool = False

    def __post_init__(self) -> None:
        if self.data_modes is None:
            self.data_modes = 2 * self.n_values

    @property
    def total_modes(self) -> int:
        return int(self.data_modes) + self.hidden_modes


class QuantumReservoir:
    """Fixed random reservoir with sequential updates."""

    def __init__(self, cfg: ReservoirConfig):
        if int(cfg.data_modes) != 2 * cfg.n_values:
            raise ValueError("data_modes must equal 2 * n_values for dual-rail encoding")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        if cfg.strict_backend and (pcvl is None or CircuitBuilder is None):
            raise RuntimeError("Perceval/Merlin are required but not available.")

        self._mix = self._build_merlin_perceval_unitary()

    def _haar_unitary(self, n: int) -> np.ndarray:
        a = self.rng.normal(size=(n, n)) + 1j * self.rng.normal(size=(n, n))
        q, r = np.linalg.qr(a)
        d = np.diag(r)
        ph = d / np.abs(d)
        return q * ph

    def _build_merlin_perceval_unitary(self) -> np.ndarray:
        n = self.cfg.total_modes
        if pcvl is None or CircuitBuilder is None:
            return self._haar_unitary(n)

        builder = CircuitBuilder(n_modes=n)
        for i in range(max(1, self.cfg.depth)):
            builder.add_entangling_layer(trainable=False, name=f"E{i}")

        c = pcvl.Circuit(n)
        c.add(0, builder.to_pcvl_circuit())

        for _ in range(max(1, self.cfg.depth // 2)):
            layer = pcvl.Circuit(n)
            for m in range(n):
                theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
                layer.add(m, pcvl.PS(theta))
            c.add(0, layer)

        u_rand = self._haar_unitary(n)
        c.add(0, pcvl.Unitary(pcvl.Matrix(u_rand)))
        return np.array(c.compute_unitary(), dtype=np.complex128)

    def angle_encode(self, values: np.ndarray) -> np.ndarray:
        if values.shape != (self.cfg.n_values,):
            raise ValueError(f"Expected shape ({self.cfg.n_values},), got {values.shape}")

        theta = np.pi * np.tanh(self.cfg.alpha * values)
        amps = np.zeros(int(self.cfg.data_modes), dtype=np.float64)
        for i, th in enumerate(theta):
            amps[2 * i] = np.sin(th)
            amps[2 * i + 1] = np.cos(th)
        norm = np.linalg.norm(amps)
        if norm > 0:
            amps /= norm
        return amps

    def _step(self, values: np.ndarray, hidden: np.ndarray):
        data = self.angle_encode(values)
        full = np.concatenate([data, hidden], axis=0).astype(np.complex128)
        mixed = self._mix @ full
        intensities = np.abs(mixed) ** 2
        hidden_raw = np.real(mixed[int(self.cfg.data_modes) :])
        hidden_next = np.tanh(self.cfg.state_nonlinearity * hidden_raw)
        data_meas = intensities[: int(self.cfg.data_modes)]
        all_meas = intensities
        return data_meas, all_meas, hidden_next

    def transform_sequence(self, seq: np.ndarray) -> np.ndarray:
        expected = (self.cfg.lookback_window, self.cfg.n_values)
        if seq.shape != expected:
            raise ValueError(f"Expected seq shape {expected}, got {seq.shape}")

        hidden = np.zeros(self.cfg.hidden_modes, dtype=np.float64)
        data_meas_steps = []
        final_all = None
        for i in range(self.cfg.lookback_window):
            data_meas, all_meas, hidden = self._step(seq[i], hidden)
            data_meas_steps.append(data_meas)
            final_all = all_meas
        if final_all is None:
            raise RuntimeError("Reservoir produced no output.")
        return np.concatenate(data_meas_steps + [final_all], axis=0).astype(np.float64)

    @property
    def feature_dim(self) -> int:
        return self.cfg.lookback_window * int(self.cfg.data_modes) + self.cfg.total_modes

    @property
    def backend_info(self) -> str:
        return f"perceval={'yes' if pcvl is not None else 'no'}, merlin={'yes' if CircuitBuilder is not None else 'no'}"


class QuantumReservoirRegressor(ModelInterface):
    """
    Reservoir feature map + linear readout.
    It can consume nearest-neighbor sequence inputs when X is arranged as:
    [lookback_window * n_values (+ optional tail features)].
    """

    supports_batch_prediction: bool = True

    def __init__(
        self,
        lookback_window: int,
        n_values: Optional[int] = None,
        hidden_modes: int = 6,
        depth: int = 12,
        alpha: float = 1.0,
        state_nonlinearity: float = 1.0,
        seed: int = 7,
        strict_backend: bool = False,
        ridge_alpha: float = 1e-3,
        include_tail_features: bool = True,
    ) -> None:
        self.lookback_window = int(lookback_window)
        self.n_values = n_values
        self.hidden_modes = int(hidden_modes)
        self.depth = int(depth)
        self.alpha = float(alpha)
        self.state_nonlinearity = float(state_nonlinearity)
        self.seed = int(seed)
        self.strict_backend = bool(strict_backend)
        self.ridge_alpha = float(ridge_alpha)
        self.include_tail_features = bool(include_tail_features)

        self._reservoir: Optional[QuantumReservoir] = None
        self._readout = Ridge(alpha=self.ridge_alpha)
        self._resolved_n_values: Optional[int] = None
        self._tail_dim: int = 0
        self._flatten_output = False

    def clone(self) -> "QuantumReservoirRegressor":
        return QuantumReservoirRegressor(
            lookback_window=self.lookback_window,
            n_values=self.n_values,
            hidden_modes=self.hidden_modes,
            depth=self.depth,
            alpha=self.alpha,
            state_nonlinearity=self.state_nonlinearity,
            seed=self.seed,
            strict_backend=self.strict_backend,
            ridge_alpha=self.ridge_alpha,
            include_tail_features=self.include_tail_features,
        )

    def _resolve_layout(self, X: np.ndarray) -> tuple[int, int]:
        x_dim = int(X.shape[1])
        if self.n_values is None:
            inferred_n_values = max(1, x_dim // self.lookback_window)
            tail_dim = x_dim - inferred_n_values * self.lookback_window
        else:
            inferred_n_values = int(self.n_values)
            tail_dim = x_dim - inferred_n_values * self.lookback_window
            if tail_dim < 0:
                raise ValueError(
                    f"X feature dim ({x_dim}) is too small for lookback={self.lookback_window} "
                    f"and n_values={inferred_n_values}."
                )
        return inferred_n_values, tail_dim

    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        if self._reservoir is None or self._resolved_n_values is None:
            raise ValueError("Model is not fitted.")

        X_np = np.asarray(X, dtype=np.float64)
        base_dim = self.lookback_window * self._resolved_n_values
        seq_flat = X_np[:, :base_dim]
        seqs = seq_flat.reshape(-1, self.lookback_window, self._resolved_n_values)

        reservoir_features = np.vstack(
            [self._reservoir.transform_sequence(seq) for seq in seqs]
        ).astype(np.float64)

        if self.include_tail_features and self._tail_dim > 0:
            tail = X_np[:, base_dim : base_dim + self._tail_dim]
            return np.concatenate([reservoir_features, tail], axis=1)
        return reservoir_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
            self._flatten_output = True
        elif y_np.ndim == 2:
            self._flatten_output = y_np.shape[1] == 1
        else:
            raise ValueError(f"Expected y as 1D/2D array, got shape={y_np.shape}")

        resolved_n_values, tail_dim = self._resolve_layout(X_np)
        self._resolved_n_values = resolved_n_values
        self._tail_dim = int(max(0, tail_dim))

        cfg = ReservoirConfig(
            n_values=self._resolved_n_values,
            hidden_modes=self.hidden_modes,
            lookback_window=self.lookback_window,
            depth=self.depth,
            alpha=self.alpha,
            state_nonlinearity=self.state_nonlinearity,
            seed=self.seed,
            strict_backend=self.strict_backend,
        )
        self._reservoir = QuantumReservoir(cfg)

        Z = self._prepare_features(X_np)
        self._readout.fit(Z, y_np)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self._prepare_features(np.asarray(X, dtype=np.float64))
        pred = np.asarray(self._readout.predict(Z))
        if self._flatten_output:
            return pred.reshape(-1)
        return pred
