from __future__ import annotations

import importlib.util
from typing import Optional

import numpy as np

from .training_pipeline import FeatureExtractorInterface, ModelInterface


def is_torch_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch

        _ = torch.tensor([0.0]).float()
        _m = torch.nn.Linear(1, 1)
        _ = torch.optim.Adam(_m.parameters(), lr=1e-3)
        return True
    except Exception:
        return False


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for TorchCNNRegressor. "
            "Install it in Colab/local env before running CNN experiments."
        ) from exc
    return torch, nn, optim


class TorchCNNRegressor(ModelInterface):
    """
    1D CNN regressor for tabular time-lag features.
    Uses MSE loss only.
    """

    supports_batch_prediction: bool = True

    def __init__(
        self,
        conv_channels: int = 64,
        kernel_size: int = 3,
        hidden_dim: int = 256,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.conv_channels = int(conv_channels)
        self.kernel_size = int(kernel_size)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        self._device = None
        self._net = None
        self._input_dim: Optional[int] = None
        self._output_dim: Optional[int] = None
        self._flatten_output = False

    def clone(self) -> "TorchCNNRegressor":
        return TorchCNNRegressor(
            conv_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            hidden_dim=self.hidden_dim,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        torch, nn, optim = _require_torch()

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"Expected X as 2D array, got shape={X_np.shape}")
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
            self._flatten_output = True
        elif y_np.ndim == 2:
            self._flatten_output = y_np.shape[1] == 1
        else:
            raise ValueError(f"Expected y as 1D/2D array, got shape={y_np.shape}")

        self._input_dim = int(X_np.shape[1])
        self._output_dim = int(y_np.shape[1])
        pad = self.kernel_size // 2

        self._net = nn.Sequential(
            nn.Conv1d(1, self.conv_channels, kernel_size=self.kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(
                self.conv_channels,
                self.conv_channels,
                kernel_size=self.kernel_size,
                padding=pad,
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.conv_channels * self._input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self._output_dim),
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net.to(self._device)

        X_seq = X_np[:, None, :]  # [batch, channels=1, seq_len=input_dim]
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_seq),
            torch.from_numpy(y_np),
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        loss_fn = nn.MSELoss()
        opt = optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self._net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in dl:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                opt.zero_grad()
                pred = self._net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(f"[CNN] epoch={epoch + 1}/{self.epochs} loss={epoch_loss / len(dl):.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch, _, _ = _require_torch()
        if self._net is None or self._input_dim is None or self._output_dim is None:
            raise ValueError("TorchCNNRegressor is not fitted.")

        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2 or X_np.shape[1] != self._input_dim:
            raise ValueError(
                f"Expected X shape [n_samples, {self._input_dim}], got {X_np.shape}"
            )

        X_seq = X_np[:, None, :]
        self._net.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X_seq).to(self._device)
            pred = self._net(xb).cpu().numpy()
        if self._flatten_output:
            return pred.reshape(-1)
        return pred

    def extract_latent(self, X: np.ndarray, layer_index: int = -2) -> np.ndarray:
        """
        Extract intermediate activations from the sequential network.
        Default layer_index=-2 is the hidden activation before the final output layer.
        """
        torch, _, _ = _require_torch()
        if self._net is None or self._input_dim is None:
            raise ValueError("TorchCNNRegressor is not fitted.")

        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2 or X_np.shape[1] != self._input_dim:
            raise ValueError(
                f"Expected X shape [n_samples, {self._input_dim}], got {X_np.shape}"
            )

        X_seq = X_np[:, None, :]
        modules = list(self._net.children())
        resolved_idx = layer_index if layer_index >= 0 else len(modules) + layer_index
        if not (0 <= resolved_idx < len(modules)):
            raise ValueError(f"Invalid layer_index={layer_index} for network depth={len(modules)}")

        self._net.eval()
        with torch.no_grad():
            x = torch.from_numpy(X_seq).to(self._device)
            for i, mod in enumerate(modules):
                x = mod(x)
                if i == resolved_idx:
                    break
            latent = x.cpu().numpy()
        return np.asarray(latent)


class FrozenCNNLatentFeatureExtractor(FeatureExtractorInterface):
    """
    Feature extractor that reuses a fitted TorchCNNRegressor and outputs
    intermediate-layer latent activations.
    """

    def __init__(self, cnn_model: TorchCNNRegressor, layer_index: int = -2) -> None:
        self.cnn_model = cnn_model
        self.layer_index = int(layer_index)

    def clone(self) -> "FrozenCNNLatentFeatureExtractor":
        return FrozenCNNLatentFeatureExtractor(
            cnn_model=self.cnn_model,
            layer_index=self.layer_index,
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        _ = X
        _ = y
        # No fitting needed; uses frozen pretrained CNN weights.

    def transform(self, X: np.ndarray) -> np.ndarray:
        latent = self.cnn_model.extract_latent(X, layer_index=self.layer_index)
        return np.asarray(latent, dtype=np.float32).reshape(latent.shape[0], -1)
