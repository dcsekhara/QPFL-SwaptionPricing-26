import numpy as np


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for deep learning models. Install torch (recommended in Colab)."
        ) from exc
    return torch, nn, optim


def _infer_sequence_shape(X: np.ndarray, y: np.ndarray):
    n_outputs = y.shape[1]
    if X.shape[1] % n_outputs != 0:
        raise ValueError(
            "Cannot infer sequence shape: X feature dimension is not divisible by y feature dimension."
        )
    seq_len = X.shape[1] // n_outputs
    return seq_len, n_outputs


class TorchMLPRegressor:
    def __init__(
        self,
        hidden_dims=(512, 256),
        epochs=50,
        batch_size=64,
        lr=1e-3,
        random_state=42,
        verbose=False,
    ):
        self.hidden_dims = tuple(hidden_dims)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self._net = None
        self._device = None

    def fit(self, X, y):
        torch, nn, optim = _require_torch()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        in_dim = X.shape[1]
        out_dim = y.shape[1]

        layers = []
        prev = in_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self._net = nn.Sequential(*layers)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net.to(self._device)

        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X), torch.from_numpy(y)
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        loss_fn = nn.MSELoss()
        opt = optim.Adam(self._net.parameters(), lr=self.lr)

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
                print(f"[MLP] epoch={epoch+1}/{self.epochs} loss={epoch_loss/len(dl):.6f}")
        return self

    def predict(self, X):
        torch, _, _ = _require_torch()
        X = np.asarray(X, dtype=np.float32)
        self._net.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).to(self._device)
            pred = self._net(xb).cpu().numpy()
        return pred


class TorchCNNRegressor:
    def __init__(
        self,
        conv_channels=64,
        kernel_size=3,
        hidden_dim=256,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        random_state=42,
        verbose=False,
    ):
        self.conv_channels = int(conv_channels)
        self.kernel_size = int(kernel_size)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self._net = None
        self._device = None
        self._seq_len = None
        self._n_features = None

    def fit(self, X, y):
        torch, nn, optim = _require_torch()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        seq_len, n_features = _infer_sequence_shape(X, y)
        self._seq_len = seq_len
        self._n_features = n_features

        out_dim = y.shape[1]
        pad = self.kernel_size // 2
        self._net = nn.Sequential(
            nn.Conv1d(n_features, self.conv_channels, kernel_size=self.kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(self.conv_channels, self.conv_channels, kernel_size=self.kernel_size, padding=pad),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.conv_channels * seq_len, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, out_dim),
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net.to(self._device)

        X_seq = X.reshape(X.shape[0], seq_len, n_features).transpose(0, 2, 1)
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_seq), torch.from_numpy(y)
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        loss_fn = nn.MSELoss()
        opt = optim.Adam(self._net.parameters(), lr=self.lr)

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
                print(f"[CNN] epoch={epoch+1}/{self.epochs} loss={epoch_loss/len(dl):.6f}")
        return self

    def predict(self, X):
        torch, _, _ = _require_torch()
        X = np.asarray(X, dtype=np.float32)
        X_seq = X.reshape(X.shape[0], self._seq_len, self._n_features).transpose(0, 2, 1)
        self._net.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X_seq).to(self._device)
            pred = self._net(xb).cpu().numpy()
        return pred


class TorchLSTMRegressor:
    def __init__(
        self,
        hidden_dim=256,
        num_layers=1,
        dropout=0.0,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        random_state=42,
        verbose=False,
    ):
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self._net = None
        self._device = None
        self._seq_len = None
        self._n_features = None

    def fit(self, X, y):
        torch, nn, optim = _require_torch()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        seq_len, n_features = _infer_sequence_shape(X, y)
        self._seq_len = seq_len
        self._n_features = n_features
        out_dim = y.shape[1]

        class _LSTMReg(nn.Module):
            def __init__(self, in_dim, hidden_dim, num_layers, dropout, out_dim):
                super().__init__()
                effective_dropout = dropout if num_layers > 1 else 0.0
                self.lstm = nn.LSTM(
                    input_size=in_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=effective_dropout,
                    batch_first=True,
                )
                self.head = nn.Linear(hidden_dim, out_dim)

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.head(last)

        self._net = _LSTMReg(
            in_dim=n_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            out_dim=out_dim,
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net.to(self._device)

        X_seq = X.reshape(X.shape[0], seq_len, n_features)
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_seq), torch.from_numpy(y)
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        loss_fn = nn.MSELoss()
        opt = optim.Adam(self._net.parameters(), lr=self.lr)

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
                print(f"[LSTM] epoch={epoch+1}/{self.epochs} loss={epoch_loss/len(dl):.6f}")
        return self

    def predict(self, X):
        torch, _, _ = _require_torch()
        X = np.asarray(X, dtype=np.float32)
        X_seq = X.reshape(X.shape[0], self._seq_len, self._n_features)
        self._net.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X_seq).to(self._device)
            pred = self._net(xb).cpu().numpy()
        return pred

