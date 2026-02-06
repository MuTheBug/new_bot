"""ML prediction models: LightGBM gradient booster + Transformer temporal model.

Ensemble produces a combined signal used for entry/exit decisions.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from bot.utils.logger import get_logger

log = get_logger("ml")
warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================
# Purged K-Fold for time-series
# ======================================================================

class PurgedKFold:
    """K-Fold cross-validation with purging and embargo for time-series."""

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X: np.ndarray):
        n = len(X)
        embargo = int(n * self.embargo_pct)
        fold_size = n // self.n_splits
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)
            train_end = max(0, test_start - embargo)
            train_after_start = min(n, test_end + embargo)
            train_idx = np.concatenate([
                np.arange(0, train_end),
                np.arange(train_after_start, n),
            ]).astype(int)
            test_idx = np.arange(test_start, test_end).astype(int)
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


# ======================================================================
# LightGBM Model
# ======================================================================

class LGBMPredictor:
    """LightGBM binary classifier for direction prediction."""

    def __init__(self, params: Optional[dict] = None):
        if lgb is None:
            raise ImportError("lightgbm is required: pip install lightgbm")
        default_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        self._n_estimators = default_params.pop("n_estimators", 500)
        self._early_stopping = default_params.pop("early_stopping_rounds", 50)
        self._params = default_params
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            feature_names: Optional[List[str]] = None) -> "LGBMPredictor":
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        callbacks = [lgb.log_evaluation(period=0)]
        if self._early_stopping:
            callbacks.append(lgb.early_stopping(self._early_stopping))
        self.model = lgb.train(
            self._params,
            dtrain,
            num_boost_round=self._n_estimators,
            valid_sets=[dval],
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        imp = self.model.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)


# ======================================================================
# Transformer Model (requires PyTorch)
# ======================================================================

# nn.Module subclasses can only be defined when torch is available
_TransformerBlock = None  # type: ignore[assignment]
TransformerPredictor = None  # type: ignore[assignment]

if torch is not None:

    class _TransformerBlock(nn.Module):  # type: ignore[no-redef]
        """Single transformer encoder block."""

        def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x):
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_out)
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
            return x

    class TransformerPredictor(nn.Module):  # type: ignore[no-redef]
        """Lightweight Transformer for sequence-based price prediction."""

        def __init__(self, n_features: int, d_model: int = 64,
                     nhead: int = 4, num_layers: int = 2,
                     dropout: float = 0.1, seq_len: int = 48):
            super().__init__()
            self.seq_len = seq_len
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            self.blocks = nn.ModuleList([
                _TransformerBlock(d_model, nhead, dropout)
                for _ in range(num_layers)
            ])
            self.head = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            # x: (batch, seq_len, n_features)
            x = self.input_proj(x) + self.pos_emb
            for block in self.blocks:
                x = block(x)
            x = x[:, -1, :]
            return self.head(x).squeeze(-1)


class TransformerTrainer:
    """Training and inference wrapper for TransformerPredictor."""

    def __init__(self, n_features: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, seq_len: int = 48,
                 lr: float = 1e-3, epochs: int = 30,
                 batch_size: int = 64):
        if torch is None:
            raise ImportError("torch is required: pip install torch")
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerPredictor(
            n_features, d_model, nhead, num_layers, dropout, seq_len
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.BCELoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

    def _make_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None
                        ) -> Tuple:
        """Slide a window over the feature matrix to create sequences."""
        seqs, labels = [], []
        for i in range(self.seq_len, len(X)):
            seqs.append(X[i - self.seq_len:i])
            if y is not None:
                labels.append(y[i])
        X_seq = np.array(seqs, dtype=np.float32)
        if y is not None:
            y_seq = np.array(labels, dtype=np.float32)
            return X_seq, y_seq
        return (X_seq,)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> "TransformerTrainer":
        X_tr_seq, y_tr_seq = self._make_sequences(X_train, y_train)
        X_va_seq, y_va_seq = self._make_sequences(X_val, y_val)

        train_ds = TensorDataset(
            torch.from_numpy(X_tr_seq).to(self.device),
            torch.from_numpy(y_tr_seq).to(self.device),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        X_val_t = torch.from_numpy(X_va_seq).to(self.device)
        y_val_t = torch.from_numpy(y_va_seq).to(self.device)

        best_val_loss = float("inf")
        best_state = None
        patience, patience_counter = 10, 0

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item() * len(xb)
            self.scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = self.criterion(val_pred, y_val_t).item()
            self.model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        result = self._make_sequences(X)
        X_seq = result[0]
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_seq).to(self.device)
            preds = self.model(X_t).cpu().numpy()
        return preds


# ======================================================================
# Ensemble
# ======================================================================

class EnsemblePredictor:
    """Weighted ensemble of LightGBM + Transformer predictions."""

    def __init__(self, lgbm_weight: float = 0.6, transformer_weight: float = 0.4):
        self.lgbm_weight = lgbm_weight
        self.transformer_weight = transformer_weight
        self.lgbm: Optional[LGBMPredictor] = None
        self.transformer: Optional[TransformerTrainer] = None
        self._feature_names: List[str] = []
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._scaler_mean = np.nanmean(X, axis=0)
            self._scaler_std = np.nanstd(X, axis=0)
            self._scaler_std[self._scaler_std < 1e-8] = 1.0
        return (X - self._scaler_mean) / self._scaler_std

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: pd.DataFrame, y_val: np.ndarray,
            ml_config=None) -> "EnsemblePredictor":
        self._feature_names = list(X_train.columns)
        X_tr = X_train.values.astype(np.float32)
        X_va = X_val.values.astype(np.float32)

        # Replace inf/nan
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)

        # --- LightGBM ---
        log.info("Training LightGBM...")
        params = ml_config.lgbm_params.copy() if ml_config else {}
        self.lgbm = LGBMPredictor(params)
        self.lgbm.fit(X_tr, y_train, X_va, y_val, self._feature_names)

        # --- Transformer ---
        X_tr_scaled = self._scale(X_tr, fit=True)
        X_va_scaled = self._scale(X_va)

        if torch is not None:
            log.info("Training Transformer...")
            cfg = ml_config or type("C", (), {
                "transformer_d_model": 64, "transformer_nhead": 4,
                "transformer_num_layers": 2, "transformer_dropout": 0.1,
                "transformer_seq_len": 48, "transformer_lr": 1e-3,
                "transformer_epochs": 30, "transformer_batch_size": 64,
            })()
            self.transformer = TransformerTrainer(
                n_features=X_tr.shape[1],
                d_model=cfg.transformer_d_model,
                nhead=cfg.transformer_nhead,
                num_layers=cfg.transformer_num_layers,
                dropout=cfg.transformer_dropout,
                seq_len=cfg.transformer_seq_len,
                lr=cfg.transformer_lr,
                epochs=cfg.transformer_epochs,
                batch_size=cfg.transformer_batch_size,
            )
            self.transformer.fit(X_tr_scaled, y_train, X_va_scaled, y_val)
        else:
            log.warning("PyTorch unavailable — using LightGBM only")
            self.lgbm_weight = 1.0
            self.transformer_weight = 0.0

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Filter to the features used during training (feature selection)
        if self._feature_names and isinstance(X, pd.DataFrame):
            missing = [c for c in self._feature_names if c not in X.columns]
            if missing:
                raise ValueError(f"Missing features for prediction: {missing}")
            X = X[self._feature_names]
        X_arr = np.nan_to_num(X.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        lgbm_pred = self.lgbm.predict_proba(X_arr)

        if self.transformer is not None and self.transformer_weight > 0:
            X_scaled = self._scale(X_arr)
            tf_pred = self.transformer.predict_proba(X_scaled)
            # Transformer output is shorter by seq_len
            offset = len(lgbm_pred) - len(tf_pred)
            combined = np.full(len(lgbm_pred), np.nan)
            combined[:offset] = lgbm_pred[:offset]
            combined[offset:] = (
                self.lgbm_weight * lgbm_pred[offset:]
                + self.transformer_weight * tf_pred
            )
            return combined
        return lgbm_pred

    def feature_importance(self) -> pd.Series:
        if self.lgbm is not None:
            return self.lgbm.feature_importance()
        return pd.Series(dtype=float)
