"""
LSTM 기반 예측 모델 (4-Feature 입력, 2-Layer LSTM).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

TORCH_DTYPE = torch.float32


FEATURE_COLUMNS = ["rps", "p99_latency", "error_rate", "cpu_percent"]


class RandomSequenceDataset(torch.utils.data.IterableDataset):
    """대규모 시계열에서 무작위 윈도우를 샘플링하는 IterableDataset."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        window_size: int,
        horizon: int,
        samples_per_epoch: int,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.features = features
        self.targets = targets
        self.window_size = window_size
        self.horizon = horizon
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self._epoch = 0
        self.max_start = len(self.features) - (self.window_size + self.horizon)
        if self.max_start < 0:
            raise ValueError("데이터 길이가 window_size + horizon보다 짧습니다.")

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        max_index = self.max_start + 1
        num_samples = max(1, self.samples_per_epoch)
        indices = rng.integers(0, max_index, size=num_samples)
        for start in indices:
            end = start + self.window_size
            horizon_end = end + self.horizon
            x = self.features[start:end]
            y = self.targets[end:horizon_end]
            yield (
                torch.tensor(x, dtype=TORCH_DTYPE),
                torch.tensor(y, dtype=TORCH_DTYPE),
            )


class StratifiedSequenceDataset(torch.utils.data.IterableDataset):
    """시나리오별 균등 샘플링을 지원하는 IterableDataset."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        labels: np.ndarray,
        window_size: int,
        horizon: int,
        samples_per_epoch: int,
        seed: int = 42,
        return_labels: bool = False,
    ) -> None:
        super().__init__()
        if len(features) != len(targets) or len(features) != len(labels):
            raise ValueError("features, targets, labels 길이가 일치해야 합니다.")
        self.features = features
        self.targets = targets
        self.labels = labels
        self.window_size = window_size
        self.horizon = horizon
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.return_labels = return_labels
        self._epoch = 0

        self.max_start = len(self.features) - (self.window_size + self.horizon)
        if self.max_start < 0:
            raise ValueError("데이터 길이가 window_size + horizon보다 짧습니다.")

        valid_range = np.arange(0, self.max_start + 1)
        label_array = labels[: self.max_start + 1]
        self.indices_by_label: Dict[str, np.ndarray] = {}
        for label in np.unique(label_array):
            mask = label_array == label
            indices = valid_range[mask]
            if len(indices) > 0:
                self.indices_by_label[str(label)] = indices
        if not self.indices_by_label:
            raise ValueError("유효한 시나리오별 인덱스를 찾을 수 없습니다.")

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        labels = list(self.indices_by_label.keys())
        num_labels = len(labels)
        total_samples = max(1, self.samples_per_epoch)
        base = total_samples // num_labels
        remainder = total_samples % num_labels
        counts = {label: base for label in labels}
        for label in labels[:remainder]:
            counts[label] += 1

        samples = []
        for label in labels:
            label_indices = self.indices_by_label[label]
            if len(label_indices) == 0:
                continue
            chosen = rng.choice(label_indices, size=counts[label], replace=len(label_indices) < counts[label])
            for start in chosen:
                end = start + self.window_size
                horizon_end = end + self.horizon
                x = self.features[start:end]
                y = self.targets[end:horizon_end]
                if self.return_labels:
                    samples.append(
                        (
                            torch.tensor(x, dtype=TORCH_DTYPE),
                            torch.tensor(y, dtype=TORCH_DTYPE),
                            label,
                        )
                    )
                else:
                    samples.append(
                        (
                            torch.tensor(x, dtype=TORCH_DTYPE),
                            torch.tensor(y, dtype=TORCH_DTYPE),
                        )
                    )
        rng.shuffle(samples)
        for sample in samples:
            yield sample


class LSTMNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_units_1: int,
        hidden_units_2: int,
        dense_units: int,
        horizon: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units_1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_units_1,
            hidden_size=hidden_units_2,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_units_2, dense_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_units, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        last = out2[:, -1, :]
        last = self.dropout(last)
        last = self.fc1(last)
        last = self.relu(last)
        return self.fc2(last)


@dataclass
class TrainingSummary:
    train_loss: List[float]
    convergence: bool
    model_path: str | None
    epochs_trained: int


class LSTMPredictor:
    def __init__(
        self,
        window_size: int = 60,
        horizon: int = 60,
        hidden_units_1: int = 64,
        hidden_units_2: int = 32,
        dense_units: int = 16,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        device: str | None = None,
    ) -> None:
        self.window_size = window_size
        self.horizon = horizon
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.feature_columns = FEATURE_COLUMNS
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = LSTMNet(
            input_size=len(self.feature_columns),
            hidden_units_1=hidden_units_1,
            hidden_units_2=hidden_units_2,
            dense_units=dense_units,
            horizon=horizon,
            dropout=dropout,
        ).to(self.device)
        self.scaler = StandardScaler()
        self._fitted = False

    def _build_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        values = data[self.feature_columns].astype(float).to_numpy()
        target = data["rps"].astype(float).to_numpy()
        total = len(data)
        if total < self.window_size + self.horizon:
            raise ValueError("데이터 길이가 window_size + horizon보다 짧습니다.")

        inputs = []
        outputs = []
        for start in range(total - self.window_size - self.horizon + 1):
            end = start + self.window_size
            horizon_end = end + self.horizon
            inputs.append(values[start:end])
            outputs.append(target[end:horizon_end])
        return np.asarray(inputs), np.asarray(outputs)

    def fit(
        self,
        train_data: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        model_path: str | Path | None = None,
        patience: int = 5,
        min_delta: float = 1e-4,
        samples_per_epoch: int = 50_000,
        stratified_sampling: bool = False,
        scenario_column: str = "scenario_label",
    ) -> TrainingSummary:
        missing_cols = [col for col in self.feature_columns if col not in train_data.columns]
        if missing_cols:
            raise ValueError(f"train_data에 다음 컬럼이 필요합니다: {missing_cols}")

        feature_matrix = train_data[self.feature_columns].astype(float).to_numpy()
        target_array = train_data["rps"].astype(float).to_numpy()
        self.scaler.fit(feature_matrix)
        features_scaled = self.scaler.transform(feature_matrix).astype(np.float32)
        targets = target_array.astype(np.float32)

        dataset: torch.utils.data.IterableDataset
        if stratified_sampling and scenario_column in train_data.columns:
            labels = train_data[scenario_column].astype(str).to_numpy()
            dataset = StratifiedSequenceDataset(
                features_scaled,
                targets,
                labels,
                window_size=self.window_size,
                horizon=self.horizon,
                samples_per_epoch=samples_per_epoch,
                seed=42,
            )
        else:
            dataset = RandomSequenceDataset(
                features_scaled,
                targets,
                window_size=self.window_size,
                horizon=self.horizon,
                samples_per_epoch=samples_per_epoch,
                seed=42,
            )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_loss: List[float] = []
        self.model.train()
        best_loss = float("inf")
        wait = 0
        epochs_completed = 0
        for epoch in range(epochs):
            dataset.set_epoch(epoch)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            batch_losses: List[float] = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.detach().cpu().item())
            epoch_loss = float(np.mean(batch_losses))
            train_loss.append(epoch_loss)
            epochs_completed = epoch + 1

            if epoch_loss + min_delta < best_loss:
                best_loss = epoch_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        convergence = False
        if len(train_loss) >= 5:
            best = min(train_loss)
            if best > 0:
                convergence = (train_loss[-1] / best) < 1.1

        saved_path: str | None = None
        if model_path:
            saved_path = str(model_path)
            self.save(saved_path)

        self._fitted = True
        return TrainingSummary(
            train_loss=train_loss,
            convergence=convergence,
            model_path=saved_path,
            epochs_trained=epochs_completed,
        )

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("모델 학습이 선행되어야 합니다.")

    def predict(self, context: pd.DataFrame) -> np.ndarray:
        self._ensure_fitted()
        missing_cols = [col for col in self.feature_columns if col not in context.columns]
        if missing_cols:
            raise ValueError(f"context에 필요한 컬럼 누락: {missing_cols}")
        if len(context) < self.window_size:
            raise ValueError("context 길이가 window_size보다 짧습니다.")

        window = context[self.feature_columns].astype(float).tail(self.window_size).to_numpy()
        window_scaled = self.scaler.transform(window).reshape(1, self.window_size, len(self.feature_columns))
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(window_scaled, dtype=TORCH_DTYPE).to(self.device)
            preds = self.model(tensor).cpu().numpy().flatten()
        return preds

    def save(self, path: str | Path) -> None:
        state = {
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "window_size": self.window_size,
            "horizon": self.horizon,
            "hidden_units_1": self.hidden_units_1,
            "hidden_units_2": self.hidden_units_2,
            "dense_units": self.dense_units,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
        }
        torch.save(state, path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.window_size = int(state["window_size"])
        self.horizon = int(state["horizon"])
        self.hidden_units_1 = int(state["hidden_units_1"])
        self.hidden_units_2 = int(state["hidden_units_2"])
        self.dense_units = int(state["dense_units"])
        self.dropout = float(state["dropout"])
        self.learning_rate = float(state["learning_rate"])

        self.model = LSTMNet(
            input_size=len(self.feature_columns),
            hidden_units_1=self.hidden_units_1,
            hidden_units_2=self.hidden_units_2,
            dense_units=self.dense_units,
            horizon=self.horizon,
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(state["model_state"])
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(state["scaler_mean"])
        self.scaler.scale_ = np.array(state["scaler_scale"])
        self.scaler.var_ = self.scaler.scale_ ** 2
        self._fitted = True
