from collections import Counter

import numpy as np
import pandas as pd
import torch

from src.lstm_model import FEATURE_COLUMNS, LSTMPredictor, StratifiedSequenceDataset


def generate_series(length: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = np.linspace(0, 8 * np.pi, length)
    rps = 1000 + 100 * np.sin(x) + rng.normal(0, 30, size=length)
    p99 = np.clip(150 + rps * 0.08, 0, None)
    error = np.clip(0.02 + rng.normal(0, 0.002, size=length), 0, 0.1)
    cpu = np.clip(rps / 5000 * 100, 0, 100)
    return pd.DataFrame(
        {
            "rps": rps,
            "p99_latency": p99,
            "error_rate": error,
            "cpu_percent": cpu,
        }
    )


def test_lstm_fit_and_predict(tmp_path):
    df = generate_series(220)
    model = LSTMPredictor(window_size=30, horizon=10, hidden_units_1=32, hidden_units_2=16, dropout=0.1)
    summary = model.fit(df, epochs=2, batch_size=16, patience=2, samples_per_epoch=128)

    assert 1 <= len(summary.train_loss) <= 2
    assert isinstance(summary.convergence, bool)
    assert 1 <= summary.epochs_trained <= 2

    context = df.tail(40)
    assert set(FEATURE_COLUMNS).issubset(context.columns)
    preds = model.predict(context)
    assert preds.shape == (10,)

    save_path = tmp_path / "lstm.pt"
    model.save(save_path)

    reloaded = LSTMPredictor(window_size=30, horizon=10, hidden_units_1=32, hidden_units_2=16, dropout=0.1)
    reloaded.load(save_path)
    preds_loaded = reloaded.predict(context)
    assert preds_loaded.shape == (10,)
    assert not np.isnan(preds_loaded).any()


def test_lstm_requires_fit():
    df = generate_series(200)
    model = LSTMPredictor(window_size=20, horizon=5, hidden_units_1=16, hidden_units_2=8, dropout=0.1)
    model.fit(df, epochs=1, batch_size=8, patience=1, samples_per_epoch=64)

    with torch.no_grad():
        pred = model.predict(df.tail(25))
    assert pred.shape == (5,)


def test_stratified_sequence_dataset_balances_labels():
    features = np.random.randn(120, len(FEATURE_COLUMNS)).astype(np.float32)
    targets = np.random.randn(120).astype(np.float32)
    labels = np.array(["periodic"] * 60 + ["burst"] * 60)
    dataset = StratifiedSequenceDataset(
        features,
        targets,
        labels,
        window_size=10,
        horizon=5,
        samples_per_epoch=40,
        seed=0,
        return_labels=True,
    )
    dataset.set_epoch(0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8)
    counts: Counter[str] = Counter()
    for batch in loader:
        _, _, label_batch = batch
        for label in label_batch:
            counts[label] += 1
    assert set(counts.keys()) == {"periodic", "burst"}
    assert abs(counts["periodic"] - counts["burst"]) <= 8


def test_lstm_stratified_sampling_flag():
    df = generate_series(260)
    df["scenario_label"] = np.where(np.arange(len(df)) < 130, "periodic", "burst")
    model = LSTMPredictor(window_size=20, horizon=5, hidden_units_1=16, hidden_units_2=8, dropout=0.1)
    summary = model.fit(
        df,
        epochs=1,
        batch_size=8,
        patience=1,
        samples_per_epoch=64,
        stratified_sampling=True,
        scenario_column="scenario_label",
    )
    assert summary.epochs_trained >= 1
