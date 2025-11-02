import numpy as np
import pandas as pd
import pytest

from src.evaluation import compute_metrics, generate_report, run_statistical_tests


def sample_detailed(scenario="normal"):
    timestamps = pd.date_range("2023-01-01", periods=120, freq="S", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "p99_latency": np.linspace(200, 250, len(timestamps)),
            "accepted": np.tile([1, 1, 0, 1], len(timestamps) // 4),
            "is_spike": False,
        }
    )
    if scenario == "spike":
        df.loc[30:60, "is_spike"] = True
    return df


def test_compute_metrics_normal():
    df = sample_detailed("normal")
    metrics = compute_metrics(df, "normal")
    assert 0 <= metrics["success_rate"] <= 1
    assert isinstance(metrics["confidence_interval"], tuple)


def test_compute_metrics_spike_adaptation():
    df = sample_detailed("spike")
    metrics = compute_metrics(df, "spike")
    assert "adaptation_time" in metrics


def test_compute_metrics_gradual_predictive_mae():
    timestamps = pd.date_range("2023-01-01", periods=6, freq="S", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "rps": [500, 620, 780, 950, 1200, 1500],
            "predicted_rps": [580, 700, 890, 1020, np.nan, np.nan],
            "accepted": [1, 1, 1, 1, 1, 1],
            "error_rate": np.linspace(0.01, 0.04, 6),
            "is_spike": False,
        }
    )
    metrics = compute_metrics(df, "gradual")
    assert metrics["predictive_mae"] is not None


def test_compute_metrics_periodic_pattern():
    timestamps = pd.date_range("2023-01-01", periods=20, freq="S", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "rps": [800] * 10 + [2400] * 10,
            "accepted": [1] * 20,
            "error_rate": [0.02] * 10 + [0.04] * 10,
            "is_spike": False,
            "is_transition": [True] + [False] * 9 + [True] + [False] * 9,
            "period": 20,
            "model": ["LSTMPredictor"] * 20,
        }
    )
    metrics = compute_metrics(df, "periodic")
    assert metrics["pattern_recognition"] is not None


def test_run_statistical_tests():
    model1 = [0.9, 0.92, 0.93, 0.95]
    model2 = [0.85, 0.87, 0.86, 0.9]
    stats = run_statistical_tests(model1, model2)
    assert "p_value" in stats
    assert "effect_size" in stats


def test_run_statistical_tests_requires_matching_lengths():
    with pytest.raises(ValueError):
        run_statistical_tests([0.9, 0.8], [0.9])


def test_generate_report():
    markdown = generate_report(
        {
            "metrics": [
                {"model": "LSTM", "scenario": "normal", "success_rate": 0.9, "p99_latency": 220.0, "stability_score": 0.95, "adaptation_time": None},
                {"model": "LinUCB", "scenario": "normal", "success_rate": 0.88, "p99_latency": 230.0, "stability_score": 0.93, "adaptation_time": None},
            ],
            "tests": [
                {"metric": "success_rate", "scenario": "normal", "t_statistic": 2.1, "p_value": 0.04, "effect_size": 0.7, "significant": True}
            ],
        }
    )
    assert "# 실험 결과 요약" in markdown
    assert "통계 검정 결과" in markdown
