import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.statistical_analysis import (
    compute_pairwise_tests,
    enrich_with_details,
    load_metrics,
    summarize,
)


def create_sample_results(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    timestamps = pd.date_range("2023-01-01", periods=60, freq="S", tz="UTC")

    for seed, offset in enumerate([0, 1]):
        for model, success in [("LSTM", 0.95 - 0.01 * offset), ("LinUCB", 0.9 - 0.015 * offset)]:
            if model == "LSTM":
                pattern = np.array([1, 1, 1, 0, 1, 1, 1, 1])
            else:
                pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0])
            accepted = np.tile(pattern, len(timestamps) // len(pattern) + 1)
            detail_df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "p99_latency": np.linspace(200 + offset, 220 + offset, len(timestamps)),
                    "accepted": accepted[: len(timestamps)],
                    "is_spike": False,
                }
            )
            json_path = results_dir / f"{model}_normal_{seed}.json"
            json_path.write_text(
                json.dumps(
                    {
                        "model": model,
                        "scenario": "normal",
                        "seed": seed,
                        "success_rate": success,
                        "p99_latency": float(detail_df["p99_latency"].mean()),
                        "stability_score": float(detail_df["accepted"].rolling(window=10, min_periods=1).mean().ge(0.95).mean()),
                        "adaptation_time": None,
                    }
                ),
                encoding="utf-8",
            )
            detail_df.to_csv(json_path.with_name(json_path.stem + "_details.csv"), index=False)

    return results_dir


def test_statistics_pipeline(tmp_path):
    results_dir = create_sample_results(tmp_path)

    metrics = load_metrics(results_dir)
    metrics = enrich_with_details(results_dir, metrics)
    summary = summarize(metrics)

    assert not summary.empty
    assert set(summary["model"]) == {"LSTM", "LinUCB"}

    stats = compute_pairwise_tests(metrics, metric="success_rate", scenario="normal", model_a="LSTM", model_b="LinUCB")
    assert "p_value" in stats
    assert stats["metric"] == "success_rate"
    assert "mean_a" in stats and "mean_b" in stats
    assert stats["count"] >= 2
