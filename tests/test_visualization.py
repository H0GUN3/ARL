import json
from pathlib import Path

import pandas as pd

from experiments.visualization import load_metric_files, plot_p99_boxplot, plot_success_rate_bar


def create_sample_results(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    data = [
        {
            "model": "LSTM",
            "scenario": "normal",
            "p99_latency": 210.0,
            "success_rate": 0.92,
            "stability_score": 0.75,
        },
        {
            "model": "LSTM",
            "scenario": "spike",
            "p99_latency": 250.0,
            "success_rate": 0.85,
            "stability_score": 0.55,
        },
        {
            "model": "LinUCB",
            "scenario": "normal",
            "p99_latency": 220.0,
            "success_rate": 0.9,
            "stability_score": 0.7,
        },
        {
            "model": "LinUCB",
            "scenario": "spike",
            "p99_latency": 240.0,
            "success_rate": 0.88,
            "stability_score": 0.6,
        },
    ]

    for idx, record in enumerate(data):
        path = results_dir / f"sample_{idx}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
    return results_dir


def test_plot_functions(tmp_path: Path):
    results_dir = create_sample_results(tmp_path)
    df = load_metric_files(results_dir.glob("*.json"))
    assert not df.empty

    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()

    p99_path = plot_p99_boxplot(df, plots_dir / "p99")
    success_path = plot_success_rate_bar(df, plots_dir / "success")

    assert p99_path.exists()
    assert success_path.exists()
