from pathlib import Path

import pandas as pd

from src.scenario_extraction import load_scenario_csvs


def test_load_scenario_csvs(tmp_path: Path):
    df = pd.DataFrame(
        {
            "second": [0, 1, 2],
            "rps": [100, 120, 110],
            "p99_latency": [200, 210, 205],
            "error_rate": [0.01, 0.02, 0.015],
            "cpu_percent": [10, 12, 11],
        }
    )
    csv_path = tmp_path / "periodic_conversation.csv"
    df.to_csv(csv_path, index=False)

    scenarios = load_scenario_csvs(tmp_path)
    assert "periodic" in scenarios
    loaded = scenarios["periodic"]
    assert "scenario" in loaded.columns
    assert "scenario_label" in loaded.columns
    assert (loaded["scenario_label"] == "periodic").all()
