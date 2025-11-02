import numpy as np
import pandas as pd

from src.data_pipeline import (
    PipelineConfig,
    add_multi_resolution_features,
    create_timeseries,
    load_and_validate_burstgpt,
    run_pipeline,
    split_dataset,
    validate_timeseries,
)


def write_sample_burstgpt(tmp_path):
    timestamps = np.arange(30)
    models = np.where(timestamps % 2 == 0, "ChatGPT", "GPT-4")
    request = 50 + 5 * timestamps
    response = np.maximum(20 + 3 * timestamps, 0)
    total = request + response
    log_types = np.where(timestamps % 3 == 0, "Conversation log", "API log")
    data = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Model": models,
            "Request tokens": request,
            "Response tokens": response,
            "Total tokens": total,
            "Log Type": log_types,
        }
    )
    split_index = len(data) // 2
    data1 = data.iloc[:split_index]
    data2 = data.iloc[split_index:]
    (tmp_path / "BurstGPT_1.csv").write_text(data1.to_csv(index=False))
    (tmp_path / "BurstGPT_2.csv").write_text(data2.to_csv(index=False))


def test_load_and_validate_burstgpt(tmp_path):
    write_sample_burstgpt(tmp_path)
    df = load_and_validate_burstgpt(tmp_path)
    assert len(df) == 30
    assert df["Timestamp"].is_monotonic_increasing


def test_create_timeseries_columns(tmp_path):
    write_sample_burstgpt(tmp_path)
    raw = load_and_validate_burstgpt(tmp_path)
    ts = create_timeseries(raw)
    assert {"second", "rps", "p99_latency", "error_rate", "cpu_percent"}.issubset(ts.columns)
    assert ts["rps"].max() > 0


def test_add_multi_resolution_features(tmp_path):
    write_sample_burstgpt(tmp_path)
    raw = load_and_validate_burstgpt(tmp_path)
    ts = create_timeseries(raw)
    enriched = add_multi_resolution_features(ts, raw, resolutions=(0.5, 0.1))
    expected_cols = {"rps_500ms_max", "rps_500ms_mean", "rps_100ms_max", "rps_100ms_mean"}
    assert expected_cols.issubset(enriched.columns)
    assert (enriched[list(expected_cols)].fillna(0) >= 0).all().all()


def test_validate_timeseries_with_custom_thresholds(tmp_path):
    write_sample_burstgpt(tmp_path)
    raw = load_and_validate_burstgpt(tmp_path)
    ts = create_timeseries(raw)
    validate_timeseries(ts, min_length=1, max_rps=1000, max_latency=5000)


def test_split_dataset_order(tmp_path):
    write_sample_burstgpt(tmp_path)
    raw = load_and_validate_burstgpt(tmp_path)
    ts = create_timeseries(raw)
    train, warmup, test = split_dataset(ts, train_ratio=0.4, warmup_ratio=0.3, test_ratio=0.3)
    assert train["second"].max() < warmup["second"].min() < test["second"].min()
    assert len(train) + len(warmup) + len(test) == len(ts)


def test_run_pipeline_small_dataset(tmp_path):
    write_sample_burstgpt(tmp_path)
    cfg = PipelineConfig(
        data_dir=tmp_path,
        output_dir=tmp_path,
        min_length=1,
        max_rps=1000,
        max_latency=5000,
    )
    paths = run_pipeline(cfg)
    for path in paths.values():
        assert path.exists()
    timeseries_df = pd.read_csv(paths["timeseries"])
    assert {"rps_100ms_max", "rps_10ms_max"}.issubset(timeseries_df.columns)
    context_cols = {"rps_delta_5s", "rps_std_30s", "time_of_day_sin", "time_of_day_cos"}
    assert context_cols.issubset(timeseries_df.columns)
