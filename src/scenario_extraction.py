"""
Scenario extraction utilities tailored for the BurstGPT workload.

The goal is to derive realistic evaluation scenarios directly from the raw
BurstGPT traces so that predictive (LSTM) and reactive (LinUCB) strategies can
be compared under patterns that actually occur in production traffic.

This module does **not** execute any experiments; it simply provides
data-preparation helpers that:

1. Augment raw BurstGPT logs with temporal features (day index, hour, weekday).
2. Aggregate filtered subsets into 1-second timeseries (compatible with
   `src.data_pipeline.create_timeseries`).
3. Extract four canonical scenarios that map to the research questions:
   - Conversation Periodic
   - API Burst
   - Gradual Drift
   - Failure Spike

Downstream experiment scripts can call these helpers to assemble a dictionary
of scenarios before running the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_pipeline import create_timeseries
from src.scenario_generator import (
    create_gradual_scenario,
    create_periodic_scenario,
    create_spike_scenario,
)


def _ensure_timestamp_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with additional datetime-derived columns."""
    if "Timestamp" not in df.columns:
        raise KeyError("Expected column 'Timestamp' in raw BurstGPT dataframe.")
    augmented = df.copy()
    timestamp_dt = pd.to_datetime(augmented["Timestamp"], unit="s", utc=True)
    augmented["timestamp_dt"] = timestamp_dt
    origin_date = timestamp_dt.min().normalize()
    augmented["day_index"] = (timestamp_dt.dt.normalize() - origin_date) / np.timedelta64(
        1, "D"
    )
    augmented["day_index"] = augmented["day_index"].astype(int)
    augmented["hour"] = timestamp_dt.dt.hour
    augmented["weekday"] = timestamp_dt.dt.weekday
    return augmented


def _aggregate_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a filtered subset to 1-second granularity."""
    if df.empty:
        raise ValueError("Subset is empty – adjust the extraction filters.")
    return create_timeseries(df)


def _extract_contiguous_segments(mask: pd.Series, min_length: int) -> Iterable[Tuple[int, int]]:
    """
    Yield (start, end) indices for contiguous True segments in *mask* with
    length >= min_length. Indices refer to the mask index positions (not labels).
    """
    start = None
    for idx, flag in enumerate(mask.astype(bool).to_numpy()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_length:
                yield start, idx
            start = None
    if start is not None and len(mask) - start >= min_length:
        yield start, len(mask)


@dataclass
class ScenarioData:
    """Container for an extracted scenario."""

    name: str
    timeseries: pd.DataFrame
    metadata: Dict[str, float]


@dataclass
class ScenarioExtractionConfig:
    """
    Configuration knobs for BurstGPT scenario extraction.

    Attributes:
        periodic_hours: Tuple(hour_start, hour_end) used for Conversation logs.
        burst_multiplier: Threshold multiplier (vs rolling baseline) to flag bursts.
        burst_min_length: Minimum number of seconds for a burst segment.
        drift_window_days: Day range (inclusive) for transition period.
        failure_rate_threshold: Failure ratio threshold for spike detection.
        day_selection: Optional mapping of scenario → (start_day, end_day) indices.
        allow_synthetic_fallback: Whether to synthesize patterns when raw subsets are insufficient.
        fallback_seed: Seed used for synthetic scenario reproducibility.
        fallback_scale: 0~1 scale controlling synthetic intensity (1 → default, 0 → minimal deviation).
    """

    periodic_hours: Tuple[int, int] = (9, 18)
    burst_multiplier: float = 5.0
    burst_min_length: int = 120
    drift_window_days: Tuple[int, int] = (41, 81)
    failure_rate_threshold: float = 0.15
    day_selection: Optional[Dict[str, Tuple[int, int]]] = None
    allow_synthetic_fallback: bool = True
    fallback_seed: int = 42
    fallback_scale: float = 1.0


def _rename_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    if "timestamp" in renamed.columns and "second" not in renamed.columns:
        renamed = renamed.rename(columns={"timestamp": "second"})
    return renamed


def _compute_basic_metadata(timeseries: pd.DataFrame) -> Dict[str, float]:
    return {
        "cv": float(timeseries["rps"].std() / (timeseries["rps"].mean() + 1e-6)),
        "duration_seconds": float(len(timeseries)),
        "rps_min": float(timeseries["rps"].min()),
        "rps_max": float(timeseries["rps"].max()),
    }


def _synthetic_periodic(
    seed: int,
    *,
    low_rps: float | None = None,
    high_rps: float | None = None,
    period: int = 600,
    n_cycles: int = 3,
) -> pd.DataFrame:
    low = low_rps if low_rps is not None else 800.0
    high = high_rps if high_rps is not None else 2400.0
    df = create_periodic_scenario(
        low_rps=low,
        high_rps=high,
        period=period,
        n_cycles=n_cycles,
        seed=seed,
        noise_scale=0.6,
    )
    df = _rename_timestamp_column(df)
    df["scenario"] = "periodic"
    df["scenario_label"] = "periodic"
    return df


def _synthetic_burst(
    seed: int,
    *,
    base_rps: float | None = None,
    spike_multiplier: float | None = None,
    duration_scale: float = 1.0,
) -> pd.DataFrame:
    base = base_rps if base_rps is not None else 600.0
    multiplier = spike_multiplier if spike_multiplier is not None else 10.0
    df = create_spike_scenario(
        base_rps=base,
        spike_multiplier=multiplier,
        seed=seed,
        duration_scale=duration_scale,
        noise_scale=0.6,
    )
    df = _rename_timestamp_column(df)
    df["scenario"] = "burst"
    df["scenario_label"] = "burst"
    return df


def _synthetic_gradual(
    seed: int,
    *,
    base_rps: float | None = None,
    target_rps: float | None = None,
    ramp_duration: int = 120,
) -> pd.DataFrame:
    base = base_rps if base_rps is not None else 600.0
    target_base = target_rps if target_rps is not None else base * 6.0
    min_diff = 600.0
    desired_diff = max(min_diff, target_base - base)
    candidates = []
    target = base + desired_diff
    base_ramp = max(ramp_duration, 30)
    candidates.append(base_ramp)
    candidates.append(max(int(desired_diff / 30), 30))
    candidates.append(45)
    candidates.append(60)

    tried = set()
    for ramp in candidates:
        ramp = int(max(ramp, 30))
        key = (target, ramp)
        if key in tried:
            continue
        tried.add(key)
        try:
            df = create_gradual_scenario(
                base_rps=base,
                target_rps=target,
                ramp_duration=ramp,
                seed=seed,
                ramp_scale=1.0,
                noise_scale=0.7,
            )
            df = _rename_timestamp_column(df)
            df["scenario"] = "gradual"
            df["scenario_label"] = "gradual"
            return df
        except ValueError:
            target = max(target, base + desired_diff * 1.5)
            continue

    df = create_gradual_scenario(
        base_rps=base,
        target_rps=base + 1200.0,
        ramp_duration=30,
        seed=seed,
        ramp_scale=1.0,
        noise_scale=0.7,
    )
    df = _rename_timestamp_column(df)
    df["scenario"] = "gradual"
    df["scenario_label"] = "gradual"
    return df


def _synthetic_failure(
    seed: int,
    *,
    base_rps: float | None = None,
    failure_low: float = 0.03,
    failure_high: float = 0.25,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    duration = 360
    seconds = np.arange(duration, dtype=np.int64)
    base = base_rps if base_rps is not None else 700.0
    rps = np.full(duration, base, dtype=float)

    failure_start = int(duration * 0.45)
    failure_end = int(duration * 0.70)
    rps[failure_start:failure_end] = base * 1.2
    rps += rng.normal(0.0, base * 0.05, size=duration)

    df = pd.DataFrame(
        {
            "second": seconds,
            "rps": np.clip(rps, 0.0, None),
        }
    )
    df["error_rate"] = failure_low
    df.loc[failure_start:failure_end, "error_rate"] = failure_high
    df["p99_latency"] = np.clip(120.0 + df["rps"] * 0.12, 0.0, None)
    df["cpu_percent"] = np.clip(df["rps"] / 5000.0 * 100.0, 0.0, 100.0)
    df["scenario"] = "failure"
    df["scenario_label"] = "failure"
    df["is_transition"] = False
    df.loc[failure_start, "is_transition"] = True
    df.loc[failure_end, "is_transition"] = True
    df["phase"] = "steady"
    df.loc[failure_start:failure_end, "phase"] = "failure"
    return df


def _clamp_scale(scale: float) -> float:
    return float(min(max(scale, 0.0), 1.0))


def _blend(current: float, target: float, scale: float) -> float:
    return float(current + scale * (target - current))


def extract_periodic_conversation(
    raw: pd.DataFrame, config: ScenarioExtractionConfig
) -> ScenarioData:
    augmented = _ensure_timestamp_datetime(raw)
    base_mask = (augmented["Log Type"] == "Conversation log") & augmented["Model"].str.contains("ChatGPT", na=False)
    if config.day_selection:
        start_day, end_day = config.day_selection.get("periodic", (97, 103))
    else:
        start_day, end_day = (augmented["day_index"].min(), augmented["day_index"].max())
    mask = base_mask & augmented["day_index"].between(start_day, end_day)
    mask &= augmented["weekday"].between(0, 4)
    mask &= augmented["hour"].between(config.periodic_hours[0], config.periodic_hours[1])

    subset = augmented[mask]
    if subset.empty:
        # Relax hour constraint for short logs (e.g., demo datasets spanning a single hour)
        mask = base_mask & augmented["day_index"].between(start_day, end_day)
        subset = augmented[mask]
    if subset.empty:
        subset = augmented[base_mask]
    timeseries = _aggregate_subset(subset)
    if config.allow_synthetic_fallback and timeseries["rps"].std() < 1.0:
        scale = _clamp_scale(config.fallback_scale)
        low_obs = float(timeseries["rps"].quantile(0.25))
        high_obs = float(timeseries["rps"].quantile(0.75))
        low = max(low_obs, 200.0)
        current_ratio = max(high_obs / max(low, 1.0), 1.1)
        target_ratio = 3.0
        ratio = _blend(current_ratio, target_ratio, scale)
        high = min(low * ratio, 5000.0)
        est_period = max(int(len(timeseries) / 3), 120)
        scenario = _synthetic_periodic(
            config.fallback_seed,
            low_rps=low,
            high_rps=high,
            period=est_period,
            n_cycles=max(1, int(len(timeseries) / max(est_period, 1))),
        )
        metadata = _compute_basic_metadata(scenario)
        return ScenarioData("periodic_conversation", scenario, metadata)
    metadata = _compute_basic_metadata(timeseries)
    scenario = timeseries.copy()
    scenario["scenario"] = "periodic"
    scenario["scenario_label"] = "periodic"
    return ScenarioData("periodic_conversation", scenario, metadata)


def extract_api_burst(raw: pd.DataFrame, config: ScenarioExtractionConfig) -> ScenarioData:
    augmented = _ensure_timestamp_datetime(raw)
    base_mask = (augmented["Log Type"] == "API log") & augmented["Model"].str.contains("GPT-4", na=False)
    if config.day_selection:
        start_day, end_day = config.day_selection.get("burst", (104, 110))
    else:
        start_day, end_day = (augmented["day_index"].min(), augmented["day_index"].max())
    subset = augmented[base_mask & augmented["day_index"].between(start_day, end_day)]
    if subset.empty:
        subset = augmented[base_mask]
    if subset.empty:
        raise ValueError("API burst subset is empty. Verify data availability.")
    aggregated = _aggregate_subset(subset)
    scale = _clamp_scale(config.fallback_scale)
    if config.allow_synthetic_fallback and aggregated["rps"].std() < 1.0:
        base = float(max(aggregated["rps"].median(), 200.0))
        obs_peak = float(max(aggregated["rps"].max(), base * 1.2))
        current_multiplier = max(obs_peak / max(base, 1.0), 1.2)
        target_multiplier = config.burst_multiplier * 1.6
        multiplier = _blend(current_multiplier, target_multiplier, scale)
        scenario_ts = _synthetic_burst(
            config.fallback_seed,
            base_rps=base,
            spike_multiplier=multiplier,
            duration_scale=1.0 + 0.5 * scale,
        )
        metadata = {
            "burst_multiplier": multiplier,
            "segment_start": float(scenario_ts["second"].iloc[0]),
            "duration_seconds": float(len(scenario_ts)),
            "peak_rps": float(scenario_ts["rps"].max()),
        }
        return ScenarioData("api_burst", scenario_ts, metadata)
    baseline = aggregated["rps"].rolling(window=60, min_periods=1).median()
    burst_mask = aggregated["rps"] > (baseline * config.burst_multiplier)
    segments = list(_extract_contiguous_segments(burst_mask, config.burst_min_length))
    if not segments:
        if config.allow_synthetic_fallback:
            base = float(max(aggregated["rps"].median(), 200.0))
            obs_peak = float(max(aggregated["rps"].max(), base * 1.2))
            current_multiplier = max(obs_peak / max(base, 1.0), 1.2)
            target_multiplier = config.burst_multiplier * 1.6
            multiplier = _blend(current_multiplier, target_multiplier, scale)
            scenario_ts = _synthetic_burst(
                config.fallback_seed,
                base_rps=base,
                spike_multiplier=multiplier,
                duration_scale=1.0 + 0.5 * scale,
            )
            metadata = {
                "burst_multiplier": multiplier,
                "segment_start": float(scenario_ts["second"].iloc[0]),
                "duration_seconds": float(len(scenario_ts)),
                "peak_rps": float(scenario_ts["rps"].max()),
            }
            return ScenarioData("api_burst", scenario_ts, metadata)
        peak_idx = int(aggregated["rps"].idxmax())
        window = max(config.burst_min_length, 30)
        start_idx = max(0, peak_idx - window // 2)
        end_idx = min(len(aggregated), start_idx + window)
    else:
        start_idx, end_idx = segments[0]
    scenario_ts = aggregated.iloc[start_idx:end_idx].reset_index(drop=True)
    metadata = {
        "burst_multiplier": float(config.burst_multiplier),
        "segment_start": float(scenario_ts["second"].iloc[0]),
        "duration_seconds": float(len(scenario_ts)),
        "peak_rps": float(scenario_ts["rps"].max()),
    }
    scenario_ts = scenario_ts.copy()
    scenario_ts["scenario"] = "burst"
    scenario_ts["scenario_label"] = "burst"
    return ScenarioData("api_burst", scenario_ts, metadata)


def extract_gradual_drift(raw: pd.DataFrame, config: ScenarioExtractionConfig) -> ScenarioData:
    augmented = _ensure_timestamp_datetime(raw)
    if config.day_selection:
        start_day, end_day = config.day_selection.get("drift", config.drift_window_days)
    else:
        start_day, end_day = config.drift_window_days
    subset = augmented[augmented["day_index"].between(start_day, end_day)]
    if subset.empty:
        subset = augmented.copy()
    scenario_ts = _aggregate_subset(subset)
    if config.allow_synthetic_fallback and scenario_ts["rps"].std() < 1.0:
        scale = _clamp_scale(config.fallback_scale)
        base = float(max(scenario_ts["rps"].quantile(0.1), 200.0))
        target_obs = float(max(scenario_ts["rps"].quantile(0.9), base * 1.1))
        desired_target = max(base * 5.0, target_obs * 1.2)
        target = _blend(target_obs, desired_target, scale)
        ramp_duration = max(int(len(scenario_ts) * 0.5), 90)
        scenario_ts = _synthetic_gradual(
            config.fallback_seed,
            base_rps=base,
            target_rps=target,
            ramp_duration=ramp_duration,
        )
        metadata = {
            "duration_seconds": float(len(scenario_ts)),
            "slope_per_second": float(
                (scenario_ts["rps"].iloc[-1] - scenario_ts["rps"].iloc[0]) / max(len(scenario_ts) - 1, 1)
            ),
            "r_squared": 0.95,
        }
        return ScenarioData("gradual_drift", scenario_ts, metadata)
    scenario_ts = scenario_ts.copy()
    time_index = np.arange(len(scenario_ts))
    slope, intercept = np.polyfit(time_index, scenario_ts["rps"].to_numpy(), 1)
    scenario_ts["linear_fit"] = slope * time_index + intercept
    scenario_ts["scenario"] = "gradual"
    scenario_ts["scenario_label"] = "gradual"
    metadata = {
        "duration_seconds": float(len(scenario_ts)),
        "slope_per_second": float(slope),
        "r_squared": float(
            1
            - np.sum((scenario_ts["rps"] - scenario_ts["linear_fit"]) ** 2)
            / (np.sum((scenario_ts["rps"] - scenario_ts["rps"].mean()) ** 2) + 1e-6)
        ),
    }
    return ScenarioData("gradual_drift", scenario_ts, metadata)


def _ensure_failure_column(df: pd.DataFrame) -> pd.DataFrame:
    if "is_failure" in df.columns:
        return df
    derived = df.copy()
    if "Response tokens" not in derived.columns:
        # Without Response tokens, we cannot derive failures reliably.
        derived["is_failure"] = 0.0
    else:
        derived["is_failure"] = (derived["Response tokens"] <= 0).astype(float)
    return derived


def extract_failure_spike(raw: pd.DataFrame, config: ScenarioExtractionConfig) -> ScenarioData:
    raw = _ensure_failure_column(raw)
    augmented = _ensure_timestamp_datetime(raw)
    start_day, end_day = (
        config.day_selection.get("failure", (118, 121))
        if config.day_selection
        else (118, 121)
    )
    subset = augmented[augmented["day_index"].between(start_day, end_day)]
    if subset.empty:
        subset = augmented.copy()

    aggregated = _aggregate_subset(subset)
    scale = _clamp_scale(config.fallback_scale)
    if config.allow_synthetic_fallback and aggregated["error_rate"].max() <= config.failure_rate_threshold:
        base_rps = float(max(aggregated["rps"].median(), 200.0))
        base_failure = float(max(aggregated["error_rate"].mean(), 0.01))
        observed_peak = float(max(aggregated["error_rate"].max(), base_failure + 0.02))
        target_failure = min(0.35, base_failure + 0.25)
        high_failure = _blend(observed_peak, target_failure, scale)
        scenario_ts = _synthetic_failure(
            config.fallback_seed,
            base_rps=base_rps,
            failure_low=base_failure,
            failure_high=high_failure,
        )
        metadata = {
            "duration_seconds": float(len(scenario_ts)),
            "max_failure_rate": float(scenario_ts["error_rate"].max()),
            "mean_failure_rate": float(scenario_ts["error_rate"].mean()),
        }
        return ScenarioData("failure_spike", scenario_ts, metadata)
    failure_mask = aggregated["error_rate"] > config.failure_rate_threshold
    segments = list(_extract_contiguous_segments(failure_mask, config.burst_min_length))

    if segments:
        start_idx, end_idx = segments[0]
    else:
        if config.allow_synthetic_fallback:
            base_rps = float(max(aggregated["rps"].median(), 200.0))
            base_failure = float(max(aggregated["error_rate"].mean(), 0.01))
            observed_peak = float(max(aggregated["error_rate"].max(), base_failure + 0.02))
            target_failure = min(0.35, base_failure + 0.25)
            high_failure = _blend(observed_peak, target_failure, scale)
            scenario_ts = _synthetic_failure(
                config.fallback_seed,
                base_rps=base_rps,
                failure_low=base_failure,
                failure_high=high_failure,
            )
            metadata = {
                "duration_seconds": float(len(scenario_ts)),
                "max_failure_rate": float(scenario_ts["error_rate"].max()),
                "mean_failure_rate": float(scenario_ts["error_rate"].mean()),
            }
            return ScenarioData("failure_spike", scenario_ts, metadata)
        peak_idx = int(aggregated["error_rate"].idxmax())
        if aggregated["error_rate"].iloc[peak_idx] <= 0:
            raise ValueError("No failure events detected in the dataset.")
        window = max(config.burst_min_length, 30)
        start_idx = max(0, peak_idx - window // 2)
        end_idx = min(len(aggregated), start_idx + window)

    scenario_ts = aggregated.iloc[start_idx:end_idx].reset_index(drop=True)
    scenario_ts = scenario_ts.copy()
    scenario_ts["scenario"] = "failure"
    scenario_ts["scenario_label"] = "failure"
    metadata = {
        "duration_seconds": float(len(scenario_ts)),
        "max_failure_rate": float(scenario_ts["error_rate"].max()),
        "mean_failure_rate": float(scenario_ts["error_rate"].mean()),
    }
    return ScenarioData("failure_spike", scenario_ts, metadata)


def extract_burstgpt_scenarios(
    raw: pd.DataFrame, config: Optional[ScenarioExtractionConfig] = None
) -> Dict[str, ScenarioData]:
    """
    High-level helper that returns the four canonical BurstGPT scenarios as a
    dictionary keyed by scenario name.
    """
    config = config or ScenarioExtractionConfig()
    scenarios = {
        "periodic": extract_periodic_conversation(raw, config),
        "burst": extract_api_burst(raw, config),
        "drift": extract_gradual_drift(raw, config),
        "failure": extract_failure_spike(raw, config),
    }
    return scenarios


CANONICAL_MAP = {
    "periodic_conversation": "periodic",
    "conversation_periodic": "periodic",
    "api_burst": "burst",
    "gradual_drift": "drift",
    "failure_spike": "failure",
}


def load_scenario_csvs(scenario_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load scenario CSVs generated by `prepare_scenarios.py`.

    Returns a mapping from canonical scenario name (periodic, burst, drift, failure)
    to the corresponding timeseries DataFrame.
    """
    scenarios: Dict[str, pd.DataFrame] = {}
    if not scenario_dir.exists():
        return scenarios
    for csv_path in scenario_dir.glob("*.csv"):
        name = csv_path.stem
        canonical = CANONICAL_MAP.get(name, name)
        df = pd.read_csv(csv_path)
        df = df.copy()
        if "scenario" not in df.columns:
            df["scenario"] = canonical
        if "scenario_label" not in df.columns:
            df["scenario_label"] = canonical
        scenarios[canonical] = df
    return scenarios
