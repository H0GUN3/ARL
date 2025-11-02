"""
시나리오 생성 유틸리티 (Detailed Experimental Specification).

각 시나리오는 RPS, p99 latency, error rate, cpu percent 등을 포함한
DataFrame을 반환한다. 생성 시 정량적 검증을 수행한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _build_frame(timestamps: np.ndarray, rps: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"timestamp": timestamps, "rps": rps})
    df["p99_latency"] = np.clip(120 + df["rps"] * 0.08, 0, None)
    df["error_rate"] = np.clip(0.01 + (df["rps"] / max(df["rps"].max(), 1)) * 0.05, 0, 1)
    df["cpu_percent"] = np.clip(df["rps"] / 5000 * 100, 0, 100)
    return df


def create_normal_scenario(
    mean_rps: float = 1000,
    duration: int = 1800,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = np.arange(duration, dtype=np.int64)
    daily_factor = 1.0 + 0.5 * np.sin(2 * np.pi * timestamps / 86400)
    noise = rng.normal(0, mean_rps * 0.05, size=duration)
    rps = np.clip(mean_rps * daily_factor + noise, 0, None)

    cv = np.std(rps) / np.mean(rps) if np.mean(rps) else 0
    if cv >= 0.15:
        raise ValueError(f"Normal scenario variance too high (CV={cv:.3f})")
    if not (400 <= rps.min() and rps.max() <= 1800):
        raise ValueError("Normal scenario RPS out of bounds")

    df = _build_frame(timestamps, rps)
    df["scenario"] = "normal"
    df["scenario_label"] = "normal"
    df["is_spike"] = False
    return df


def create_spike_scenario(
    base_rps: float = 500,
    spike_multiplier: float = 8,
    spike_start: int = 60,
    seed: Optional[int] = 42,
    duration_scale: float = 1.0,
    noise_scale: float = 1.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    phases = {
        "baseline": int(60 * duration_scale),
        "spike_rise": max(1, int(5 * duration_scale)),
        "peak": int(60 * duration_scale),
        "recovery": int(30 * duration_scale),
        "post": int(30 * duration_scale),
    }
    total_duration = sum(phases.values())
    timestamps = np.arange(total_duration, dtype=np.int64)
    rps = np.zeros(total_duration, dtype=float)

    idx = 0
    # Phase 1: Baseline
    baseline_noise = rng.normal(0, base_rps * 0.05 * noise_scale, size=phases["baseline"])
    rps[idx : idx + phases["baseline"]] = base_rps + baseline_noise
    idx += phases["baseline"]

    # Phase 2: Sudden spike (linear ramp over 5s)
    spike_rps = base_rps * spike_multiplier
    rise = np.linspace(base_rps, spike_rps, phases["spike_rise"], endpoint=False)
    rise += rng.normal(0, spike_rps * 0.1 * noise_scale, size=phases["spike_rise"])
    rps[idx : idx + phases["spike_rise"]] = rise
    idx += phases["spike_rise"]

    # Phase 3: Peak hold
    peak = spike_rps + rng.normal(0, spike_rps * 0.1 * noise_scale, size=phases["peak"])
    rps[idx : idx + phases["peak"]] = peak
    idx += phases["peak"]

    # Phase 4: Gradual recovery
    recovery = np.linspace(spike_rps, base_rps, phases["recovery"], endpoint=False)
    recovery += rng.normal(0, spike_rps * 0.1 * noise_scale, size=phases["recovery"])
    rps[idx : idx + phases["recovery"]] = recovery
    idx += phases["recovery"]

    # Phase 5: Post baseline
    post = base_rps + rng.normal(0, base_rps * 0.05 * noise_scale, size=phases["post"])
    rps[idx : idx + phases["post"]] = post

    df = _build_frame(timestamps, np.clip(rps, 0, None))
    baseline_mean = df.loc[: phases["baseline"] - 1, "rps"].mean()
    peak_value = df["rps"].max()
    actual_multiplier = peak_value / max(baseline_mean, 1)
    if actual_multiplier < spike_multiplier * 0.9:
        raise ValueError("Spike multiplier too weak")

    df["scenario"] = "spike"
    df["scenario_label"] = "spike"
    df["is_spike"] = True
    df["spike_start"] = spike_start
    return df


def create_gradual_scenario(
    base_rps: float = 500,
    target_rps: float = 3500,
    ramp_duration: int = 90,
    seed: Optional[int] = 42,
    target_multiplier: Optional[float] = None,
    ramp_scale: float = 1.0,
    noise_scale: float = 1.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if target_multiplier is not None:
        target_rps = base_rps * target_multiplier
    ramp_duration_scaled = max(1, int(ramp_duration * ramp_scale))
    plateau_duration = max(1, int(30 * ramp_scale))
    total_duration = 30 + ramp_duration_scaled + plateau_duration
    timestamps = np.arange(total_duration, dtype=np.int64)
    rps = np.zeros(total_duration, dtype=float)

    # Baseline
    rps[:30] = base_rps + rng.normal(0, 20 * noise_scale, size=30)

    # Ramp
    for t in range(30, 30 + ramp_duration_scaled):
        progress = (t - 30) / ramp_duration_scaled
        current = base_rps + (target_rps - base_rps) * progress
        rps[t] = current + rng.normal(0, 30 * noise_scale)

    # High plateau
    rps[30 + ramp_duration_scaled :] = target_rps + rng.normal(0, 50 * noise_scale, size=plateau_duration)

    df = _build_frame(timestamps, np.clip(rps, 0, None))
    ramp_section = df.iloc[30 : 30 + ramp_duration_scaled]
    x = np.arange(len(ramp_section))
    y = ramp_section["rps"].values
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot else 1.0
    slope_per_sec = slope

    if r_squared <= 0.90:
        raise ValueError(f"Gradual scenario linearity failure (R²={r_squared:.3f})")
    if not (20 <= slope_per_sec <= 40):
        raise ValueError(f"Gradual slope out of range: {slope_per_sec:.2f}")

    phases = np.full(total_duration, "baseline", dtype=object)
    phases[30 : 30 + ramp_duration_scaled] = "ramp"
    phases[30 + ramp_duration_scaled :] = "peak"
    df["phase"] = phases
    df["is_spike"] = False
    df["scenario"] = "gradual"
    df["scenario_label"] = "gradual"
    df["is_transition"] = False
    df.loc[30, "is_transition"] = True
    df.loc[30 + ramp_duration, "is_transition"] = True
    return df


def create_periodic_scenario(
    low_rps: float = 800,
    high_rps: float = 2400,
    period: int = 600,
    n_cycles: int = 3,
    seed: Optional[int] = 42,
    ratio: Optional[float] = None,
    noise_scale: float = 1.0,
) -> pd.DataFrame:
    if period <= 0 or n_cycles <= 0:
        raise ValueError("period와 n_cycles는 양수여야 합니다.")

    rng = np.random.default_rng(seed)
    total_duration = period * n_cycles
    timestamps = np.arange(total_duration, dtype=np.int64)
    rps = np.zeros(total_duration, dtype=float)

    half_period = period // 2
    for cycle in range(n_cycles):
        base = cycle * period
        low_segment = low_rps + rng.normal(0, 50 * noise_scale, size=half_period)
        high_segment = high_rps + rng.normal(0, 100 * noise_scale, size=period - half_period)
        rps[base : base + half_period] = low_segment
        rps[base + half_period : base + period] = high_segment

    low_mask = np.zeros_like(rps, dtype=bool)
    for cycle in range(n_cycles):
        base = cycle * period
        low_mask[base : base + half_period] = True
    low_mean = max(rps[low_mask].mean(), 1)
    high_mean = max(rps[~low_mask].mean(), 1)
    ratio_value = high_mean / low_mean
    desired_ratio = ratio if ratio is not None else 3.0
    if ratio_value < desired_ratio:
        scale = (desired_ratio * low_mean) / high_mean
        rps[~low_mask] *= scale
        high_mean *= scale
        ratio_value = high_mean / low_mean
    elif ratio_value > desired_ratio + 1.0:
        scale = ((desired_ratio + 1.0) * low_mean) / high_mean
        rps[~low_mask] *= scale
        high_mean *= scale
        ratio_value = high_mean / low_mean
    if ratio is not None and abs(ratio_value - ratio) > 0.5:
        raise ValueError("Periodic peak/valley ratio out of range")

    df = _build_frame(timestamps, np.clip(rps, 0, None))
    phases = np.array(["low"] * total_duration, dtype=object)
    for cycle in range(n_cycles):
        start = cycle * period + half_period
        end = (cycle + 1) * period
        phases[start:end] = "high"
    df["phase"] = phases
    transitions = np.zeros(total_duration, dtype=bool)
    for cycle in range(n_cycles):
        transitions[cycle * period] = True
        transition_point = cycle * period + half_period
        if transition_point < total_duration:
            transitions[transition_point] = True
    df["is_transition"] = transitions

    lag = period
    if len(df) > lag:
        autocorr = df["rps"].autocorr(lag=lag)
        if autocorr <= 0.6:
            raise ValueError(f"Periodic autocorrelation too low ({autocorr:.3f})")
    df["scenario"] = "periodic"
    df["scenario_label"] = "periodic"
    df["is_spike"] = False
    df["period"] = period
    return df
