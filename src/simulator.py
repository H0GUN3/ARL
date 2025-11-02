"""
시뮬레이터: 모델별 Rate Limiting 정책을 오프라인 데이터에 적용해 평가한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.baseline import StaticRateLimiter
from src.evaluation import compute_metrics
from src.linucb_agent import LinUCBAgent
from src.lstm_model import FEATURE_COLUMNS, LSTMPredictor


def extract_context(row: pd.Series, max_rps: float = 5000.0) -> np.ndarray:
    rps = float(row.get("rps", 0.0))
    error_rate = float(row.get("error_rate", 0.0))
    cpu_percent = float(row.get("cpu_percent", row.get("cpu", 0.0)))
    if max_rps <= 0:
        max_rps = 1.0
    context = np.array(
        [
            np.clip(rps / max_rps, 0.0, 1.5),
            np.clip(error_rate, 0.0, 1.0),
            np.clip(cpu_percent / 100.0, 0.0, 1.0),
        ],
        dtype=float,
    )
    return context


def simulate_request(throttle_limit: float, current_rps: float, rng: np.random.Generator) -> bool:
    throttle_limit = max(float(throttle_limit), 1.0)
    current_rps = max(float(current_rps), 0.0)
    if current_rps <= throttle_limit or current_rps == 0:
        return True
    probability = np.clip(throttle_limit / current_rps, 0.0, 1.0)
    return bool(rng.random() < probability)


@dataclass
class SimulationResult:
    p99_latency: float
    success_rate: float
    stability_score: float
    adaptation_time: Optional[float]
    predictive_mae: Optional[float]
    tracking_lag_seconds: Optional[float]
    pattern_recognition: Optional[float]
    proactive_adjustment: Optional[float]
    confidence_interval: Tuple[float, float]
    detailed_results: pd.DataFrame


def _determine_throttle_for_lstm(
    model: LSTMPredictor,
    history: pd.DataFrame,
) -> float:
    preds = model.predict(history[FEATURE_COLUMNS].tail(model.window_size))
    return float(max(np.max(preds) * 1.05, 1.0))


def run_simulation(
    model,
    test_data: pd.DataFrame,
    scenario: str,
    seed: int = 0,
    max_rps: float = 5000.0,
) -> SimulationResult:
    scenario = scenario.lower()
    if scenario == "drift":
        scenario = "gradual"
    elif scenario == "burst":
        scenario = "spike"
    valid_scenarios = {"normal", "spike", "gradual", "periodic", "failure"}
    if scenario not in valid_scenarios:
        raise ValueError(f"scenario는 {valid_scenarios} 중 하나여야 합니다.")
    if "rps" not in test_data.columns:
        raise ValueError("test_data에 'rps' 컬럼이 필요합니다.")

    rng = np.random.default_rng(seed)
    df = test_data.copy().reset_index(drop=True)
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range("2023-01-01", periods=len(df), freq="S", tz="UTC")
    if "scenario" not in df.columns:
        df["scenario"] = scenario
    else:
        df["scenario"] = (
            df["scenario"]
            .astype(str)
            .str.lower()
            .replace({"drift": "gradual", "burst": "spike"})
        )
    if "is_spike" not in df.columns:
        df["is_spike"] = False
    if "is_transition" not in df.columns:
        df["is_transition"] = False
    if "cpu_percent" not in df.columns:
        rps_norm = (df["rps"] - df["rps"].min()) / max(df["rps"].max() - df["rps"].min(), 1.0)
        df["cpu_percent"] = (rps_norm.clip(0.0, 1.0) * 100).ewm(alpha=0.2).mean()

    model_name = getattr(model, "__class__", type("Anon", (), {})).__name__

    detailed_records = []
    history = pd.DataFrame(columns=FEATURE_COLUMNS)
    prediction_records: list[tuple[int, float]] = []

    for idx, row in df.iterrows():
        history_row = {col: row.get(col, np.nan) for col in FEATURE_COLUMNS}
        history = pd.concat([history, pd.DataFrame([history_row])], ignore_index=True)
        throttle_limit = float(np.percentile(history["rps"], 95)) if len(history) else float(row["rps"])
        predicted_next = np.nan
        action_idx = None
        context_vec = None

        if isinstance(model, LinUCBAgent):
            context_vec = model._extract_context(row)
            action_idx = model.select_action(context_vec)
            throttle_limit = float(model.get_action_value(action_idx))
        elif isinstance(model, StaticRateLimiter):
            throttle_limit = float(model.select_action(None))
        elif isinstance(model, LSTMPredictor):
            if len(history) >= model.window_size:
                window_df = history.tail(model.window_size)
                preds = model.predict(window_df)
                predicted_next = float(preds[0])
                prediction_records.append((idx, predicted_next))
                throttle_limit = float(max(np.max(preds) * 1.1, 1.0))
        else:
            raise TypeError("지원되지 않는 모델 타입입니다.")

        throttle_limit = max(throttle_limit, 1.0)
        accepted = simulate_request(throttle_limit, row["rps"], rng)

        if isinstance(model, LinUCBAgent):
            reward = 1.0 if accepted else 0.0
            if context_vec is None:
                context_vec = model._extract_context(row)
            model.update(context_vec, action_idx, reward)

        record = {
            "timestamp": row["timestamp"],
            "rps": row["rps"],
            "error_rate": row.get("error_rate", 0.0),
            "p99_latency": row.get("p99_latency", 0.0),
            "cpu_percent": row.get("cpu_percent", row.get("cpu", 0.0)),
            "throttle_limit": throttle_limit,
            "accepted": accepted,
            "scenario": row.get("scenario", scenario),
            "is_spike": row.get("is_spike", False),
            "is_transition": row.get("is_transition", False),
            "phase": row.get("phase"),
            "period": row.get("period"),
            "predicted_rps": predicted_next,
            "model": model_name,
            "spike_start": row.get("spike_start"),
        }
        detailed_records.append(record)

    detailed_df = pd.DataFrame(detailed_records)

    tracking_lag_seconds = None
    if model_name == "LinUCBAgent":
        tracking_lag_seconds = float((detailed_df["throttle_limit"] < detailed_df["rps"]).sum())

    metrics = compute_metrics(detailed_df, scenario)
    predictive_mae = metrics.get("predictive_mae")
    if predictive_mae is None and model_name == "LSTMPredictor" and prediction_records:
        preds = []
        actuals = []
        for idx, pred in prediction_records:
            if idx + 1 < len(detailed_df):
                preds.append(pred)
                actuals.append(detailed_df.loc[idx + 1, "rps"])
        if actuals:
            predictive_mae = float(np.mean(np.abs(np.array(actuals) - np.array(preds))))

    return SimulationResult(
        p99_latency=float(metrics.get("p99_latency", 0.0)),
        success_rate=float(metrics.get("success_rate", 0.0)),
        stability_score=float(metrics.get("stability_score", 0.0)),
        adaptation_time=metrics.get("adaptation_time"),
        predictive_mae=predictive_mae,
        tracking_lag_seconds=tracking_lag_seconds,
        pattern_recognition=metrics.get("pattern_recognition"),
        proactive_adjustment=metrics.get("proactive_adjustment"),
        confidence_interval=metrics.get("confidence_interval", (0.0, 0.0)),
        detailed_results=detailed_df,
    )
