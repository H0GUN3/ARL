"""
모델 평가/통계 분석 유틸리티.

Spec: @docs/API_DESIGN.md 섹션 2.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


MISSING_VALUE = 0.0


def _confidence_interval(success_rate: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(max(success_rate * (1 - success_rate), 0.0) / n)
    lower = max(0.0, success_rate - z * se)
    upper = min(1.0, success_rate + z * se)
    return (float(lower), float(upper))


def compute_metrics(detailed_results: pd.DataFrame, scenario: str) -> Dict[str, float | Tuple[float, float] | None]:
    """
    초별 결과 데이터프레임에서 핵심 메트릭을 계산한다.
    """
    if detailed_results.empty:
        return {
            "p99_latency": MISSING_VALUE,
            "success_rate": MISSING_VALUE,
            "stability_score": MISSING_VALUE,
            "adaptation_time": None,
            "confidence_interval": (0.0, 0.0),
            "predictive_mae": None,
            "tracking_lag_seconds": None,
            "pattern_recognition": None,
            "proactive_adjustment": None,
        }

    if "accepted" not in detailed_results.columns:
        raise ValueError("detailed_results에는 'accepted' 컬럼이 포함되어야 합니다.")

    scenario = scenario.lower()
    if scenario == "drift":
        scenario = "gradual"
    elif scenario == "burst":
        scenario = "spike"
    valid = {"normal", "spike", "gradual", "periodic", "failure"}
    if scenario not in valid:
        raise ValueError(f"scenario는 {valid} 중 하나여야 합니다.")

    accepted = detailed_results["accepted"].astype(float)
    n = len(accepted)
    success_rate = float(accepted.mean())
    ci = _confidence_interval(success_rate, n)

    p99_latency = float(detailed_results.get("p99_latency", pd.Series(dtype=float)).mean())

    rolling_success = accepted.rolling(window=60, min_periods=1).mean()
    stability_score = float((rolling_success >= 0.95).mean())

    adaptation_time = None
    if scenario == "spike":
        if "spike_start" in detailed_results.columns and detailed_results["spike_start"].notna().any():
            spike_start = int(detailed_results["spike_start"].dropna().iloc[0])
        else:
            spike_indices = detailed_results.index[detailed_results.get("is_spike", False)]
            spike_start = int(spike_indices.min()) if len(spike_indices) else 0
        for t in range(spike_start, len(detailed_results) - 10):
            window_mean = float(detailed_results.iloc[t : t + 10]["accepted"].mean())
            if window_mean >= 0.95:
                adaptation_time = float(t - spike_start)
                break

    predictive_mae = None
    if "predicted_rps" in detailed_results.columns:
        preds = detailed_results["predicted_rps"].dropna()
        if not preds.empty:
            actual = detailed_results["rps"].shift(-1).loc[preds.index]
            mask = actual.notna()
            if mask.any():
                predictive_mae = float(np.mean(np.abs(preds[mask] - actual[mask])))

    tracking_lag_seconds = None
    model_series = detailed_results["model"] if "model" in detailed_results.columns else None
    if model_series is not None:
        non_null_models = model_series.dropna()
        model_name = non_null_models.iloc[0] if not non_null_models.empty else None
        if model_name == "LinUCBAgent" and {"throttle_limit", "rps"}.issubset(detailed_results.columns):
            tracking_lag_seconds = float((detailed_results["throttle_limit"] < detailed_results["rps"]).sum())

    pattern_recognition = None
    proactive_adjustment = None
    if scenario == "periodic":
        period_series = detailed_results.get("period")
        if period_series is not None and period_series.notna().any():
            lag = int(period_series.dropna().iloc[0] // 2)
            if lag > 0 and len(detailed_results) > lag:
                pattern_recognition = float(detailed_results["rps"].autocorr(lag=lag))
        if detailed_results.get("is_transition") is not None and detailed_results["is_transition"].any():
            diffs = []
            for idx in detailed_results.index[detailed_results["is_transition"]]:
                pre = detailed_results["error_rate"].iloc[max(idx - 30, 0):idx]
                post = detailed_results["error_rate"].iloc[idx:min(idx + 31, len(detailed_results))]
                if len(pre) > 0 and len(post) > 0:
                    diffs.append(float(post.mean() - pre.mean()))
            if diffs:
                proactive_adjustment = float(np.mean(diffs))

    return {
        "p99_latency": p99_latency,
        "success_rate": success_rate,
        "stability_score": stability_score,
        "adaptation_time": adaptation_time,
        "confidence_interval": ci,
        "predictive_mae": predictive_mae,
        "tracking_lag_seconds": tracking_lag_seconds,
        "pattern_recognition": pattern_recognition,
        "proactive_adjustment": proactive_adjustment,
    }


def run_statistical_tests(model1_results: Iterable[float], model2_results: Iterable[float]) -> Dict[str, float | bool]:
    """
    두 모델 결과의 paired t-test 및 효과 크기를 계산한다.
    """
    arr1 = np.array(list(model1_results), dtype=float)
    arr2 = np.array(list(model2_results), dtype=float)
    if arr1.size != arr2.size or arr1.size < 2:
        raise ValueError("동일 길이이면서 최소 2개 이상의 관측치가 필요합니다.")

    diff = arr1 - arr2
    t_stat, p_value = stats.ttest_rel(arr1, arr2, nan_policy="raise")
    diff_std = diff.std(ddof=1)
    effect_size = float(diff.mean() / diff_std) if diff_std > 0 else 0.0
    return {
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "effect_size": effect_size,
        "t_statistic": float(t_stat),
    }


def generate_report(all_results: Dict[str, List[Dict[str, object]]]) -> str:
    """
    입력된 메트릭/통계 결과를 마크다운 텍스트로 정리한다.
    """
    metrics_records = all_results.get("metrics", [])
    stats_records = all_results.get("tests", [])

    lines = ["# 실험 결과 요약", ""]

    if metrics_records:
        df = pd.DataFrame(metrics_records)
        lines.append("## 시나리오별 성능 요약")
        lines.append("")
        for scenario, group in df.groupby("scenario"):
            lines.append(f"### {scenario.capitalize()}")
            columns = [
                "success_rate",
                "p99_latency",
                "stability_score",
                "adaptation_time",
                "predictive_mae",
                "tracking_lag_seconds",
                "pattern_recognition",
                "proactive_adjustment",
            ]
            available = [col for col in columns if col in group]
            pivot = group[["model", *available]].set_index("model")
            lines.append("```\n" + pivot.to_string(float_format=lambda x: f"{x:.3f}") + "\n```")
            lines.append("")
    else:
        lines.append("_메트릭 데이터가 없습니다._")
        lines.append("")

    if stats_records:
        lines.append("## 통계 검정 결과")
        for record in stats_records:
            metric = record.get("metric")
            scenario = record.get("scenario")
            model_a = record.get("model_a")
            model_b = record.get("model_b")
            mean_a = record.get("mean_a")
            mean_b = record.get("mean_b")
            diff = record.get("difference")
            p_value = record.get("p_value")
            effect_size = record.get("effect_size")
            significant = record.get("significant")
            count = record.get("count", "")
            mean_a_str = f"{mean_a:.3f}" if isinstance(mean_a, (int, float)) else "N/A"
            mean_b_str = f"{mean_b:.3f}" if isinstance(mean_b, (int, float)) else "N/A"
            diff_str = f"{diff:.3f}" if isinstance(diff, (int, float)) else "N/A"
            p_str = f"{p_value:.3f}" if isinstance(p_value, (int, float)) else "N/A"
            d_str = f"{effect_size:.3f}" if isinstance(effect_size, (int, float)) else "N/A"
            lines.append(
                f"- {metric} ({scenario}) {model_a} vs {model_b}: "
                f"mean={mean_a_str} vs {mean_b_str}, diff={diff_str}, "
                f"t={record.get('t_statistic', float('nan')):.3f}, p={p_str}, "
                f"d={d_str}, n={count} → "
                f"{'유의' if significant else '비유의'}"
            )
    else:
        lines.append("_통계 검정 결과가 없습니다._")

    return "\n".join(lines)
