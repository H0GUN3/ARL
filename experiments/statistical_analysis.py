"""
시뮬레이션 결과를 집계하고 통계 검정을 수행하는 스크립트.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.evaluation import compute_metrics, generate_report, run_statistical_tests


def load_metrics(results_dir: Path) -> List[Dict[str, object]]:
    """results/ 디렉터리에서 *_*.json 파일을 읽어 들인다."""
    records: List[Dict[str, object]] = []
    for json_path in sorted(results_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("path", str(json_path))
        records.append(data)
    return records


def enrich_with_details(results_dir: Path, records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """상세 CSV가 존재한다면 compute_metrics로 값 재계산."""
    enriched: List[Dict[str, object]] = []
    for record in records:
        details_path = Path(record["path"]).with_name(
            Path(record["path"]).stem + "_details.csv"
        )
        if details_path.exists():
            detailed_df = pd.read_csv(details_path)
            if "timestamp" in detailed_df.columns:
                detailed_df["timestamp"] = pd.to_datetime(
                    detailed_df["timestamp"],
                    format="%Y-%m-%dT%H:%M:%SZ",
                    errors="coerce",
                )
            metrics = compute_metrics(detailed_df, record["scenario"])
            record.update(metrics)
        enriched.append(record)
    return enriched


def summarize(records: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    required_cols = [
        "p99_latency",
        "success_rate",
        "stability_score",
        "adaptation_time",
        "predictive_mae",
        "tracking_lag_seconds",
        "pattern_recognition",
        "proactive_adjustment",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    summary = (
        df.groupby(["model", "scenario"])
        .agg(
            p99_latency=("p99_latency", "mean"),
            success_rate=("success_rate", "mean"),
            stability_score=("stability_score", "mean"),
            adaptation_time=("adaptation_time", "mean"),
            predictive_mae=("predictive_mae", "mean"),
            tracking_lag_seconds=("tracking_lag_seconds", "mean"),
            pattern_recognition=("pattern_recognition", "mean"),
            proactive_adjustment=("proactive_adjustment", "mean"),
        )
        .reset_index()
    )
    return summary


def compute_pairwise_tests(
    records: List[Dict[str, object]],
    metric: str,
    scenario: str,
    model_a: str,
    model_b: str,
) -> Dict[str, object]:
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("메트릭 레코드가 비어 있습니다.")
    subset = df[(df["scenario"] == scenario) & (df["model"].isin([model_a, model_b]))]
    if subset.empty:
        raise ValueError(f"{scenario} 시나리오에서 비교할 데이터가 없습니다.")
    if "seed" not in subset.columns:
        raise ValueError("pairwise 비교에는 'seed' 컬럼이 필요합니다.")

    pivot = subset.pivot(index="seed", columns="model", values=metric)
    if model_a not in pivot or model_b not in pivot:
        raise ValueError(f"{model_a}, {model_b} 중 하나의 데이터가 누락되었습니다.")
    pivot = pivot[[model_a, model_b]].dropna()
    arr_a = pivot[model_a].to_numpy(dtype=float)
    arr_b = pivot[model_b].to_numpy(dtype=float)
    if len(arr_a) < 2 or len(arr_a) != len(arr_b):
        raise ValueError("paired t-test를 위한 동일 길이의 샘플이 필요합니다.")

    stats = run_statistical_tests(arr_a, arr_b)
    stats.update(
        {
            "metric": metric,
            "scenario": scenario,
            "model_a": model_a,
            "model_b": model_b,
            "mean_a": float(arr_a.mean()),
            "mean_b": float(arr_b.mean()),
            "difference": float(arr_a.mean() - arr_b.mean()),
            "count": int(len(arr_a)),
        }
    )
    return stats


def save_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical analysis of simulation results.")
    parser.add_argument("--results-dir", default="results", help="시뮬레이션 결과 디렉터리")
    parser.add_argument("--report", default="results/statistical_report.md", help="리포트 저장 경로")
    parser.add_argument("--metric", default="success_rate", help="t-test에 사용할 메트릭")
    parser.add_argument("--scenario", default="normal", help="t-test 대상 시나리오")
    parser.add_argument("--model-a", default="LSTM", help="비교 모델 A")
    parser.add_argument("--model-b", default="LinUCB", help="비교 모델 B")
    parser.add_argument("--metrics", help="콤마로 구분된 비교 메트릭 목록 (기본: --metric)", default=None)
    parser.add_argument("--scenarios", help="콤마로 구분된 비교 시나리오 목록 (기본: 전체)", default=None)
    parser.add_argument(
        "--comparisons",
        help="모델 비교 목록 (예: LSTM:LinUCB,LinUCB:Static). 기본은 --model-a : --model-b",
        default=None,
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"결과 디렉터리를 찾을 수 없습니다: {results_dir}")

    metrics = load_metrics(results_dir)
    metrics = enrich_with_details(results_dir, metrics)
    summary_df = summarize(metrics)

    metrics_to_compare = (
        [m.strip() for m in args.metrics.split(",") if m.strip()]
        if args.metrics
        else [args.metric]
    )
    scenarios_to_compare = (
        [s.strip() for s in args.scenarios.split(",") if s.strip()]
        if args.scenarios
        else sorted(summary_df["scenario"].unique())
    )
    comparisons = []
    if args.comparisons:
        for item in args.comparisons.split(","):
            if ":" in item:
                a, b = item.split(":", 1)
                comparisons.append((a.strip(), b.strip()))
    if not comparisons:
        comparisons = [(args.model_a, args.model_b)]

    pairwise_results: List[Dict[str, object]] = []
    for scenario in scenarios_to_compare:
        for metric_name in metrics_to_compare:
            for model_a, model_b in comparisons:
                try:
                    stats = compute_pairwise_tests(
                        metrics,
                        metric=metric_name,
                        scenario=scenario,
                        model_a=model_a,
                        model_b=model_b,
                    )
                except ValueError:
                    continue
                pairwise_results.append(stats)

    report_body = generate_report(
        {
            "metrics": metrics,
            "tests": pairwise_results,
        }
    )

    if pairwise_results:
        lines = [report_body, "", "## Pairwise Comparisons", ""]
        header = (
            "| Scenario | Metric | Model A | Mean A | Model B | Mean B | Diff | p-value | Effect size | Significant | n |"
        )
        lines.append(header)
        lines.append(
            "|---|---|---|---|---|---|---|---|---|---|---|"
        )
        for res in pairwise_results:
            lines.append(
                f"| {res['scenario']} | {res['metric']} | {res['model_a']} | "
                f"{res.get('mean_a', float('nan')):.3f} | {res['model_b']} | "
                f"{res.get('mean_b', float('nan')):.3f} | {res.get('difference', float('nan')):.3f} | "
                f"{res.get('p_value', float('nan')):.3f} | {res.get('effect_size', float('nan')):.3f} | "
                f"{'Yes' if res.get('significant') else 'No'} | {res.get('count', 0)} |"
            )
        report = "\n".join(lines)
    else:
        report = report_body

    save_report(Path(args.report), report)
    print(f"✅ Statistical analysis complete. Report saved to {args.report}")


if __name__ == "__main__":
    main()
