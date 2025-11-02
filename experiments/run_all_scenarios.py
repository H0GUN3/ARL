"""
모델별 Normal/Spike 시나리오 실험 실행 스크립트.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.baseline import StaticRateLimiter
from src.data_pipeline import (
    TEST_FILENAME,
    TIMESERIES_FILENAME,
    TRAIN_FILENAME,
    WARMUP_FILENAME,
    split_dataset,
)
from src.linucb_agent import LinUCBAgent
from src.lstm_model import FEATURE_COLUMNS, LSTMPredictor, TrainingSummary
from src.scenario_generator import (
    create_gradual_scenario,
    create_normal_scenario,
    create_periodic_scenario,
    create_spike_scenario,
)
from src.scenario_extraction import load_scenario_csvs
from src.simulator import SimulationResult, run_simulation


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=str)


def _limit_rows(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit <= 0:
        return df
    return df.head(limit)


DEFAULT_CONTEXT_KEYS = [
    "rps",
    "error_rate",
    "cpu_percent",
    "rps_delta_5s",
    "rps_std_30s",
    "time_of_day_sin",
    "time_of_day_cos",
]


def prepare_models(
    train_df: pd.DataFrame,
    warmup_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, object]:
    models: Dict[str, object] = {}

    context_keys = [key.strip() for key in args.linucb_context_keys.split(",") if key.strip()]
    if not context_keys:
        context_keys = DEFAULT_CONTEXT_KEYS

    required_train_cols = set(FEATURE_COLUMNS)
    if not required_train_cols.issubset(train_df.columns):
        missing = required_train_cols - set(train_df.columns)
        raise ValueError(f"LSTM 학습 데이터에 필요한 컬럼이 없습니다: {sorted(missing)}")
    warmup_required = set(context_keys) | {"rps", "error_rate", "cpu_percent"}
    missing_warmup_cols = warmup_required - set(warmup_df.columns)
    if missing_warmup_cols:
        raise ValueError(f"LinUCB 워밍업 데이터에 필요한 컬럼이 없습니다: {sorted(missing_warmup_cols)}")

    static_model = StaticRateLimiter.from_data(train_df)
    models["Static"] = static_model

    linucb = LinUCBAgent(
        context_keys=context_keys,
        alpha=args.linucb_alpha,
        alpha_decay=not args.disable_alpha_decay,
        decay_tau=args.linucb_decay_tau,
        min_alpha=args.linucb_min_alpha,
        max_rps=float(train_df["rps"].max()),
    )
    warmup_subset = _limit_rows(warmup_df, args.warmup_limit)
    linucb.warmup(warmup_subset)
    models["LinUCB"] = linucb

    if not args.skip_lstm:
        lstm = LSTMPredictor(
            window_size=args.window_size,
            horizon=args.horizon,
            hidden_units_1=args.hidden_units_1,
            hidden_units_2=args.hidden_units_2,
            dense_units=args.dense_units,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
        )
        train_subset = _limit_rows(train_df, args.train_limit)
        summary: TrainingSummary = lstm.fit(
            train_subset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_path=args.lstm_checkpoint if args.lstm_checkpoint else None,
            patience=args.patience,
            min_delta=args.min_delta,
            samples_per_epoch=args.samples_per_epoch,
            stratified_sampling=args.lstm_stratified,
            scenario_column=args.scenario_column,
        )
        models["LSTM"] = lstm
        if args.verbose and summary.train_loss:
            print(
                "[LSTM] 학습 완료: "
                f"loss={summary.train_loss[-1]:.4f}, convergence={summary.convergence}"
            )

    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rate limiting simulations.")
    parser.add_argument("--data", default="data/burstgpt_timeseries.csv", help="집계된 CSV 경로")
    parser.add_argument("--output-dir", default="results", help="결과 저장 디렉터리")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0], help="실험 시드 목록")
    parser.add_argument("--train-limit", type=int, default=0, help="LSTM 학습용 최대 행 수 (0=전체)")
    parser.add_argument("--warmup-limit", type=int, default=0, help="LinUCB 워밍업 최대 행 수 (0=전체)")
    parser.add_argument("--epochs", type=int, default=20, help="LSTM 학습 epoch 수")
    parser.add_argument("--batch-size", type=int, default=128, help="LSTM 배치 크기")
    parser.add_argument("--window-size", type=int, default=60, help="LSTM 입력 윈도우 크기")
    parser.add_argument("--horizon", type=int, default=60, help="LSTM 예측 horizion")
    parser.add_argument("--hidden-units-1", type=int, default=64, help="LSTM 첫 번째 LSTM 층 hidden 크기")
    parser.add_argument("--hidden-units-2", type=int, default=32, help="LSTM 두 번째 LSTM 층 hidden 크기")
    parser.add_argument("--dense-units", type=int, default=16, help="LSTM Dense 층 크기")
    parser.add_argument("--dropout", type=float, default=0.2, help="LSTM dropout")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="LSTM 학습률")
    parser.add_argument("--patience", type=int, default=8, help="LSTM 조기 종료 patience")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="LSTM 조기 종료 최소 개선폭")
    parser.add_argument("--samples-per-epoch", type=int, default=50_000, help="LSTM 학습 시 에폭당 무작위 윈도우 샘플 수")
    parser.add_argument("--lstm-checkpoint", help="LSTM 모델 저장 경로")
    parser.add_argument("--max-rows", type=int, default=0, help="시뮬레이션에 사용할 최대 행 수 (0=전체)")
    parser.add_argument("--skip-lstm", action="store_true", help="LSTM 실험 생략")
    parser.add_argument("--no-periodic", action="store_false", dest="include_periodic", help="주기적 시나리오 제외")
    parser.add_argument("--no-gradual", action="store_false", dest="include_gradual", help="점진적 시나리오 제외")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    parser.add_argument("--scenario-dir", default="data/scenarios", help="실측 시나리오 CSV 디렉터리")
    parser.add_argument("--synthetic-only", action="store_true", help="실측 시나리오를 무시하고 synthetic만 사용")
    parser.add_argument("--linucb-context-keys", default=",".join(DEFAULT_CONTEXT_KEYS), help="LinUCB 컨텍스트 컬럼 목록(콤마 구분)")
    parser.add_argument("--linucb-alpha", type=float, default=0.25, help="LinUCB 초기 alpha")
    parser.add_argument("--linucb-decay-tau", type=float, default=10_000.0, help="alpha 감쇠 τ")
    parser.add_argument("--linucb-min-alpha", type=float, default=0.05, help="alpha 감쇠 하한")
    parser.add_argument("--disable-alpha-decay", action="store_true", help="alpha 감쇠 비활성화")
    parser.add_argument("--lstm-stratified", action="store_true", help="시나리오 균등 샘플링으로 LSTM 학습")
    parser.add_argument("--scenario-column", default="scenario_label", help="Stratified 샘플링에 사용할 시나리오 컬럼명")
    parser.set_defaults(include_periodic=True, include_gradual=True)
    args = parser.parse_args()

    data_path = Path(args.data)
    if data_path.is_dir():
        base_dir = data_path
        timeseries_path = base_dir / TIMESERIES_FILENAME
    else:
        base_dir = data_path.parent
        timeseries_path = data_path

    if not timeseries_path.exists():
        raise FileNotFoundError(f"시계열 파일을 찾을 수 없습니다: {timeseries_path}")

    timeseries = pd.read_csv(timeseries_path)

    train_path = base_dir / TRAIN_FILENAME
    warmup_path = base_dir / WARMUP_FILENAME
    test_path = base_dir / TEST_FILENAME

    if train_path.exists() and warmup_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        warmup_df = pd.read_csv(warmup_path)
        test_df = pd.read_csv(test_path)
    else:
        train_df, warmup_df, test_df = split_dataset(timeseries)

    if args.max_rows and args.max_rows > 0:
        test_df = test_df.head(args.max_rows)

    feature_cols = list(FEATURE_COLUMNS)
    missing_train = set(feature_cols) - set(train_df.columns)
    missing_warmup = set(feature_cols) - set(warmup_df.columns)
    if missing_train:
        raise ValueError(f"train 데이터에 필요한 컬럼이 없습니다: {sorted(missing_train)}")
    if missing_warmup:
        raise ValueError(f"warmup 데이터에 필요한 컬럼이 없습니다: {sorted(missing_warmup)}")

    models = prepare_models(train_df, warmup_df, args)

    mean_rps = float(np.clip(test_df["rps"].mean(), 800.0, 1200.0))
    baseline_rps = float(np.clip(test_df["rps"].median(), 400.0, 600.0))
    gradual_target = float(np.clip(baseline_rps * 6.0, 3000.0, 4000.0))
    periodic_low = float(np.clip(baseline_rps * 0.8, 700.0, 900.0))
    periodic_high = float(np.clip(periodic_low * 3.0, 2200.0, 2600.0))

    scenarios: Dict[str, pd.DataFrame] = {}
    scenario_dir = Path(args.scenario_dir)
    if not args.synthetic_only:
        scenarios = load_scenario_csvs(scenario_dir)
        if args.verbose and scenarios:
            print(f"[Scenarios] Loaded real scenarios from {scenario_dir}")
    if not scenarios:
        scenarios = {
            "normal": create_normal_scenario(mean_rps=mean_rps, duration=1800),
            "spike": create_spike_scenario(base_rps=baseline_rps, spike_multiplier=8),
        }
        if args.include_gradual:
            scenarios["gradual"] = create_gradual_scenario(
                base_rps=baseline_rps,
                target_rps=gradual_target,
            )
        if args.include_periodic:
            scenarios["periodic"] = create_periodic_scenario(
                low_rps=periodic_low,
                high_rps=periodic_high,
            )
        for df in scenarios.values():
            if "scenario_label" not in df.columns and "scenario" in df.columns:
                df["scenario_label"] = df["scenario"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        for scenario_name, scenario_df in scenarios.items():
            for seed in args.seeds:
                if args.verbose:
                    print(f"[Run] model={model_name}, scenario={scenario_name}, seed={seed}")
                simulation: SimulationResult = run_simulation(
                    model,
                    scenario_df,
                    scenario=scenario_name,
                    seed=seed,
                )

                metrics = {
                    "model": model_name,
                    "scenario": scenario_name,
                    "seed": seed,
                    "p99_latency": simulation.p99_latency,
                    "success_rate": simulation.success_rate,
                    "stability_score": simulation.stability_score,
                    "adaptation_time": simulation.adaptation_time,
                    "predictive_mae": simulation.predictive_mae,
                    "tracking_lag_seconds": simulation.tracking_lag_seconds,
                    "pattern_recognition": simulation.pattern_recognition,
                    "proactive_adjustment": simulation.proactive_adjustment,
                    "confidence_interval": list(simulation.confidence_interval),
                }
                metrics_path = output_dir / f"{model_name}_{scenario_name}_{seed}.json"
                save_json(metrics_path, metrics)

                detail_path = output_dir / f"{model_name}_{scenario_name}_{seed}_details.csv"
                detailed_df = simulation.detailed_results.copy()
                if "timestamp" in detailed_df.columns and pd.api.types.is_datetime64_any_dtype(detailed_df["timestamp"]):
                    ts_series = detailed_df["timestamp"]
                    tzinfo = getattr(ts_series.dt, "tz", None)
                    if tzinfo is not None:
                        ts_series = ts_series.dt.tz_convert("UTC").dt.tz_localize(None)
                    detailed_df["timestamp"] = ts_series.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                detailed_df.to_csv(detail_path, index=False)


if __name__ == "__main__":
    main()
