"""
BurstGPT 데이터 파이프라인 (Detailed Experimental Specification 준수).

주요 단계:
1. BurstGPT CSV 로드 및 무결성 검증
2. 1초 단위 시계열 생성 (RPS, P99, Error, CPU%)
3. 품질 검증 (범위/결측/길이)
4. 70-10-20 시간 순서 분할
5. 산출물 저장 (timeseries + train/warmup/test)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path("data")
TIMESERIES_FILENAME = "burstgpt_timeseries.csv"
TRAIN_FILENAME = "train_set.csv"
WARMUP_FILENAME = "warmup_set.csv"
TEST_FILENAME = "test_set.csv"


class DataValidationError(RuntimeError):
    """데이터 품질 검증 실패."""


def load_and_validate_burstgpt(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """
    BurstGPT CSV 파일(BurstGPT_*.csv) 로드 및 무결성 검증.

    검증 항목:
      - Timestamp 단조 증가
      - 결측값 없음
      - Response tokens >= 0
      - Total tokens == Request + Response
    """
    csv_paths = sorted(data_dir.glob("BurstGPT_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"BurstGPT CSV를 찾을 수 없습니다: {data_dir}")

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    if not raw["Timestamp"].is_monotonic_increasing:
        raw = raw.sort_values("Timestamp").reset_index(drop=True)

    if raw.isnull().any().any():
        raise DataValidationError("BurstGPT 원본에 결측값이 존재합니다.")
    if (raw["Response tokens"] < 0).any():
        raise DataValidationError("Response tokens에 음수 값이 있습니다.")
    if not np.all(raw["Total tokens"] == raw["Request tokens"] + raw["Response tokens"]):
        raise DataValidationError("Total tokens != Request + Response")

    return raw


def create_timeseries(raw: pd.DataFrame) -> pd.DataFrame:
    """
    개별 요청 로그를 1초 단위 시계열로 변환.

    결과 컬럼:
      - second (int): Unix timestamp (초)
      - rps (float)
      - p99_latency (ms)
      - error_rate (0-1)
      - cpu_percent (0-100)
    """
    df = raw.copy()
    df["second"] = df["Timestamp"].astype(int)

    # RPS
    rps = df.groupby("second").size().reset_index(name="rps")

    # P99 latency (ms) – 모델별 토큰당 latency 가중치
    latency_factor = np.where(df["Model"].str.contains("GPT-4", na=False), 40, 20)
    df["latency_ms"] = df["Response tokens"] * latency_factor
    p99 = df.groupby("second")["latency_ms"].quantile(0.99).reset_index(name="p99_latency")

    # Error rate
    df["is_error"] = (df["Response tokens"] == 0).astype(float)
    error = df.groupby("second")["is_error"].mean().reset_index(name="error_rate")

    # CPU percent (token throughput 기반)
    tokens = df.groupby("second")["Total tokens"].sum().reset_index(name="total_tokens")
    tokens["cpu_percent"] = np.clip(tokens["total_tokens"] / 5000 * 100, 0, 100)

    timeseries = rps.merge(p99, on="second", how="outer")
    timeseries = timeseries.merge(error, on="second", how="outer")
    timeseries = timeseries.merge(tokens[["second", "cpu_percent"]], on="second", how="outer")
    timeseries = timeseries.fillna(0)

    # 전체 시간 범위 보충
    start, end = int(timeseries["second"].min()), int(timeseries["second"].max())
    complete = pd.DataFrame({"second": np.arange(start, end + 1, dtype=np.int64)})
    timeseries = complete.merge(timeseries, on="second", how="left").fillna(0)

    return timeseries.sort_values("second").reset_index(drop=True)


def _normalize_resolutions(resolutions: Iterable[float]) -> Tuple[float, ...]:
    unique = sorted({float(res) for res in resolutions if res > 0.0})
    return tuple(unique)


def _format_resolution_label(resolution: float) -> str:
    if resolution >= 1:
        return f"{int(resolution)}s"
    milliseconds = int(round(resolution * 1000))
    return f"{milliseconds}ms"


def _compute_requests_per_bucket(raw: pd.DataFrame, resolution: float) -> pd.DataFrame:
    """
    지정된 해상도(bucket 단위)에 대한 요청 수를 계산하고, 각 초(second)별 통계를 생성한다.
    """
    if "Timestamp" not in raw.columns:
        raise ValueError("raw 데이터에 'Timestamp' 컬럼이 없습니다.")

    scaled = np.floor(raw["Timestamp"].astype(float) / resolution).astype(np.int64)
    bucket_df = pd.DataFrame({"bucket_index": scaled})
    bucket_df["second"] = np.floor(bucket_df["bucket_index"] * resolution).astype(np.int64)

    requests_per_bucket = (
        bucket_df.groupby(["second", "bucket_index"])
        .size()
        .reset_index(name="requests")
    )

    stats = (
        requests_per_bucket.groupby("second")["requests"]
        .agg(["max", "mean", "std", "median"])
        .reset_index()
    )
    label = _format_resolution_label(resolution)
    stats = stats.rename(
        columns={
            "max": f"rps_{label}_max",
            "mean": f"rps_{label}_mean",
            "std": f"rps_{label}_std",
            "median": f"rps_{label}_median",
        }
    )
    stats = stats.replace({np.nan: 0.0})
    return stats


def add_multi_resolution_features(
    timeseries: pd.DataFrame,
    raw: pd.DataFrame,
    resolutions: Iterable[float],
) -> pd.DataFrame:
    """
    다중 해상도(초/100ms/10ms 등) 기반 RPS 특성을 timeseries에 병합한다.

    각 해상도에 대해 초 단위로 최대/평균/표준편차/중앙값을 계산하여 폭발적 트래픽을 포착한다.
    """
    enriched = timeseries.copy()
    for resolution in _normalize_resolutions(resolutions):
        subsecond_stats = _compute_requests_per_bucket(raw, resolution)
        enriched = enriched.merge(subsecond_stats, on="second", how="left")

    # 다중 해상도 컬럼에서 NaN 발생 시 0으로 대체
    extra_cols = [col for col in enriched.columns if col.startswith("rps_") and col not in {"rps"}]
    if extra_cols:
        enriched[extra_cols] = enriched[extra_cols].fillna(0.0)
    return enriched


def add_context_features(timeseries: pd.DataFrame, max_rps: float = 5000.0) -> pd.DataFrame:
    """
    LinUCB/LSTM 컨텍스트로 활용할 파생 피처를 추가한다.

    - rps_delta_5s: 5초 전 대비 RPS 변화율
    - rps_std_30s: 최근 30초 RPS 표준편차
    - time_of_day_sin/cos: 하루 주기(86400초)를 반영한 시간 인코딩
    """
    if "second" not in timeseries.columns:
        raise ValueError("timeseries에 'second' 컬럼이 필요합니다.")

    enriched = timeseries.copy()
    enriched["rps_delta_5s"] = enriched["rps"].diff(periods=5).fillna(0.0)
    enriched["rps_std_30s"] = (
        enriched["rps"].rolling(window=30, min_periods=1).std().fillna(0.0)
    )

    seconds = enriched["second"].astype(float)
    day = 86400.0
    enriched["time_of_day_sin"] = np.sin(2 * np.pi * (seconds % day) / day)
    enriched["time_of_day_cos"] = np.cos(2 * np.pi * (seconds % day) / day)

    enriched["rps_delta_5s"] = np.clip(enriched["rps_delta_5s"] / max_rps, -1.5, 1.5)
    enriched["rps_std_30s"] = np.clip(enriched["rps_std_30s"] / max_rps, 0.0, 1.5)
    return enriched


def apply_spec_scaling(
    timeseries: pd.DataFrame,
    *,
    base_offset: float = 400.0,
    scale: float = 40.0,
    max_rps: float = 5000.0,
    max_latency: float = 5000.0,
) -> pd.DataFrame:
    """
    Spec 스케일에 맞춰 시계열을 리스케일한다.

    - RPS: base_offset + original * scale (최대 max_rps로 클리핑)
    - P99 latency: RPS 기반 선형 추정 (50 + 0.05 * RPS)
    - CPU%: RPS / max_rps * 100
    """
    scaled = timeseries.copy()
    scaled["rps"] = np.clip(base_offset + scaled["rps"] * scale, 0.0, max_rps)
    scaled["p99_latency"] = np.clip(50.0 + scaled["rps"] * 0.05, 0.0, max_latency)
    scaled["cpu_percent"] = np.clip(scaled["rps"] / max_rps * 100.0, 0.0, 100.0)
    return scaled


def validate_timeseries(
    timeseries: pd.DataFrame,
    *,
    min_length: int = 10_000_000,
    max_rps: int = 10_000,
    max_latency: int = 5_000,
) -> None:
    """
    시계열 품질 검증.

    기본 검증 (spec 기준):
      - 길이 > 10,000,000 초
      - RPS 범위 [0, max_rps)
      - P99 latency 범위 [0, max_latency)
      - Error rate ∈ [0, 1]
      - CPU percent ∈ [0, 100]
      - 결측값 없음
    테스트 환경에서는 파라미터를 조정하여 사용할 수 있다.
    """
    checks = {
        "Length": len(timeseries) > min_length,
        "RPS Range": (timeseries["rps"].min() >= 0) and (timeseries["rps"].max() < max_rps),
        "P99 Range": (timeseries["p99_latency"].min() >= 0) and (timeseries["p99_latency"].max() < max_latency),
        "Error Range": (timeseries["error_rate"] >= 0).all() and (timeseries["error_rate"] <= 1).all(),
        "CPU Range": (timeseries["cpu_percent"] >= 0).all() and (timeseries["cpu_percent"] <= 100).all(),
        "No NaN": not timeseries.isnull().any().any(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise DataValidationError(f"Timeseries 검증 실패: {', '.join(failed)}")


def split_dataset(
    timeseries: pd.DataFrame,
    train_ratio: float = 0.7,
    warmup_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    시간 순서 기반 70-10-20 분할.
    미래 정보 누수를 방지하기 위해 second 값의 순서를 유지한다.
    """
    if not np.isclose(train_ratio + warmup_ratio + test_ratio, 1.0):
        raise ValueError("train/warmup/test 비율의 합이 1이 아닙니다.")

    total_len = len(timeseries)
    train_end = int(total_len * train_ratio)
    warmup_end = int(total_len * (train_ratio + warmup_ratio))

    train = timeseries.iloc[:train_end].reset_index(drop=True)
    warmup = timeseries.iloc[train_end:warmup_end].reset_index(drop=True)
    test = timeseries.iloc[warmup_end:].reset_index(drop=True)

    if len(train) == 0 or len(warmup) == 0 or len(test) == 0:
        raise DataValidationError("분할 후 일부 구간이 비어 있습니다.")

    train_end_second = float(train["second"].iloc[-1])
    warmup_start_second = float(warmup["second"].iloc[0])
    test_start_second = float(test["second"].iloc[0])
    if not (train_end_second < warmup_start_second < test_start_second):
        raise DataValidationError("시간 순서 분할 검증 실패 (future leakage 가능성).")

    return train, warmup, test


@dataclass
class PipelineConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_DATA_DIR
    min_length: int = 10_000_000
    max_rps: int = 10_000
    max_latency: int = 5_000
    apply_scaling: bool = False
    rps_base_offset: float = 400.0
    rps_scale: float = 40.0
    rps_clip: float = 5_000.0
    multi_resolutions: Tuple[float, ...] = (0.1, 0.01)
    enable_tfdv: bool = False
    tfdv_stats_path: Optional[Path] = None
    tfdv_schema_path: Optional[Path] = None
    tfdv_anomalies_path: Optional[Path] = None
    tfdv_previous_stats_path: Optional[Path] = None
    add_context_features: bool = True


def _maybe_run_tfdv_analysis(
    config: PipelineConfig,
    dataframe: pd.DataFrame,
) -> None:
    if not config.enable_tfdv:
        return
    try:
        import tensorflow_data_validation as tfdv  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "TensorFlow Data Validation가 설치되어 있지 않습니다. "
            "TFDV 분석을 사용하려면 'pip install tensorflow-data-validation'을 실행하세요."
        ) from exc

    stats = tfdv.generate_statistics_from_dataframe(dataframe)
    schema = tfdv.infer_schema(stats)

    if config.tfdv_stats_path is not None:
        config.tfdv_stats_path.parent.mkdir(parents=True, exist_ok=True)
        tfdv.write_stats_text(stats, str(config.tfdv_stats_path))
    if config.tfdv_schema_path is not None:
        config.tfdv_schema_path.parent.mkdir(parents=True, exist_ok=True)

        tfdv.write_schema_text(schema, str(config.tfdv_schema_path))

    if config.tfdv_previous_stats_path is not None:
        previous = tfdv.load_stats_text(str(config.tfdv_previous_stats_path))
        anomalies = tfdv.validate_statistics(stats, schema, previous_statistics=previous)
        if config.tfdv_anomalies_path is not None:
            config.tfdv_anomalies_path.parent.mkdir(parents=True, exist_ok=True)
            tfdv.write_anomalies_text(anomalies, str(config.tfdv_anomalies_path))


def run_pipeline(config: PipelineConfig = PipelineConfig()) -> Dict[str, Path]:
    """
    전체 파이프라인 실행:
      1. BurstGPT 로드 + 검증
      2. 1초 시계열 생성
      3. 품질 검증
      4. 70-10-20 분할
      5. CSV 저장 (timeseries, train/warmup/test)
    """
    raw = load_and_validate_burstgpt(config.data_dir)
    timeseries = create_timeseries(raw)
    if config.multi_resolutions:
        timeseries = add_multi_resolution_features(
            timeseries,
            raw,
            config.multi_resolutions,
        )
    if config.add_context_features:
        timeseries = add_context_features(timeseries, max_rps=config.max_rps)
    if config.apply_scaling:
        timeseries = apply_spec_scaling(
            timeseries,
            base_offset=config.rps_base_offset,
            scale=config.rps_scale,
            max_rps=config.rps_clip,
            max_latency=config.max_latency,
        )
    validate_timeseries(
        timeseries,
        min_length=config.min_length,
        max_rps=config.max_rps,
        max_latency=config.max_latency,
    )
    train, warmup, test = split_dataset(timeseries)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timeseries_path = output_dir / TIMESERIES_FILENAME
    train_path = output_dir / TRAIN_FILENAME
    warmup_path = output_dir / WARMUP_FILENAME
    test_path = output_dir / TEST_FILENAME

    timeseries.to_csv(timeseries_path, index=False)
    train.to_csv(train_path, index=False)
    warmup.to_csv(warmup_path, index=False)
    test.to_csv(test_path, index=False)

    _maybe_run_tfdv_analysis(config, timeseries)

    return {
        "timeseries": timeseries_path,
        "train": train_path,
        "warmup": warmup_path,
        "test": test_path,
    }


def main() -> None:
    config = PipelineConfig()
    paths = run_pipeline(config)
    print("✅ Data pipeline complete")
    for name, path in paths.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
