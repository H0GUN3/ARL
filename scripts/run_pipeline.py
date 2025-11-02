"""
데이터 파이프라인 실행 스크립트.

예시:
    python scripts/run_pipeline.py --with-tfdv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BurstGPT data pipeline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="원본 BurstGPT CSV가 저장된 디렉터리 (기본값: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="파이프라인 결과 CSV가 저장될 디렉터리 (기본값: data/)",
    )
    parser.add_argument(
        "--with-tfdv",
        action="store_true",
        help="TFDV 통계/스키마/이상치 리포트를 artifacts/tfdv/에 생성합니다.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10_000_000,
        help="validate_timeseries에 사용할 최소 길이 제약 (기본값: 10,000,000)",
    )
    parser.add_argument(
        "--no-context-features",
        action="store_true",
        help="컨텍스트 파생 피처(rps_delta_5s 등) 추가를 비활성화합니다.",
    )
    parser.add_argument(
        "--apply-scaling",
        action="store_true",
        help="Spec 스케일(기본: RPS=400~5000, P99=50+0.05*RPS)로 리스케일합니다.",
    )
    parser.add_argument(
        "--rps-base-offset",
        type=float,
        default=400.0,
        help="리스케일 시 RPS 기본 offset (기본값: 400)",
    )
    parser.add_argument(
        "--rps-scale",
        type=float,
        default=40.0,
        help="리스케일 시 곱해질 scale factor (기본값: 40)",
    )
    parser.add_argument(
        "--rps-clip",
        type=float,
        default=5_000.0,
        help="리스케일 이후 RPS 상한 (기본값: 5000)",
    )
    parser.add_argument(
        "--max-latency",
        type=float,
        default=5_000.0,
        help="리스케일 이후 P99 latency 상한 (기본값: 5000ms)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    artifacts_dir = Path("artifacts/tfdv")
    enable_tfdv = args.with_tfdv
    return PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_length=args.min_length,
        enable_tfdv=enable_tfdv,
        tfdv_stats_path=artifacts_dir / "stats.pbtxt" if enable_tfdv else None,
        tfdv_schema_path=artifacts_dir / "schema.pbtxt" if enable_tfdv else None,
        tfdv_anomalies_path=artifacts_dir / "anomalies.pbtxt" if enable_tfdv else None,
        multi_resolutions=(1.0, 0.1, 0.01),
        add_context_features=not args.no_context_features,
        apply_scaling=args.apply_scaling,
        rps_base_offset=args.rps_base_offset,
        rps_scale=args.rps_scale,
        rps_clip=args.rps_clip,
        max_latency=args.max_latency,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    paths = run_pipeline(config)
    print("✅ pipeline outputs")
    for name, path in paths.items():
        print(f"  - {name}: {path}")
    if args.with_tfdv:
        print("📊 TFDV artifacts saved under artifacts/tfdv/")


if __name__ == "__main__":
    main()
