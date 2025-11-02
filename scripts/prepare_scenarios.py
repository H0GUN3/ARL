"""
Generate BurstGPT-based evaluation scenarios and save them as CSV files.

Example:
    python scripts/prepare_scenarios.py --data-dir data --output-dir data/scenarios
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline import load_and_validate_burstgpt
from src.scenario_extraction import (
    ScenarioExtractionConfig,
    extract_burstgpt_scenarios,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BurstGPT scenarios.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="BurstGPT CSV가 위치한 디렉터리 (기본: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scenarios"),
        help="추출된 시나리오를 저장할 경로",
    )
    parser.add_argument(
        "--burst-multiplier",
        type=float,
        default=5.0,
        help="API burst 식별에 사용할 배수 임계값 (기본: 5.0)",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.15,
        help="Failure spike 식별용 실패율 임계값 (기본: 0.15)",
    )
    parser.add_argument(
        "--burst-min-length",
        type=int,
        default=120,
        help="Burst/Failure segment 최소 길이 (초)",
    )
    parser.add_argument(
        "--disable-fallback",
        action="store_true",
        help="실측 데이터가 부족해도 synthetic fallback을 사용하지 않습니다.",
    )
    parser.add_argument(
        "--fallback-scale",
        type=float,
        default=1.0,
        help="synthetic fallback 강도 (0.0=최소, 1.0=기본, 음수/1초과 입력 시 자동 클램프)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_and_validate_burstgpt(args.data_dir)
    config = ScenarioExtractionConfig(
        burst_multiplier=args.burst_multiplier,
        failure_rate_threshold=args.failure_threshold,
        burst_min_length=args.burst_min_length,
        allow_synthetic_fallback=not args.disable_fallback,
        fallback_scale=args.fallback_scale,
    )
    scenarios = extract_burstgpt_scenarios(raw, config)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, scenario in scenarios.items():
        path = output_dir / f"{scenario.name}.csv"
        scenario.timeseries.to_csv(path, index=False)
        meta_path = output_dir / f"{scenario.name}_metadata.json"
        meta_path.write_text(
            "{\n"
            + ",\n".join(f'  "{k}": {v:.6f}' for k, v in scenario.metadata.items())
            + "\n}\n",
            encoding="utf-8",
        )
        print(f"Saved {key}: {path}")


if __name__ == "__main__":
    main()
