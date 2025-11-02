"""
시각화 스크립트.

`results/` 디렉터리에 저장된 시뮬레이션 요약을 기반으로 모델별 메트릭을
도식화하여 `plots/` 아래에 저장한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

SCENARIO_DISPLAY_MAP = {
    "normal": "Normal",
    "spike": "Spike",
    "burst": "Burst",
    "gradual": "Gradual",
    "drift": "Gradual",
    "periodic": "Periodic",
    "failure": "Failure",
}

REQUIRED_COLUMNS = {"model", "scenario", "p99_latency", "success_rate", "stability_score"}


def _display_scenario(name: str) -> str:
    return SCENARIO_DISPLAY_MAP.get(str(name).lower(), str(name).title())


def load_metric_files(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if path.suffix != ".json":
            continue
        df = pd.read_json(path, lines=False, typ="series").to_frame().T
        df["path"] = str(path)
        if "scenario" in df.columns:
            df["scenario"] = df["scenario"].apply(_display_scenario)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("시각화에 사용할 JSON 결과를 찾을 수 없습니다.")
    return pd.concat(frames, ignore_index=True)


def validate_metrics_df(df: pd.DataFrame) -> None:
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"시각화를 위해 필요한 컬럼 누락: {sorted(missing_cols)}")
    if df["model"].isnull().any():
        raise ValueError("모델 정보에 결측값이 있습니다.")
    if df["scenario"].isnull().any():
        raise ValueError("시나리오 정보에 결측값이 있습니다.")


def ensure_plots_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_p99_boxplot(
    df: pd.DataFrame,
    output_path: Path,
    details_dir: Path | None = None,
    title_suffix: str | None = None,
) -> Path:
    if details_dir is not None:
        detail_frames: List[pd.DataFrame] = []
        for _, row in df.iterrows():
            detail_name = Path(row["path"]).with_name(Path(row["path"]).stem + "_details.csv")
            detail_path = details_dir / detail_name.name
            if detail_path.exists():
                detailed_df = pd.read_csv(detail_path)
                detailed_df["model"] = row["model"]
                detailed_df["scenario"] = _display_scenario(row["scenario"])
                detail_frames.append(detailed_df)
        if detail_frames:
            long_df = pd.concat(detail_frames, ignore_index=True)
            plt.figure(figsize=(8, 5))
            ax = sns.boxplot(data=long_df, x="model", y="p99_latency", hue="scenario")
            title = "모델별 P99 Latency 분포"
            if title_suffix:
                title = f"[{title_suffix}] {title}"
            ax.set_title(title)
            ax.set_xlabel("Model")
            ax.set_ylabel("P99 Latency (ms)")
            plt.tight_layout()
            output_path = output_path.with_suffix(".png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            return output_path
    plt.figure(figsize=(8, 5))
    ax = sns.boxplot(data=df, x="model", y="p99_latency", hue="scenario")
    title = "모델별 P99 Latency 비교"
    if title_suffix:
        title = f"[{title_suffix}] {title}"
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("P99 Latency (ms)")
    plt.tight_layout()
    output_path = output_path.with_suffix(".png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_success_rate_bar(df: pd.DataFrame, output_path: Path, title_suffix: str | None = None) -> Path:
    grouped = (
        df.groupby(["model", "scenario"])["success_rate"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=grouped, x="model", y="success_rate", hue="scenario")
    title = "모델별 Success Rate 평균"
    if title_suffix:
        title = f"[{title_suffix}] {title}"
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    output_path = output_path.with_suffix(".png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_stability_bar(df: pd.DataFrame, output_path: Path, title_suffix: str | None = None) -> Path:
    grouped = (
        df.groupby(["model", "scenario"])["stability_score"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=grouped, x="model", y="stability_score", hue="scenario")
    title = "모델별 Stability Score 평균"
    if title_suffix:
        title = f"[{title_suffix}] {title}"
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Stability Score")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    output_path = output_path.with_suffix(".png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="시뮬레이션 결과 시각화")
    parser.add_argument("--results-dir", default="results", help="결과 JSON이 저장된 디렉터리")
    parser.add_argument("--plots-dir", default="plots", help="이미지 저장 디렉터리")
    parser.add_argument("--version-tag", help="그래프 파일명/제목에 사용할 버전 태그")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    version_tag = args.version_tag or results_dir.name
    plots_root = ensure_plots_dir(Path(args.plots_dir))
    plots_dir = ensure_plots_dir(plots_root / version_tag)

    json_files = list(results_dir.glob("*.json"))
    df = load_metric_files(json_files)
    validate_metrics_df(df)

    p99_path = plots_dir / f"comparison_p99_boxplot_{version_tag}"
    success_path = plots_dir / f"success_rate_barplot_{version_tag}"
    stability_path = plots_dir / f"stability_score_barplot_{version_tag}"

    plot_p99_boxplot(df, p99_path, details_dir=results_dir, title_suffix=version_tag)
    plot_success_rate_bar(df, success_path, title_suffix=version_tag)
    plot_stability_bar(df, stability_path, title_suffix=version_tag)
    print(f"✅ Plots saved under {plots_dir}")


if __name__ == "__main__":
    main()
