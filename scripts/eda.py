"""
Structured exploratory data analysis for the BurstGPT sample logs.

The script consolidates the CSV files under `data/`, computes descriptive
statistics, and writes an EDA summary to `scripts/eda_report.md`.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import indent

import pandas as pd


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load every BurstGPT CSV, normalizing column names and tracking origin."""
    csv_paths = sorted(data_dir.glob("BurstGPT_*.csv"))
    if not csv_paths:
        raise FileNotFoundError("No BurstGPT_*.csv files found under data/")

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        df["source_file"] = path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    numeric_cols = ["timestamp", "request_tokens", "response_tokens", "total_tokens"]
    combined[numeric_cols] = combined[numeric_cols].apply(pd.to_numeric, errors="coerce")
    combined["tokens_gap"] = combined["total_tokens"] - (
        combined["request_tokens"] + combined["response_tokens"]
    )
    combined["response_ratio"] = combined["response_tokens"] / combined["total_tokens"]
    return combined


def format_section(title: str, body: str) -> str:
    """Render a markdown section, indenting multi-line content."""
    return f"## {title}\n\n{body.strip()}\n"


def render_table(df: pd.DataFrame, float_format: str = ".2f") -> str:
    """Convert a DataFrame into a fenced code block to avoid extra deps."""
    formatter = lambda x: format(x, float_format)  # type: ignore[arg-type]
    return "```\n" + df.to_string(float_format=formatter) + "\n```"


def analyze(df: pd.DataFrame) -> str:
    """Produce the full markdown report from the combined dataset."""
    total_rows = len(df)
    unique_models = df["model"].nunique(dropna=True)
    time_span = df["timestamp"].agg(["min", "max"])
    basic_info = (
        f"- Total rows: {total_rows:,}\n"
        f"- Unique models: {unique_models}\n"
        f"- Time span (seconds): {time_span['min']} → {time_span['max']}\n"
        f"- Source files: {', '.join(sorted(df['source_file'].unique()))}"
    )

    missing = (df.isna().mean() * 100).round(2).rename("missing_pct")
    missing_table = render_table(missing.reset_index().rename(columns={"index": "field"}))

    numeric_cols = [
        "request_tokens",
        "response_tokens",
        "total_tokens",
        "tokens_gap",
        "response_ratio",
    ]
    numeric_summary = df[numeric_cols].describe(percentiles=[0.5, 0.9, 0.99])

    ratio_outliers = (
        df.loc[
            (df["response_ratio"] > 1) | (df["response_ratio"] < 0),
            ["timestamp", "model", "response_ratio", "source_file"],
        ]
        .head(10)
    )

    by_model = (
        df.groupby("model")[["request_tokens", "response_tokens", "total_tokens"]]
        .agg(["count", "mean", "median"])
        .sort_values(("total_tokens", "count"), ascending=False)
    )

    log_types = df["log_type"].value_counts().to_frame("rows")
    correlations = df[["request_tokens", "response_tokens", "total_tokens"]].corr()

    per_second = df.groupby("timestamp").size().rename("requests_per_second")
    burstiness = per_second.describe(percentiles=[0.5, 0.9, 0.99]).to_frame().T
    busiest_windows = (
        per_second.nlargest(10)
        .reset_index()
        .rename(columns={"timestamp": "second", "requests_per_second": "request_count"})
    )

    sections = [
        format_section("Dataset Overview", basic_info),
        format_section("Missing Values (%)", missing_table),
        format_section("Log Type Distribution", render_table(log_types)),
        format_section("Token Distribution", render_table(numeric_summary)),
        format_section(
            "Response Ratio Outliers (top 10)",
            "No extreme ratios detected."
            if ratio_outliers.empty
            else render_table(ratio_outliers, float_format=".3f"),
        ),
        format_section("Token Correlations", render_table(correlations)),
        format_section("Activity per Model", render_table(by_model.round(2))),
        format_section(
            "Request Volume per Second",
            render_table(burstiness.T.rename(columns={0: "requests_per_second"})),
        ),
        format_section(
            "Request Volume Distribution (sample)",
            render_table(
                per_second.value_counts()
                .sort_index()
                .to_frame("seconds_with_count")
                .head(20)
            ),
        ),
        format_section("Highest Request Seconds (top 10)", render_table(busiest_windows)),
    ]
    return "\n".join(sections)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    df = load_data(project_root / "data")
    report = analyze(df)

    output_path = Path(__file__).with_name("eda_report.md")
    output_path.write_text(
        "# BurstGPT Dataset EDA\n\n"
        "This report summarizes the exploratory analysis executed via `eda.py`.\n\n"
        f"{report}",
        encoding="utf-8",
    )
    print(f"EDA complete. Report written to {output_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
