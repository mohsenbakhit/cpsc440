#!/usr/bin/env python3
"""
train_baselines_intro_only.py

Train baseline classifiers using only introduction-time features.
This script reuses the modeling/evaluation pipeline from train_baselines.py
but restricts feature columns to avoid downstream process leakage.

Usage
-----
python scripts/train_baselines_intro_only.py
python scripts/train_baselines_intro_only.py --fast-mode --parallel-evals --jobs -1

Outputs
-------
- <outdir>/baseline_metrics_intro_only.csv
- <outdir>/baseline_metrics_intro_only.json
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

import train_baselines as tb

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT_DIR / "data" / "normalized" / "bills.csv"
DEFAULT_OUTDIR = ROOT_DIR / "data" / "normalized"

# Keep only features that are available at, or very close to, introduction time.
INTRO_NUMERIC_COLS = [
    "year",
    "title_word_count",
    "description_word_count",
    "month_introduced",
    "parliament_number",
    "session_number",
    "reinstated",
]

INTRO_CATEGORICAL_COLS = [
    "bill_type",
    "bill_type_raw",
    "chamber",
    "party",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train introduction-only baseline bill-passage classifiers"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to normalized CSV",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for metrics output",
    )
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=500,
        help="Minimum training rows required to run a split",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="CPU workers to use (-1 = all available cores)",
    )
    parser.add_argument(
        "--parallel-evals",
        action="store_true",
        help="Run model evaluations in parallel within each split",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use lighter text settings for faster training",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm progress bars and use plain print updates",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Request GPU acceleration (not used by current sklearn baselines; informational flag)",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show sklearn training warnings (default hides repetitive warnings)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.show_warnings:
        warnings.filterwarnings(
            "ignore",
            message="Skipping features without any observed values",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="The max_iter was reached which means the coef_ did not converge",
        )

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Override feature lists used by train_baselines pipeline.
    tb.NUMERIC_COLS = INTRO_NUMERIC_COLS
    tb.CATEGORICAL_COLS = INTRO_CATEGORICAL_COLS

    print(f"Loading normalized data: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    df = tb.clean_dataframe(df)

    cpu_jobs = tb.resolve_jobs(args.jobs)
    use_progress_bar = (not args.no_progress_bar) and (tb.tqdm is not None)

    print("Introduction-only mode: ON")
    print(f"Numeric features: {tb.NUMERIC_COLS}")
    print(f"Categorical features: {tb.CATEGORICAL_COLS}")
    print(f"CPU workers: {cpu_jobs}")
    if args.fast_mode:
        print("Fast mode: ON (lighter TF-IDF settings)")
    if args.parallel_evals:
        print("Parallel evaluations: ON")
    if args.use_gpu:
        print("GPU flag set, but current sklearn baseline models run on CPU.")
    if tb.tqdm is None and not args.no_progress_bar:
        print("tqdm is not installed; falling back to print progress updates.")

    print(f"Rows: {len(df):,} | Sources: {df['source'].value_counts().to_dict()}")
    print("Running introduction-only baseline experiments...")
    metrics_df = tb.run_experiments(
        df,
        min_train_size=args.min_train_size,
        cpu_jobs=cpu_jobs,
        parallel_evals=args.parallel_evals,
        fast_mode=args.fast_mode,
        use_progress_bar=use_progress_bar,
    )

    if metrics_df.empty:
        print("No experiments were run (likely due to insufficient train size).")
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "baseline_metrics_intro_only.csv"
    json_path = args.outdir / "baseline_metrics_intro_only.json"

    metrics_df.to_csv(csv_path, index=False)
    metrics_df.to_json(json_path, orient="records", indent=2)

    print(f"Saved metrics CSV:  {csv_path}")
    print(f"Saved metrics JSON: {json_path}")

    print("\nTop models per split by PR-AUC:")
    for split_name, part in metrics_df.groupby("split", sort=True):
        best = part.sort_values("pr_auc", ascending=False).head(3)
        print(f"\n{split_name}")
        print(best[["model", "pr_auc", "precision", "recall", "balanced_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
