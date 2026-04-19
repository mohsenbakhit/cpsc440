"""
train_baselines.py

Train baseline classifiers on normalized legislative data and save evaluation metrics.

Usage
-----
python scripts/train_baselines.py
python scripts/train_baselines.py --input data/normalized/bills.csv --outdir data/normalized

Outputs
-------
- <outdir>/baseline_metrics.csv
- <outdir>/baseline_metrics.json

Notes
-----
- Evaluates both in-country temporal splits and cross-country transfer.
- Uses PR-AUC (average precision) as the primary metric for class imbalance.
- This script uses the currently normalized features as-is; some process features
  may reflect downstream legislative progress depending on your research setup.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT_DIR / "data" / "normalized" / "bills.csv"
DEFAULT_OUTDIR = ROOT_DIR / "data" / "normalized"

# Shared metadata features currently present in normalized output.
NUMERIC_COLS = [
    "year",
    "title_word_count",
    "description_word_count",
    "month_introduced",
    "parliament_number",
    "session_number",
    "reinstated",
    "reached_house_second_reading",
    "reached_house_third_reading",
    "reached_senate_third_reading",
    "days_active",
    "num_sponsors",
    "num_history_steps",
    "num_text_versions",
    "num_rollcalls",
    "final_yea_pct",
    "has_committee",
]

CATEGORICAL_COLS = [
    "bill_type",
    "bill_type_raw",
    "chamber",
    "party",
]

TEXT_COL = "text_blob"
TARGET_COL = "passed"


@dataclass
class Split:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline bill-passage classifiers")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to normalized CSV")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Directory for metrics output")
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


def resolve_jobs(jobs: int) -> int:
    if jobs == -1:
        return max(1, os.cpu_count() or 1)
    return max(1, jobs)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure target is integer 0/1.
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    # Parse dates for temporal sorting; missing dates fall back later.
    if "introduced_date" in df.columns:
        df["introduced_date"] = pd.to_datetime(df["introduced_date"], errors="coerce")
    else:
        df["introduced_date"] = pd.NaT

    # Build a compact text column from title + description.
    title = df.get("title", "").fillna("").astype(str)
    desc = df.get("description", "").fillna("").astype(str)
    df[TEXT_COL] = (title + " " + desc).str.strip()

    # Coerce numeric columns safely when present.
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure categorical columns exist and are strings.
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # Stable temporal key: introduced_date, then year, then original order.
    df["_row_id"] = np.arange(len(df), dtype=int)
    year_fallback = pd.to_numeric(df.get("year"), errors="coerce")
    df["_sort_year"] = year_fallback.fillna(-1)

    return df


def get_metadata_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_COLS),
            ("cat", cat_pipe, CATEGORICAL_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def get_metadata_text_preprocessor(max_features: int, ngram_range: tuple[int, int]) -> ColumnTransformer:
    metadata = get_metadata_preprocessor()

    text_pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=ngram_range,
                    min_df=5,
                    max_features=max_features,
                    stop_words="english",
                ),
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("meta", metadata, NUMERIC_COLS + CATEGORICAL_COLS),
            ("text", text_pipe, TEXT_COL),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def get_text_only_preprocessor(max_features: int, ngram_range: tuple[int, int]) -> ColumnTransformer:
    text_pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=ngram_range,
                    min_df=5,
                    max_features=max_features,
                    stop_words="english",
                ),
            )
        ]
    )

    return ColumnTransformer(
        transformers=[("text", text_pipe, TEXT_COL)],
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_models(cpu_jobs: int, fast_mode: bool, parallel_evals: bool) -> dict[str, Pipeline]:
    meta = get_metadata_preprocessor()

    # If parallelizing across model fits, keep per-model threading conservative
    # to avoid CPU oversubscription.
    inner_jobs = 1 if parallel_evals else cpu_jobs

    if fast_mode:
        meta_text_features = 10000
        text_only_features = 15000
        ngram_range = (1, 1)
    else:
        meta_text_features = 20000
        text_only_features = 30000
        ngram_range = (1, 2)

    meta_text = get_metadata_text_preprocessor(
        max_features=meta_text_features,
        ngram_range=ngram_range,
    )
    text_only = get_text_only_preprocessor(
        max_features=text_only_features,
        ngram_range=ngram_range,
    )

    models: dict[str, Pipeline] = {
        "dummy_stratified": Pipeline(
            steps=[
                ("prep", meta),
                ("model", DummyClassifier(strategy="stratified", random_state=42)),
            ]
        ),
        "logreg_l2_meta": Pipeline(
            steps=[
                ("prep", meta),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=2000,
                        n_jobs=inner_jobs,
                    ),
                ),
            ]
        ),
        "logreg_l1_meta": Pipeline(
            steps=[
                ("prep", meta),
                (
                    "model",
                    LogisticRegression(
                        penalty="l1",
                        C=0.5,
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=3000,
                        n_jobs=inner_jobs,
                    ),
                ),
            ]
        ),
        "random_forest_meta": Pipeline(
            steps=[
                ("prep", meta),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=inner_jobs,
                    ),
                ),
            ]
        ),
        "logreg_l2_meta_text": Pipeline(
            steps=[
                ("prep", meta_text),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        solver="saga",
                        class_weight="balanced",
                        max_iter=5000,
                        n_jobs=inner_jobs,
                    ),
                ),
            ]
        ),
        "linear_svm_meta_text": Pipeline(
            steps=[
                ("prep", meta_text),
                (
                    "model",
                    LinearSVC(
                        C=0.8,
                        class_weight="balanced",
                        max_iter=5000,
                    ),
                ),
            ]
        ),
        "naive_bayes_text": Pipeline(
            steps=[
                ("prep", text_only),
                ("model", MultinomialNB(alpha=0.5)),
            ]
        ),
    }

    return models


def make_temporal_split(df: pd.DataFrame, source: str, test_frac: float = 0.2) -> Split | None:
    part = df[df["source"] == source].copy()
    if len(part) < 2:
        return None

    part = part.sort_values(["introduced_date", "_sort_year", "_row_id"])
    cut = int(len(part) * (1 - test_frac))
    cut = max(1, min(cut, len(part) - 1))

    train_idx = part.index[:cut].to_numpy()
    test_idx = part.index[cut:].to_numpy()

    return Split(name=f"temporal_{source}", train_idx=train_idx, test_idx=test_idx)


def make_cross_country_split(df: pd.DataFrame, train_source: str, test_source: str) -> Split | None:
    train_idx = df[df["source"] == train_source].index.to_numpy()
    test_idx = df[df["source"] == test_source].index.to_numpy()

    if len(train_idx) == 0 or len(test_idx) == 0:
        return None

    return Split(name=f"transfer_{train_source}_to_{test_source}", train_idx=train_idx, test_idx=test_idx)


def get_score_values(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    # Fall back to class predictions when score outputs are unavailable.
    return model.predict(x)


def safe_metric(fn: Callable, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        value = fn(y_true, y_score)
        return float(value)
    except Exception:
        return float("nan")


def evaluate_model(model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_score = get_score_values(model, x_test)

    out = {
        "pr_auc": safe_metric(average_precision_score, y_test, y_score),
        "roc_auc": safe_metric(roc_auc_score, y_test, y_score),
        "f1": safe_metric(lambda a, b: f1_score(a, b, zero_division=0), y_test, y_pred),
        "precision": safe_metric(lambda a, b: precision_score(a, b, zero_division=0), y_test, y_pred),
        "recall": safe_metric(lambda a, b: recall_score(a, b, zero_division=0), y_test, y_pred),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, y_test, y_pred),
        "predicted_positive_rate": float(np.mean(y_pred == 1)),
    }
    return out


def evaluate_model_named(
    model_name: str,
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    t0 = time.perf_counter()
    metrics = evaluate_model(model, x_train, y_train, x_test, y_test)
    metrics["fit_seconds"] = round(time.perf_counter() - t0, 3)
    metrics["model"] = model_name
    return metrics


def run_experiments(
    df: pd.DataFrame,
    min_train_size: int,
    cpu_jobs: int,
    parallel_evals: bool,
    fast_mode: bool,
    use_progress_bar: bool,
) -> pd.DataFrame:
    models = build_models(cpu_jobs=cpu_jobs, fast_mode=fast_mode, parallel_evals=parallel_evals)

    splits = [
        make_temporal_split(df, "us"),
        make_temporal_split(df, "canada"),
        make_cross_country_split(df, "us", "canada"),
        make_cross_country_split(df, "canada", "us"),
    ]
    splits = [s for s in splits if s is not None]

    rows: list[dict] = []
    total_evals = len(splits) * len(models)
    eval_counter = 0

    if use_progress_bar and tqdm is not None:
        progress = tqdm(total=total_evals, desc="Training baselines", unit="model")
    else:
        progress = None

    for split in splits:
        print(f"\n[split] {split.name}")
        x_train = df.loc[split.train_idx, NUMERIC_COLS + CATEGORICAL_COLS + [TEXT_COL]]
        y_train = df.loc[split.train_idx, TARGET_COL]
        x_test = df.loc[split.test_idx, NUMERIC_COLS + CATEGORICAL_COLS + [TEXT_COL]]
        y_test = df.loc[split.test_idx, TARGET_COL]

        if len(x_train) < min_train_size:
            continue

        print(
            f"  train={len(x_train):,}, test={len(x_test):,}, "
            f"train_pos={y_train.mean():.4f}, test_pos={y_test.mean():.4f}"
        )

        train_source_counts = df.loc[split.train_idx, "source"].value_counts().to_dict()
        test_source_counts = df.loc[split.test_idx, "source"].value_counts().to_dict()

        if parallel_evals:
            print(f"  evaluating {len(models)} models in parallel (jobs={cpu_jobs})")
            eval_results = Parallel(n_jobs=cpu_jobs, prefer="threads")(
                delayed(evaluate_model_named)(
                    model_name,
                    model,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                )
                for model_name, model in models.items()
            )

            for metrics in eval_results:
                eval_counter += 1
                if progress is not None:
                    progress.update(1)
                else:
                    print(f"  [{eval_counter}/{total_evals}] done {metrics['model']} ({metrics['fit_seconds']:.2f}s)")

                rows.append(
                    {
                        "split": split.name,
                        "model": metrics["model"],
                        "n_train": int(len(x_train)),
                        "n_test": int(len(x_test)),
                        "train_positive_rate": float(y_train.mean()),
                        "test_positive_rate": float(y_test.mean()),
                        "train_source_mix": json.dumps(train_source_counts),
                        "test_source_mix": json.dumps(test_source_counts),
                        **{k: v for k, v in metrics.items() if k != "model"},
                    }
                )
        else:
            for model_name, model in models.items():
                metrics = evaluate_model_named(model_name, model, x_train, y_train, x_test, y_test)
                eval_counter += 1
                if progress is not None:
                    progress.update(1)
                else:
                    print(f"  [{eval_counter}/{total_evals}] done {model_name} ({metrics['fit_seconds']:.2f}s)")

                rows.append(
                    {
                        "split": split.name,
                        "model": model_name,
                        "n_train": int(len(x_train)),
                        "n_test": int(len(x_test)),
                        "train_positive_rate": float(y_train.mean()),
                        "test_positive_rate": float(y_test.mean()),
                        "train_source_mix": json.dumps(train_source_counts),
                        "test_source_mix": json.dumps(test_source_counts),
                        **{k: v for k, v in metrics.items() if k != "model"},
                    }
                )

    if progress is not None:
        progress.close()

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.sort_values(["split", "pr_auc"], ascending=[True, False]).reset_index(drop=True)
    return result


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

    print(f"Loading normalized data: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    df = clean_dataframe(df)

    cpu_jobs = resolve_jobs(args.jobs)
    use_progress_bar = (not args.no_progress_bar) and (tqdm is not None)

    print(f"CPU workers: {cpu_jobs}")
    if args.fast_mode:
        print("Fast mode: ON (lighter TF-IDF settings)")
    if args.parallel_evals:
        print("Parallel evaluations: ON")
    if args.use_gpu:
        print("GPU flag set, but current sklearn baseline models run on CPU.")
    if tqdm is None and not args.no_progress_bar:
        print("tqdm is not installed; falling back to print progress updates.")

    print(f"Rows: {len(df):,} | Sources: {df['source'].value_counts().to_dict()}")
    print("Running baseline experiments...")
    metrics_df = run_experiments(
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
    csv_path = args.outdir / "baseline_metrics.csv"
    json_path = args.outdir / "baseline_metrics.json"

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
