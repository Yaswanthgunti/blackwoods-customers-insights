"""Reproducible customer-insights case study for the Blackwoods role.

The analysis uses two public UCI datasets and compares a transparent
regularised logistic regression with a nonlinear radial-basis SVM.

Run from the repository root with:

    python src/run_analysis.py
"""

from __future__ import annotations

import json
import math
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC


SEED = 42
BOOTSTRAP_REPEATS = 1_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

RETAIL_FILE = RAW_DIR / "Online Retail.xlsx"
SHOPPER_FILE = RAW_DIR / "online_shoppers_intention.csv"

# Data source: Chen, D. (2015), UCI Online Retail, CC BY 4.0.
# https://doi.org/10.24432/C5BW33 (report citation: chen2015onlineretail)
RETAIL_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
# Data source: Sakar, C. O. and Kastro, Y. (2018), UCI Online Shoppers
# Purchasing Intention, CC BY 4.0. https://doi.org/10.24432/C5F88Q
# (report citation: sakar2018onlineshoppers)
SHOPPER_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)


DISPLAY_NAMES = {
    "recency_days": "Days since last purchase",
    "frequency_orders": "Previous orders",
    "monetary_value": "Previous spend",
    "average_order_value": "Average order value",
    "units_purchased": "Units purchased",
    "average_basket_units": "Average basket size",
    "unique_products": "Unique products",
    "active_days": "Active purchase days",
    "tenure_days": "Observed customer tenure",
    "mean_unit_price": "Mean unit price",
    "cancellation_line_rate": "Cancellation-line rate",
    "is_uk": "UK customer",
    "Administrative": "Administrative pages",
    "Administrative_Duration": "Administrative duration",
    "Informational": "Informational pages",
    "Informational_Duration": "Informational duration",
    "ProductRelated": "Product pages",
    "ProductRelated_Duration": "Product-page duration",
    "BounceRates": "Bounce rate",
    "ExitRates": "Exit rate",
    "PageValues": "Page value",
    "SpecialDay": "Special-day proximity",
    "Month": "Month",
    "OperatingSystems": "Operating system",
    "Browser": "Browser",
    "Region": "Region",
    "TrafficType": "Traffic type",
    "VisitorType": "Visitor type",
    "Weekend": "Weekend",
}


@dataclass
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: pd.Series
    preprocessor: ColumnTransformer
    identifiers: pd.Series
    profile: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, FIGURE_DIR, TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _download_and_extract(url: str, target_name: str) -> None:
    archive = RAW_DIR / f"{target_name}.zip"
    print(f"Downloading {target_name} from UCI...")
    urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as source:
        source.extractall(RAW_DIR)


def ensure_data() -> None:
    ensure_directories()
    if not RETAIL_FILE.exists():
        _download_and_extract(RETAIL_URL, "uci_online_retail")
    if not SHOPPER_FILE.exists():
        _download_and_extract(SHOPPER_URL, "uci_online_shoppers")


def build_retail_bundle() -> tuple[DatasetBundle, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(RETAIL_FILE)
    raw_rows = len(raw)
    raw_duplicates = int(raw.duplicated().sum())
    missing_customer_ids = int(raw["CustomerID"].isna().sum())

    data = raw.drop_duplicates().copy()
    data = data[data["CustomerID"].notna()].copy()
    data["CustomerID"] = data["CustomerID"].astype(int)
    data["InvoiceNoText"] = data["InvoiceNo"].astype(str)
    data["is_cancellation"] = (
        data["InvoiceNoText"].str.startswith("C") | (data["Quantity"] <= 0)
    )
    data["valid_purchase"] = (
        (~data["is_cancellation"])
        & (data["Quantity"] > 0)
        & (data["UnitPrice"] > 0)
    )
    data["line_revenue"] = data["Quantity"] * data["UnitPrice"]

    cutoff = pd.Timestamp("2011-09-01")
    history_all = data[data["InvoiceDate"] < cutoff].copy()
    history = history_all[history_all["valid_purchase"]].copy()
    future = data[
        (data["InvoiceDate"] >= cutoff) & data["valid_purchase"]
    ].copy()

    order_level = (
        history.groupby(["CustomerID", "InvoiceNoText"], as_index=False)
        .agg(
            order_date=("InvoiceDate", "max"),
            order_value=("line_revenue", "sum"),
            order_units=("Quantity", "sum"),
            order_unique_products=("StockCode", "nunique"),
        )
    )

    customer = order_level.groupby("CustomerID").agg(
        last_order=("order_date", "max"),
        first_order=("order_date", "min"),
        frequency_orders=("InvoiceNoText", "nunique"),
        monetary_value=("order_value", "sum"),
        average_order_value=("order_value", "mean"),
        units_purchased=("order_units", "sum"),
        average_basket_units=("order_units", "mean"),
        active_days=("order_date", lambda s: s.dt.date.nunique()),
    )
    customer["recency_days"] = (cutoff - customer["last_order"]).dt.days
    customer["tenure_days"] = (
        customer["last_order"] - customer["first_order"]
    ).dt.days

    product_stats = history.groupby("CustomerID").agg(
        unique_products=("StockCode", "nunique"),
        mean_unit_price=("UnitPrice", "mean"),
    )
    cancellation_rate = history_all.groupby("CustomerID")["is_cancellation"].mean()
    primary_country = history.groupby("CustomerID")["Country"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"
    )

    customer = customer.join(product_stats)
    customer["cancellation_line_rate"] = cancellation_rate.reindex(
        customer.index
    ).fillna(0.0)
    customer["is_uk"] = primary_country.reindex(customer.index).eq(
        "United Kingdom"
    ).astype(int)
    future_customers = set(future["CustomerID"].unique())
    customer["repeat_purchase"] = customer.index.to_series().isin(
        future_customers
    ).astype(int)

    features = [
        "recency_days",
        "frequency_orders",
        "monetary_value",
        "average_order_value",
        "units_purchased",
        "average_basket_units",
        "unique_products",
        "active_days",
        "tenure_days",
        "mean_unit_price",
        "cancellation_line_rate",
        "is_uk",
    ]
    X = customer[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = customer["repeat_purchase"].astype(int)

    log_features = features[:-2]
    linear_features = features[-2:]
    log_pipeline = Pipeline(
        [
            (
                "log1p",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            ),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("log_numeric", log_pipeline, log_features),
            ("linear_numeric", StandardScaler(), linear_features),
        ],
        remainder="drop",
    )

    monthly = (
        data[data["valid_purchase"]]
        .assign(month=lambda frame: frame["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(
            revenue=("line_revenue", "sum"),
            invoices=("InvoiceNoText", "nunique"),
            customers=("CustomerID", "nunique"),
        )
    )

    recency_labels = ["Most recent", "Recent", "Lapsing", "Least recent"]
    insight_frame = customer.reset_index()[
        ["CustomerID", "recency_days", "frequency_orders", "repeat_purchase"]
    ].copy()
    insight_frame["recency_group"] = pd.qcut(
        insight_frame["recency_days"],
        q=4,
        labels=recency_labels,
        duplicates="drop",
    )
    recency_insight = (
        insight_frame.groupby("recency_group", observed=True, as_index=False)
        .agg(
            customers=("CustomerID", "size"),
            repeat_purchase_rate=("repeat_purchase", "mean"),
            median_recency_days=("recency_days", "median"),
        )
    )

    customer.reset_index().to_csv(
        PROCESSED_DIR / "online_retail_customer_features.csv", index=False
    )
    monthly.to_csv(TABLE_DIR / "online_retail_monthly_summary.csv", index=False)
    recency_insight.to_csv(
        TABLE_DIR / "online_retail_recency_insight.csv", index=False
    )

    profile = {
        "source_rows": raw_rows,
        "source_columns": int(raw.shape[1]),
        "source_duplicate_rows": raw_duplicates,
        "missing_customer_ids": missing_customer_ids,
        "known_customers": int(data["CustomerID"].nunique()),
        "modelling_customers": int(len(customer)),
        "history_start": str(history["InvoiceDate"].min()),
        "history_end": str(history["InvoiceDate"].max()),
        "outcome_start": str(future["InvoiceDate"].min()),
        "outcome_end": str(future["InvoiceDate"].max()),
        "positive_cases": int(y.sum()),
        "positive_rate": float(y.mean()),
    }

    return (
        DatasetBundle(
            name="Future repeat purchase",
            X=X,
            y=y,
            preprocessor=preprocessor,
            identifiers=pd.Series(customer.index, index=customer.index, name="CustomerID"),
            profile=profile,
        ),
        monthly,
        recency_insight,
    )


def build_shopper_bundle() -> tuple[DatasetBundle, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(SHOPPER_FILE)
    raw_rows = len(raw)
    raw_duplicates = int(raw.duplicated().sum())
    data = raw.drop_duplicates().reset_index(drop=True)

    target = "Revenue"
    X = data.drop(columns=[target]).copy()
    y = data[target].astype(int)

    categorical = [
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
    ]
    numeric = [column for column in X.columns if column not in categorical]
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            ),
        ],
        remainder="drop",
    )

    visitor_insight = (
        data.groupby("VisitorType", as_index=False)
        .agg(sessions=("Revenue", "size"), purchase_rate=("Revenue", "mean"))
        .sort_values("purchase_rate", ascending=False)
    )
    # Preserve the large, meaningful mass at exactly zero instead of splitting
    # tied zero values arbitrarily across quantiles. Positive values are then
    # divided into quartiles to show the monotonic business relationship.
    data["page_value_group"] = "Zero"
    positive_page_value = data["PageValues"] > 0
    data.loc[positive_page_value, "page_value_group"] = pd.qcut(
        data.loc[positive_page_value, "PageValues"],
        q=4,
        labels=["Positive Q1", "Positive Q2", "Positive Q3", "Positive Q4"],
        duplicates="drop",
    ).astype(str)
    data["page_value_group"] = pd.Categorical(
        data["page_value_group"],
        categories=["Zero", "Positive Q1", "Positive Q2", "Positive Q3", "Positive Q4"],
        ordered=True,
    )
    page_value_insight = (
        data.groupby("page_value_group", observed=True, as_index=False)
        .agg(
            sessions=("Revenue", "size"),
            purchase_rate=("Revenue", "mean"),
            median_page_value=("PageValues", "median"),
        )
    )
    visitor_insight.to_csv(
        TABLE_DIR / "online_shoppers_visitor_insight.csv", index=False
    )
    page_value_insight.to_csv(
        TABLE_DIR / "online_shoppers_page_value_insight.csv", index=False
    )
    data.to_csv(PROCESSED_DIR / "online_shoppers_deduplicated.csv", index=False)

    profile = {
        "source_rows": raw_rows,
        "source_columns": int(raw.shape[1]),
        "source_duplicate_rows": raw_duplicates,
        "modelling_sessions": int(len(data)),
        "missing_values": int(data.isna().sum().sum()),
        "positive_cases": int(y.sum()),
        "positive_rate": float(y.mean()),
    }

    return (
        DatasetBundle(
            name="Session purchase conversion",
            X=X,
            y=y,
            preprocessor=preprocessor,
            identifiers=pd.Series(data.index, name="session_id"),
            profile=profile,
        ),
        visitor_insight,
        page_value_insight,
    )


def candidate_models(preprocessor: ColumnTransformer) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    # Implemented with scikit-learn (Pedregosa et al., 2011;
    # https://jmlr.org/papers/v12/pedregosa11a.html). The nonlinear comparison
    # follows the support-vector formulation of Cortes and Vapnik (1995),
    # https://doi.org/10.1007/BF00994018.
    logistic = Pipeline(
        [
            ("preprocess", clone(preprocessor)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5_000,
                    random_state=SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )
    svm = Pipeline(
        [
            ("preprocess", clone(preprocessor)),
            (
                "model",
                SVC(
                    class_weight="balanced",
                    kernel="rbf",
                    cache_size=1_000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    return {
        "Logistic regression": (
            logistic,
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "RBF support vector machine": (
            svm,
            {
                "model__C": [0.5, 1.0, 5.0, 10.0],
                "model__gamma": ["scale", 0.01, 0.1],
            },
        ),
    }


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def metric_values(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (scores >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(SEED)
    collected: dict[str, list[float]] = {
        "average_precision": [],
        "roc_auc": [],
        "f1": [],
        "recall": [],
        "precision": [],
    }
    n = len(y_true)
    for _ in range(repeats):
        indices = rng.integers(0, n, n)
        y_sample = y_true[indices]
        if np.unique(y_sample).size < 2:
            continue
        sample_metrics = metric_values(y_sample, scores[indices], threshold)
        for metric in collected:
            collected[metric].append(sample_metrics[metric])
    return {
        metric: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for metric, values in collected.items()
    }


def paired_ap_difference(
    y_true: np.ndarray,
    svm_scores: np.ndarray,
    logistic_scores: np.ndarray,
    repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, float]:
    observed = average_precision_score(y_true, svm_scores) - average_precision_score(
        y_true, logistic_scores
    )
    rng = np.random.default_rng(SEED + 7)
    differences: list[float] = []
    n = len(y_true)
    for _ in range(repeats):
        indices = rng.integers(0, n, n)
        y_sample = y_true[indices]
        if np.unique(y_sample).size < 2:
            continue
        difference = average_precision_score(
            y_sample, svm_scores[indices]
        ) - average_precision_score(y_sample, logistic_scores[indices])
        differences.append(float(difference))
    return {
        "svm_minus_logistic_ap": float(observed),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
    }


def lift_table(y_true: np.ndarray, scores: np.ndarray, dataset: str, model: str) -> pd.DataFrame:
    ranked = pd.DataFrame({"actual": y_true, "score": scores}).sort_values(
        "score", ascending=False
    )
    base_rate = float(ranked["actual"].mean())
    rows = []
    for percentage in range(10, 101, 10):
        n = max(1, math.ceil(len(ranked) * percentage / 100))
        selected = ranked.head(n)
        precision_at_depth = float(selected["actual"].mean())
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "population_percent": percentage,
                "precision_at_depth": precision_at_depth,
                "lift": precision_at_depth / base_rate if base_rate else np.nan,
                "captured_positives": float(selected["actual"].sum() / ranked["actual"].sum()),
            }
        )
    return pd.DataFrame(rows)


def fit_dataset(bundle: DatasetBundle) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.25,
        stratify=bundle.y,
        random_state=SEED,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    model_results: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    lift_frames: list[pd.DataFrame] = []

    for model_name, (pipeline, grid) in candidate_models(bundle.preprocessor).items():
        print(f"Tuning {model_name} for {bundle.name}...")
        # Average precision is the primary selection metric because the session
        # outcome is imbalanced; see Saito and Rehmsmeier (2015),
        # https://doi.org/10.1371/journal.pone.0118432.
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        search.fit(X_train, y_train)

        oof_scores = cross_val_predict(
            clone(search.best_estimator_),
            X_train,
            y_train,
            cv=cv,
            method="decision_function",
            n_jobs=-1,
        )
        threshold = best_f1_threshold(y_train.to_numpy(), oof_scores)
        estimator = search.best_estimator_
        test_scores = estimator.decision_function(X_test)
        metrics = metric_values(y_test.to_numpy(), test_scores, threshold)
        intervals = bootstrap_metric_intervals(
            y_test.to_numpy(), test_scores, threshold
        )

        permutation = permutation_importance(
            estimator,
            X_test,
            y_test,
            scoring="average_precision",
            n_repeats=25,
            random_state=SEED,
            n_jobs=-1,
        )
        for feature, mean, std in zip(
            X_test.columns,
            permutation.importances_mean,
            permutation.importances_std,
        ):
            importance_rows.append(
                {
                    "dataset": bundle.name,
                    "model": model_name,
                    "feature": feature,
                    "display_feature": DISPLAY_NAMES.get(feature, feature),
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                }
            )

        prediction = (test_scores >= threshold).astype(int)
        prediction_rows.append(
            pd.DataFrame(
                {
                    "dataset": bundle.name,
                    "model": model_name,
                    "row_index": X_test.index.astype(str),
                    "actual": y_test.to_numpy(),
                    "score": test_scores,
                    "prediction": prediction,
                }
            )
        )
        lift_frames.append(
            lift_table(
                y_test.to_numpy(), test_scores, bundle.name, model_name
            )
        )
        for metric_name, value in metrics.items():
            low, high = intervals.get(metric_name, (np.nan, np.nan))
            metric_rows.append(
                {
                    "dataset": bundle.name,
                    "model": model_name,
                    "metric": metric_name,
                    "value": value,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

        model_results[model_name] = {
            "best_params": search.best_params_,
            "cv_average_precision": float(search.best_score_),
            "threshold": threshold,
            "test_metrics": metrics,
            "bootstrap_95_ci": intervals,
            "test_scores": test_scores,
            "test_actual": y_test.to_numpy(),
        }

    metrics_frame = pd.DataFrame(metric_rows)
    importance_frame = pd.DataFrame(importance_rows)
    predictions_frame = pd.concat(prediction_rows, ignore_index=True)
    lifts_frame = pd.concat(lift_frames, ignore_index=True)

    svm_result = model_results["RBF support vector machine"]
    logistic_result = model_results["Logistic regression"]
    comparison = paired_ap_difference(
        svm_result["test_actual"],
        svm_result["test_scores"],
        logistic_result["test_scores"],
    )

    for result in model_results.values():
        result.pop("test_scores", None)
        result.pop("test_actual", None)

    return {
        "dataset": bundle.name,
        "profile": bundle.profile,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "models": model_results,
        "paired_ap_comparison": comparison,
        "metrics_frame": metrics_frame,
        "importance_frame": importance_frame,
        "predictions_frame": predictions_frame,
        "lifts_frame": lifts_frame,
    }


def save_model_outputs(results: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.concat([result["metrics_frame"] for result in results], ignore_index=True)
    importance = pd.concat(
        [result["importance_frame"] for result in results], ignore_index=True
    )
    predictions = pd.concat(
        [result["predictions_frame"] for result in results], ignore_index=True
    )
    lifts = pd.concat([result["lifts_frame"] for result in results], ignore_index=True)

    metrics.to_csv(TABLE_DIR / "model_metrics.csv", index=False)
    importance.to_csv(TABLE_DIR / "permutation_importance.csv", index=False)
    predictions.to_csv(TABLE_DIR / "holdout_predictions.csv", index=False)
    lifts.to_csv(TABLE_DIR / "lift_by_decile.csv", index=False)

    paired_rows = []
    for result in results:
        paired_rows.append({"dataset": result["dataset"], **result["paired_ap_comparison"]})
    pd.DataFrame(paired_rows).to_csv(
        TABLE_DIR / "paired_average_precision_difference.csv", index=False
    )

    serialisable_results = []
    for result in results:
        compact = {
            key: value
            for key, value in result.items()
            if not key.endswith("_frame")
        }
        serialisable_results.append(_json_ready(compact))
    with (TABLE_DIR / "model_results.json").open("w", encoding="utf-8") as handle:
        json.dump(serialisable_results, handle, indent=2)

    profile_rows = []
    for result in results:
        for field, value in result["profile"].items():
            profile_rows.append(
                {"dataset": result["dataset"], "field": field, "value": value}
            )
    pd.DataFrame(profile_rows).to_csv(TABLE_DIR / "dataset_profiles.csv", index=False)
    return metrics, importance, lifts


def _save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#243447",
            "text.color": "#243447",
            "axes.edgecolor": "#AAB7C4",
            "grid.color": "#DCE3E8",
            "grid.linewidth": 0.6,
        }
    )


def plot_model_performance(metrics: pd.DataFrame) -> None:
    selected = metrics[metrics["metric"].isin(["average_precision", "roc_auc"])].copy()
    model_order = ["Logistic regression", "RBF support vector machine"]
    palette = {
        "Logistic regression": "#1F77B4",
        "RBF support vector machine": "#F28E2B",
    }
    datasets = list(selected["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(10.2, 4.2), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    metric_labels = {"average_precision": "PR-AUC", "roc_auc": "ROC-AUC"}
    for ax, dataset in zip(axes, datasets):
        subset = selected[selected["dataset"] == dataset]
        x = np.arange(2)
        width = 0.34
        for offset_index, model in enumerate(model_order):
            model_rows = subset[subset["model"] == model].set_index("metric")
            values = [model_rows.loc[m, "value"] for m in ["average_precision", "roc_auc"]]
            positions = x + (offset_index - 0.5) * width
            lower = [
                max(0.0, value - model_rows.loc[m, "ci_low"])
                if pd.notna(model_rows.loc[m, "ci_low"])
                else 0.0
                for m, value in zip(["average_precision", "roc_auc"], values)
            ]
            upper = [
                max(0.0, model_rows.loc[m, "ci_high"] - value)
                if pd.notna(model_rows.loc[m, "ci_high"])
                else 0.0
                for m, value in zip(["average_precision", "roc_auc"], values)
            ]
            bars = ax.bar(
                positions,
                values,
                width,
                label=model.replace("RBF support vector machine", "RBF SVM"),
                color=palette[model],
                edgecolor="white",
                linewidth=0.8,
                yerr=np.array([lower, upper]),
                error_kw={"ecolor": "#334E68", "elinewidth": 0.8, "capsize": 2},
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.018,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(x, [metric_labels[m] for m in ["average_precision", "roc_auc"]])
        ax.set_ylim(0, 1.08)
        ax.set_title(dataset, fontsize=11)
        ax.set_xlabel("")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Holdout score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Model performance is similar; SVM adds little holdout gain",
        y=0.995,
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Scores are from untouched holdout sets; model selection used training folds only.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.77, bottom=0.16, wspace=0.12)
    _save_figure(fig, "model_performance")


def plot_feature_importance(importance: pd.DataFrame, metrics: pd.DataFrame) -> None:
    best_models = (
        metrics[metrics["metric"] == "average_precision"]
        .sort_values(["dataset", "value"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .first()[["dataset", "model"]]
    )
    datasets = list(best_models["dataset"])
    fig, axes = plt.subplots(1, len(datasets), figsize=(11, 4.6))
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        model = best_models.loc[best_models["dataset"] == dataset, "model"].iloc[0]
        subset = importance[
            (importance["dataset"] == dataset) & (importance["model"] == model)
        ].nlargest(7, "importance_mean")
        subset = subset.sort_values("importance_mean")
        ax.barh(
            subset["display_feature"],
            subset["importance_mean"],
            xerr=subset["importance_std"],
            color="#4C78A8",
            alpha=0.92,
            error_kw={"ecolor": "#334E68", "elinewidth": 0.8, "capsize": 2},
        )
        ax.axvline(0, color="#657786", linewidth=0.8)
        ax.set_title(f"{dataset}\nBest model: {model.replace('RBF support vector machine', 'RBF SVM')}", fontsize=10.5)
        ax.set_xlabel("Decrease in PR-AUC when shuffled")
        ax.set_ylabel("")
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Tenure and session value dominate the two decision layers", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, "permutation_importance")


def plot_lift(lifts: pd.DataFrame, metrics: pd.DataFrame) -> None:
    best_models = (
        metrics[metrics["metric"] == "average_precision"]
        .sort_values(["dataset", "value"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .first()[["dataset", "model"]]
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    styles = [
        ("#1F77B4", "o"),
        ("#F28E2B", "s"),
    ]
    for (dataset, model), (color, marker) in zip(
        best_models.itertuples(index=False, name=None), styles
    ):
        subset = lifts[(lifts["dataset"] == dataset) & (lifts["model"] == model)]
        ax.plot(
            subset["population_percent"],
            subset["lift"],
            marker=marker,
            color=color,
            linewidth=2,
            label=dataset,
        )
    ax.axhline(1, color="#657786", linestyle="--", linewidth=1, label="Random targeting")
    ax.set_xlabel("Highest-scored share of customers or sessions (%)")
    ax.set_ylabel("Lift over the base purchase rate")
    ax.set_xticks(range(10, 101, 10))
    ax.set_title("Targeted prioritisation concentrates likely purchasers", fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, "decision_lift")


def plot_business_insights(
    recency: pd.DataFrame,
    visitor: pd.DataFrame,
    page_value: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    recency_plot = recency.copy()
    recency_plot["rate_percent"] = recency_plot["repeat_purchase_rate"] * 100
    bars = axes[0].bar(
        recency_plot["recency_group"],
        recency_plot["rate_percent"],
        color=["#1B6CA8", "#4C90C0", "#8EBAD5", "#C9DDEB"],
    )
    for bar, value in zip(bars, recency_plot["rate_percent"]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}%",
            ha="center",
            fontsize=8.5,
        )
    axes[0].set_title("Repeat purchase falls as customers lapse", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Future repeat-purchase rate")
    axes[0].set_ylim(0, min(105, recency_plot["rate_percent"].max() + 12))
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].spines[["top", "right"]].set_visible(False)

    page_plot = page_value.copy()
    page_plot["rate_percent"] = page_plot["purchase_rate"] * 100
    bars = axes[1].bar(
        page_plot["page_value_group"],
        page_plot["rate_percent"],
        color=["#FDE0C5", "#F8C58C", "#F2A65A", "#E98232", "#C65C18"],
    )
    for bar, value in zip(bars, page_plot["rate_percent"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}%",
            ha="center",
            fontsize=8.5,
        )
    axes[1].set_title("High-value page journeys signal conversion", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Session purchase rate")
    axes[1].set_ylim(0, min(105, page_plot["rate_percent"].max() + 12))
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("The datasets provide complementary customer-decision signals", y=1.04, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, "business_insights")


def main() -> None:
    ensure_data()
    set_plot_style()

    print("Preparing customer-level retail prediction data...")
    retail_bundle, _, recency_insight = build_retail_bundle()
    print("Preparing session-level conversion data...")
    shopper_bundle, visitor_insight, page_value_insight = build_shopper_bundle()

    results = [fit_dataset(retail_bundle), fit_dataset(shopper_bundle)]
    metrics, importance, lifts = save_model_outputs(results)

    plot_model_performance(metrics)
    plot_feature_importance(importance, metrics)
    plot_lift(lifts, metrics)
    plot_business_insights(recency_insight, visitor_insight, page_value_insight)

    print("Analysis complete.")
    print(metrics.pivot_table(index=["dataset", "model"], columns="metric", values="value").round(3))


if __name__ == "__main__":
    main()
