
from pathlib import Path
import json

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
MODEL_FILE = PROJECT_ROOT / "data" / "features" / "isolation_forest.joblib"
REPORT_FILE = PROJECT_ROOT / "reports" / "threshold_analysis.json"

ALERT_BUDGETS = [0.005, 0.01, 0.02, 0.05]
SELECTED_BUDGET = 0.02


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")



def load_split(con, source, start_step, end_step):
    query = f"""
        WITH raw AS (
            SELECT
                ROW_NUMBER() OVER () AS source_row_id,
                *
            FROM read_csv_auto(
                '{source}',
                header = true,
                parallel = false
            )
        )
        SELECT
            source_row_id,
            step,
            isFraud,
            amount,
            type,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest,

            LN(1 + GREATEST(amount, 0))
                AS log_amount,

            LN(1 + GREATEST(oldbalanceOrg, 0))
                AS log_origin_balance,

            LN(1 + GREATEST(oldbalanceDest, 0))
                AS log_destination_balance,

            CASE
                WHEN oldbalanceOrg = 0 THEN 1
                ELSE 0
            END AS origin_balance_is_zero,

            CASE
                WHEN oldbalanceDest = 0 THEN 1
                ELSE 0
            END AS destination_balance_is_zero,

            CASE
                WHEN oldbalanceOrg = 0 THEN NULL
                ELSE amount / oldbalanceOrg
            END AS amount_to_origin_balance_ratio,

            CASE WHEN type = 'CASH_IN' THEN 1 ELSE 0 END
                AS type_cash_in,

            CASE WHEN type = 'CASH_OUT' THEN 1 ELSE 0 END
                AS type_cash_out,

            CASE WHEN type = 'DEBIT' THEN 1 ELSE 0 END
                AS type_debit,

            CASE WHEN type = 'PAYMENT' THEN 1 ELSE 0 END
                AS type_payment,

            CASE WHEN type = 'TRANSFER' THEN 1 ELSE 0 END
                AS type_transfer

        FROM raw
        WHERE step BETWEEN {start_step} AND {end_step}
    """

    df = con.execute(query).df()

    if df["source_row_id"].duplicated().any():
        raise ValueError("Duplicate source_row_id values found.")

    print(
        f"Loaded steps {start_step}-{end_step}: "
        f"{len(df):,} rows"
    )

    return df



def prepare_features(df, features, medians):
    return (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
    )


def calculate_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "alerts": int(predictions.sum()),
        "alert_rate": float(predictions.mean()),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    features = saved["features"]
    medians = pd.Series(saved["training_medians"])

    con = duckdb.connect(database=":memory:")

    try:
        source = sql_path(RAW_FILE)

        print("Loading validation data...")
        validation = load_split(con, source, 521, 631)

        print("Loading test data...")
        test = load_split(con, source, 632, 743)

    finally:
        con.close()

    X_validation = prepare_features(
        validation, features, medians
    )
    X_test = prepare_features(
        test, features, medians
    )

    # Higher score means more anomalous.
    validation_scores = -model.decision_function(X_validation)
    y_validation = (
        validation["isFraud"].astype(int).to_numpy()
    )

    test_scores = -model.decision_function(X_test)
    y_test = test["isFraud"].astype(int).to_numpy()

    validation_ap = average_precision_score(
        y_validation, validation_scores
    )

    print("\n=== VALIDATION THRESHOLD COMPARISON ===")
    print(f"Validation rows: {len(validation):,}")
    print(f"Validation fraud labels: {y_validation.sum():,}")
    print(f"Validation average precision: {validation_ap:.4f}")

    results = []

    for budget in ALERT_BUDGETS:
        # Choose thresholds using validation data only.
        threshold = float(
            np.quantile(validation_scores, 1 - budget)
        )

        metrics = calculate_metrics(
            y_validation,
            validation_scores,
            threshold,
        )

        metrics["target_alert_budget"] = budget
        results.append(metrics)

        print(
            f"\nBudget: {budget:.1%}"
            f"\n  Threshold: {threshold:.6f}"
            f"\n  Actual alert rate: {metrics['alert_rate']:.2%}"
            f"\n  Precision: {metrics['precision']:.4f}"
            f"\n  Recall: {metrics['recall']:.4f}"
            f"\n  Alerts: {metrics['alerts']:,}"
        )

    # Lock the selected threshold before evaluating test results.
    selected_threshold = float(
        np.quantile(validation_scores, 1 - SELECTED_BUDGET)
    )

    print("\n=== FINAL TEST EVALUATION ===")
    print(f"Selected budget: {SELECTED_BUDGET:.1%}")
    print(f"Test rows: {len(test):,}")
    print(f"Test fraud labels: {y_test.sum():,}")

    test_metrics = calculate_metrics(
        y_test,
        test_scores,
        selected_threshold,
    )

    test_metrics["average_precision"] = float(
        average_precision_score(y_test, test_scores)
    )
    test_metrics["threshold_source"] = (
        "Validation anomaly-score quantile; "
        "test not used for threshold selection"
    )

    # Build an analyst-facing queue without fraud labels.
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    review_queue = test[
        [
            "source_row_id",
            "step",
            "amount",
            "type",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ]
    ].copy()

    review_queue["anomaly_score"] = test_scores
    review_queue["is_alert"] = (
        test_scores >= selected_threshold
    ).astype("int8")

    review_queue["review_priority"] = np.where(
        review_queue["is_alert"].eq(1),
        "Review",
        "Not flagged",
    )

    # Each source row must appear only once in the queue.
    if review_queue["source_row_id"].duplicated().any():
        raise ValueError(
            "Review queue contains duplicate source_row_id values."
        )

    if len(review_queue) != len(test):
        raise ValueError(
            "Review queue row count does not match test data."
        )

    review_queue = review_queue.sort_values(
        ["is_alert", "anomaly_score"],
        ascending=[False, False],
    )

    queue_path = output_dir / "anomaly_review_queue.parquet"
    review_queue.to_parquet(queue_path, index=False)

    print(f"\nReview queue rows: {len(review_queue):,}")
    print(
        "Unique source row IDs: "
        f"{review_queue['source_row_id'].nunique():,}"
    )
    print(
        f"Alert rows: {int(review_queue['is_alert'].sum()):,}"
    )
    print(f"Review queue saved to: {queue_path}")

    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    report = {
        "model": "IsolationForest",
        "validation_rows": int(len(validation)),
        "validation_fraud_count": int(y_validation.sum()),
        "validation_average_precision": float(validation_ap),
        "alert_budget_comparison": results,
        "selected_alert_budget": SELECTED_BUDGET,
        "selected_threshold": selected_threshold,
        "final_test_metrics": test_metrics,
        "review_queue_rows": int(len(review_queue)),
        "review_queue_alerts": int(
            review_queue["is_alert"].sum()
        ),
        "caveat": (
            "Anomaly alerts identify unusual behavior, not confirmed fraud. "
            "Ties in anomaly scores can make the actual alert rate differ "
            "from the target budget. source_row_id is a source-file row "
            "position, not a real-world transaction identifier."
        ),
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
