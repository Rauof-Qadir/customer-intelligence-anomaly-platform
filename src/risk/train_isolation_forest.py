
from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "isolation_forest_baseline.json"
MODEL_FILE = PROJECT_ROOT / "data" / "features" / "isolation_forest.joblib"

TRAIN_SAMPLE_SIZE = 200_000

FEATURES = [
    "log_amount",
    "log_origin_balance",
    "log_destination_balance",
    "origin_balance_is_zero",
    "destination_balance_is_zero",
    "amount_to_origin_balance_ratio",
    "type_cash_in",
    "type_cash_out",
    "type_debit",
    "type_payment",
    "type_transfer",
]


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def load_features(con, source, condition, limit=None):
    limit_sql = f"LIMIT {limit}" if limit else ""

    query = f"""
        SELECT
            step,
            amount,
            isFraud,
            log_amount,
            log_origin_balance,
            log_destination_balance,
            origin_balance_is_zero,
            destination_balance_is_zero,
            amount_to_origin_balance_ratio,
            type_cash_in,
            type_cash_out,
            type_debit,
            type_payment,
            type_transfer
        FROM (
            SELECT
                step,
                amount,
                isFraud,
                LN(1 + amount) AS log_amount,
                LN(1 + oldbalanceOrg) AS log_origin_balance,
                LN(1 + oldbalanceDest) AS log_destination_balance,

                CASE WHEN oldbalanceOrg = 0
                    THEN 1 ELSE 0 END AS origin_balance_is_zero,

                CASE WHEN oldbalanceDest = 0
                    THEN 1 ELSE 0 END AS destination_balance_is_zero,

                CASE WHEN oldbalanceOrg > 0
                    THEN amount / oldbalanceOrg
                    ELSE NULL END AS amount_to_origin_balance_ratio,

                CASE WHEN type = 'CASH_IN'
                    THEN 1 ELSE 0 END AS type_cash_in,

                CASE WHEN type = 'CASH_OUT'
                    THEN 1 ELSE 0 END AS type_cash_out,

                CASE WHEN type = 'DEBIT'
                    THEN 1 ELSE 0 END AS type_debit,

                CASE WHEN type = 'PAYMENT'
                    THEN 1 ELSE 0 END AS type_payment,

                CASE WHEN type = 'TRANSFER'
                    THEN 1 ELSE 0 END AS type_transfer

            FROM read_csv_auto('{source}', header = true)
        ) t
        WHERE {condition}
        {limit_sql}
    """

    return con.execute(query).df()


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    source = sql_path(RAW_FILE)
    con = duckdb.connect(database=":memory:")

    try:
        print("Loading deterministic training sample...")

        # Hash-based ordering gives a reproducible sample.
        train = load_features(
            con,
            source,
            "step BETWEEN 1 AND 520",
        )

        # Select a deterministic sample without using fraud labels.
        train = (
            train.assign(
                sample_key=pd.util.hash_pandas_object(
                    train[["step", "amount", "log_origin_balance"]],
                    index=False,
                ).to_numpy()
            )
            .sort_values("sample_key")
            .head(TRAIN_SAMPLE_SIZE)
            .drop(columns="sample_key")
            .reset_index(drop=True)
        )

        print(f"Training rows: {len(train):,}")
        print("Loading validation and test data...")

        validation = load_features(
            con, source, "step BETWEEN 521 AND 631"
        )

        test = load_features(
            con, source, "step BETWEEN 632 AND 743"
        )

    finally:
        con.close()

    # Fit imputation values on training data only.
    train_medians = train[FEATURES].median()

    X_train = (
        train[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(train_medians)
    )

    X_validation = (
        validation[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(train_medians)
    )

    X_test = (
        test[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(train_medians)
    )

    print("Training Isolation Forest...")

    model = IsolationForest(
        n_estimators=150,
        max_samples=256,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    # Fraud labels are deliberately excluded from model fitting.
    model.fit(X_train)

    # Higher score means more anomalous.
    validation_scores = -model.decision_function(X_validation)
    test_scores = -model.decision_function(X_test)

    # Choose a threshold using validation only:
    # flag approximately the top 1% most anomalous validation rows.
    threshold = float(np.quantile(validation_scores, 0.99))

    predictions = (test_scores >= threshold).astype(int)
    y_test = test["isFraud"].astype(int).to_numpy()

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    average_precision = average_precision_score(
        y_test, test_scores
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    report = {
        "model": "IsolationForest",
        "random_state": 42,
        "training_sample_size": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "features": FEATURES,
        "threshold_source": "99th percentile of validation anomaly scores",
        "threshold": threshold,
        "test_alert_count": int(predictions.sum()),
        "test_alert_rate": float(predictions.mean()),
        "test_fraud_count": int(y_test.sum()),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_average_precision": float(average_precision),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "interpretation": (
            "Anomaly scores indicate unusual patterns, not confirmed fraud."
        ),
    }

    # Save the model and training medians for reproducibility.
    import joblib

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "training_medians": train_medians.to_dict(),
            "threshold": threshold,
        },
        MODEL_FILE,
    )

    REPORT_FILE.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== ISOLATION FOREST BASELINE ===")
    print(f"Test rows: {len(test):,}")
    print(f"Test fraud labels: {y_test.sum():,}")
    print(f"Flagged transactions: {predictions.sum():,}")
    print(f"Alert rate: {predictions.mean():.2%}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Average precision (PR-AUC area): {average_precision:.4f}")
    print(f"\nModel saved: {MODEL_FILE}")
    print(f"Report saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
