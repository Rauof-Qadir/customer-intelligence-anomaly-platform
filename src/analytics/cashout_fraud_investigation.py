
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "transactions.csv"
QUEUE_PATH = ROOT / "data" / "processed" / "anomaly_review_queue.parquet"
REPORT_DIR = ROOT / "reports"

# Validation and test windows established in this project.
VALIDATION_START, VALIDATION_END = 521, 631
TEST_START, TEST_END = 632, 743


def load_split(con, raw_path_sql, start_step, end_step):
    return con.execute(
        f"""
        WITH numbered AS (
            SELECT
                ROW_NUMBER() OVER () AS source_row_id,
                step,
                type,
                amount,
                oldbalanceOrg,
                newbalanceOrig,
                oldbalanceDest,
                newbalanceDest,
                isFraud
            FROM read_csv_auto(
                '{raw_path_sql}',
                header = true,
                parallel = false
            )
        )
        SELECT *
        FROM numbered
        WHERE step BETWEEN {start_step} AND {end_step}
        """
    ).df()


def add_features(df):
    df = df.copy()

    df["log_amount"] = np.log1p(df["amount"].clip(lower=0))
    df["log_origin_balance"] = np.log1p(
        df["oldbalanceOrg"].clip(lower=0)
    )
    df["log_destination_balance"] = np.log1p(
        df["oldbalanceDest"].clip(lower=0)
    )

    df["origin_balance_is_zero"] = (
        df["oldbalanceOrg"] == 0
    ).astype("int8")

    df["destination_balance_is_zero"] = (
        df["oldbalanceDest"] == 0
    ).astype("int8")

    df["amount_to_origin_balance_ratio"] = (
        df["amount"] / df["oldbalanceOrg"].replace(0, np.nan)
    )

    df["sender_balance_delta"] = (
        df["newbalanceOrig"] - df["oldbalanceOrg"]
    )

    df["receiver_balance_delta"] = (
        df["newbalanceDest"] - df["oldbalanceDest"]
    )

    df["sender_delta_error"] = (
        df["sender_balance_delta"] + df["amount"]
    )

    df["receiver_delta_error"] = (
        df["receiver_balance_delta"] - df["amount"]
    )

    return df


def make_feature_summary(df, feature_columns, split_name):
    rows = []

    for feature in feature_columns:
        for fraud_value, group in df.groupby("isFraud", dropna=False):
            values = pd.to_numeric(group[feature], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).dropna()

            rows.append(
                {
                    "split": split_name,
                    "feature": feature,
                    "isFraud": int(fraud_value),
                    "transactions": len(group),
                    "valid_feature_values": len(values),
                    "mean": values.mean() if len(values) else np.nan,
                    "median": values.median() if len(values) else np.nan,
                    "p90": values.quantile(0.90) if len(values) else np.nan,
                    "p99": values.quantile(0.99) if len(values) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def main():
    for path in (RAW_PATH, QUEUE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    queue = pd.read_parquet(QUEUE_PATH)

    required_queue_columns = {
        "source_row_id",
        "step",
        "type",
        "amount",
        "anomaly_score",
        "is_alert",
    }

    missing = required_queue_columns - set(queue.columns)
    if missing:
        raise ValueError(
            f"Review queue is missing columns: {sorted(missing)}"
        )

    if queue["source_row_id"].isna().any():
        raise ValueError("Queue contains missing source_row_id values.")

    if queue["source_row_id"].duplicated().any():
        raise ValueError("Queue source_row_id values are not unique.")

    raw_path_sql = str(RAW_PATH).replace("\\", "/").replace("'", "''")
    con = duckdb.connect()

    try:
        validation = load_split(
            con, raw_path_sql, VALIDATION_START, VALIDATION_END
        )
        test = load_split(con, raw_path_sql, TEST_START, TEST_END)
    finally:
        con.close()

    # Restrict investigation to CASH_OUT transactions.
    validation = validation.loc[validation["type"] == "CASH_OUT"].copy()
    test = test.loc[test["type"] == "CASH_OUT"].copy()

    # Attach model scores only to the matching test transactions.
    test_queue = queue.loc[queue["type"] == "CASH_OUT"].copy()

    if test_queue["source_row_id"].duplicated().any():
        raise ValueError("CASH_OUT queue rows have duplicate source IDs.")

    test_scored = test.merge(
        test_queue[
            ["source_row_id", "anomaly_score", "is_alert"]
        ],
        on="source_row_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not test_scored["_merge"].eq("both").all():
        raise ValueError(
            "Some CASH_OUT test rows could not be matched to model scores."
        )

    if len(test_scored) != len(test_queue):
        raise ValueError(
            "CASH_OUT test row counts differ between raw data and queue."
        )

    validation = add_features(validation)
    test_scored = add_features(test_scored)

    feature_columns = [
        "amount",
        "log_amount",
        "oldbalanceOrg",
        "log_origin_balance",
        "oldbalanceDest",
        "log_destination_balance",
        "origin_balance_is_zero",
        "destination_balance_is_zero",
        "amount_to_origin_balance_ratio",
        "sender_balance_delta",
        "receiver_balance_delta",
        "sender_delta_error",
        "receiver_delta_error",
    ]

    validation_summary = make_feature_summary(
        validation, feature_columns, "validation"
    )

    test_summary = make_feature_summary(
        test_scored, feature_columns, "test_descriptive_only"
    )

    feature_report = pd.concat(
        [validation_summary, test_summary],
        ignore_index=True,
    )

    feature_path = REPORT_DIR / "cashout_feature_investigation.csv"
    feature_report.to_csv(feature_path, index=False)

    # Validation fraud rates: descriptive analysis only.
    validation_group_summary = (
        validation.groupby("isFraud", dropna=False)
        .agg(
            transactions=("source_row_id", "size"),
            median_amount=("amount", "median"),
            median_origin_balance=("oldbalanceOrg", "median"),
            median_destination_balance=("oldbalanceDest", "median"),
            zero_origin_balance_rate=(
                "origin_balance_is_zero", "mean"
            ),
            zero_destination_balance_rate=(
                "destination_balance_is_zero", "mean"
            ),
        )
        .reset_index()
    )

    validation_group_path = (
        REPORT_DIR / "cashout_validation_fraud_comparison.csv"
    )
    validation_group_summary.to_csv(validation_group_path, index=False)

    # Scores are summarized on the test set for reporting only.
    # Do not use these test results to choose features or thresholds.
    test_scored["isFraud"] = test_scored["isFraud"].astype(bool)
    test_scored["is_alert"] = test_scored["is_alert"].astype(bool)

    score_summary = (
        test_scored.groupby("isFraud", dropna=False)
        .agg(
            transactions=("source_row_id", "size"),
            alerts=("is_alert", "sum"),
            median_anomaly_score=("anomaly_score", "median"),
            p90_anomaly_score=(
                "anomaly_score",
                lambda x: x.quantile(0.90),
            ),
            mean_anomaly_score=("anomaly_score", "mean"),
        )
        .reset_index()
    )

    score_summary["alert_rate"] = (
        score_summary["alerts"] / score_summary["transactions"]
    )

    score_path = REPORT_DIR / "cashout_test_score_summary.csv"
    score_summary.to_csv(score_path, index=False)

    print("\n=== CASH_OUT INVESTIGATION ===")
    print(f"Validation CASH_OUT rows: {len(validation):,}")
    print(f"Validation fraud cases: {int(validation['isFraud'].sum()):,}")
    print(f"Test CASH_OUT rows: {len(test_scored):,}")
    print(f"Test fraud cases: {int(test_scored['isFraud'].sum()):,}")
    print(f"Test alerts: {int(test_scored['is_alert'].sum()):,}")

    print("\n=== VALIDATION FRAUD VS NON-FRAUD ===")
    print(validation_group_summary.to_string(index=False))

    print("\n=== TEST ANOMALY SCORES (DESCRIPTIVE ONLY) ===")
    print(score_summary.to_string(index=False))

    print(f"\nSaved feature report: {feature_path}")
    print(f"Saved validation comparison: {validation_group_path}")
    print(f"Saved test score summary: {score_path}")

    print(
        "\nIMPORTANT: Use validation data for feature and rule selection. "
        "Test results are descriptive and must remain reserved for final "
        "evaluation. Balance differences are not proof of fraud."
    )


if __name__ == "__main__":
    main()
