
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "transactions.csv"
QUEUE_PATH = ROOT / "data" / "processed" / "anomaly_review_queue.parquet"
REPORT_DIR = ROOT / "reports"

TEST_START_STEP = 632
TEST_END_STEP = 743
TOLERANCE = 0.01


def calculate_metrics(group, prediction_column):
    actual = group["isFraud"].astype(bool)
    predicted = group[prediction_column].astype(bool)

    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    tn = int((~actual & ~predicted).sum())
    fn = int((actual & ~predicted).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "transactions": len(group),
        "fraud_cases": int(actual.sum()),
        "alerts": int(predicted.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": float(predicted.mean()) if len(group) else 0.0,
    }


def main():
    for path in (RAW_PATH, QUEUE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    queue = pd.read_parquet(QUEUE_PATH)

    required_queue_columns = {"source_row_id", "is_alert"}
    missing = required_queue_columns - set(queue.columns)

    if missing:
        raise ValueError(
            f"Review queue is missing columns: {sorted(missing)}"
        )

    if queue["source_row_id"].isna().any():
        raise ValueError("Queue contains missing source_row_id values.")

    if queue["source_row_id"].duplicated().any():
        raise ValueError("Queue contains duplicate source_row_id values.")

    raw_path_sql = str(RAW_PATH).replace("\\", "/").replace("'", "''")
    con = duckdb.connect()

    try:
        raw = con.execute(
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
            WHERE step BETWEEN {TEST_START_STEP} AND {TEST_END_STEP}
            """
        ).df()
    finally:
        con.close()

    data = queue.merge(
        raw,
        on="source_row_id",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_queue", "_raw"),
    )

    if len(data) != len(queue):
        raise ValueError("Join changed the queue row count.")

    if not data["_merge"].eq("both").all():
        raise ValueError("Some queue rows did not match raw transactions.")

    # Verify that the queue represents the same source transactions.
    for column in (
        "step",
        "amount",
        "type",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ):
        queue_column = f"{column}_queue"
        raw_column = f"{column}_raw"

        if queue_column not in data.columns:
            continue

        if pd.api.types.is_numeric_dtype(data[queue_column]):
            same = (
                (data[queue_column] - data[raw_column]).abs() <= TOLERANCE
            ) | (
                data[queue_column].isna() & data[raw_column].isna()
            )
            if not same.all():
                raise ValueError(
                    f"Transaction identity check failed for {column}."
                )
        elif not data[queue_column].equals(data[raw_column]):
            raise ValueError(
                f"Transaction identity check failed for {column}."
            )

    data["isFraud"] = data["isFraud"].astype(bool)
    data["is_alert"] = data["is_alert"].astype(bool)

    tx_type = data["type_raw"]
    amount = data["amount_raw"]

    sender_delta = (
        data["newbalanceOrig_raw"] - data["oldbalanceOrg_raw"]
    )
    receiver_delta = (
        data["newbalanceDest_raw"] - data["oldbalanceDest_raw"]
    )

    sender_decrease = tx_type.isin(
        ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"]
    )
    sender_increase = tx_type.eq("CASH_IN")
    receiver_increase = tx_type.isin(["TRANSFER", "CASH_OUT"])
    receiver_decrease = tx_type.eq("CASH_IN")

    sender_error = (
        sender_delta
        + amount.where(sender_decrease, 0)
        - amount.where(sender_increase, 0)
    )

    receiver_error = (
        receiver_delta
        - amount.where(receiver_increase, 0)
        + amount.where(receiver_decrease, 0)
    )

    data["balance_mismatch"] = (
        (
            sender_decrease | sender_increase
        ) & (sender_error.abs() > TOLERANCE)
    ) | (
        (
            receiver_increase | receiver_decrease
        ) & (receiver_error.abs() > TOLERANCE)
    )

    # Compare the original alert with the stricter combined strategy.
    data["combined_alert"] = (
        data["is_alert"] & data["balance_mismatch"]
    )

    strategy_columns = {
        "IsolationForest": "is_alert",
        "Alert_AND_balance_mismatch": "combined_alert",
    }

    rows = []

    for transaction_type, group in data.groupby("type_raw", sort=True):
        for strategy_name, prediction_column in strategy_columns.items():
            metrics = calculate_metrics(group, prediction_column)
            metrics["transaction_type"] = transaction_type
            metrics["strategy"] = strategy_name
            rows.append(metrics)

    results = pd.DataFrame(rows)

    results = results[
        [
            "transaction_type",
            "strategy",
            "transactions",
            "fraud_cases",
            "alerts",
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
            "precision",
            "recall",
            "f1",
            "alert_rate",
        ]
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "strategy_evaluation_by_type.csv"
    results.to_csv(output_path, index=False)

    print("\n=== TRANSACTION-TYPE STRATEGY EVALUATION ===")
    print(results.to_string(index=False))
    print(f"\nSaved report: {output_path}")

    print(
        "\nInterpretation: this is an offline evaluation on labeled test data. "
        "It describes performance; it does not prove fraud or establish "
        "that the combined strategy is production-ready."
    )


if __name__ == "__main__":
    main()
