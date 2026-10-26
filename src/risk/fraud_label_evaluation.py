
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "transactions.csv"
QUEUE_PATH = ROOT / "data" / "processed" / "anomaly_review_queue.parquet"
BALANCE_REPORT_PATH = ROOT / "reports" / "balance_pattern_audit_v2.csv"
REPORT_DIR = ROOT / "reports"

TEST_START_STEP = 632
TEST_END_STEP = 743


def calculate_metrics(y_true, y_pred):
    y_true = pd.Series(y_true).astype(bool)
    y_pred = pd.Series(y_pred).astype(bool)

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fn = int((y_true & ~y_pred).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "rows": len(y_true),
        "alerts": int(y_pred.sum()),
        "fraud_cases": int(y_true.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": float(y_pred.mean()) if len(y_pred) else 0.0,
    }


def main():
    for path in (RAW_PATH, QUEUE_PATH, BALANCE_REPORT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    queue = pd.read_parquet(QUEUE_PATH)
    balance = pd.read_csv(BALANCE_REPORT_PATH)

    required_queue = {"source_row_id", "is_alert"}
    missing = required_queue - set(queue.columns)
    if missing:
        raise ValueError(f"Review queue missing columns: {sorted(missing)}")

    if queue["source_row_id"].isna().any():
        raise ValueError("Queue has missing source_row_id values.")

    if queue["source_row_id"].duplicated().any():
        raise ValueError("Queue source_row_id values are not unique.")

    # The balance report is aggregated by transaction type and alert status.
    # It does not contain row-level mismatch indicators, so this script
    # independently reconstructs row-level diagnostic signals from raw data.
    con = duckdb.connect()
    raw_sql_path = str(RAW_PATH).replace("\\", "/").replace("'", "''")

    labels = con.execute(
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
                '{raw_sql_path}',
                header = true,
                parallel = false
            )
        )
        SELECT *
        FROM numbered
        WHERE step BETWEEN {TEST_START_STEP} AND {TEST_END_STEP}
        """
    ).df()

    con.close()

    if labels["source_row_id"].duplicated().any():
        raise ValueError("Raw test data has duplicate source_row_id values.")

    data = queue.merge(
        labels,
        on="source_row_id",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_queue", "_raw"),
    )

    if len(data) != len(queue):
        raise ValueError("Join changed the review queue row count.")

    if not data["_merge"].eq("both").all():
        raise ValueError("Some queue rows could not be matched to raw data.")

    if data["isFraud"].isna().any():
        raise ValueError("Fraud labels contain missing values.")

    # Confirm queue identity against original raw transactions.
    for col in ("step", "amount", "type", "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest"):
        q_col = f"{col}_queue"
        r_col = f"{col}_raw"

        if q_col not in data.columns:
            # A column may only be present on the raw side.
            continue

        if pd.api.types.is_numeric_dtype(data[q_col]):
            same = (
                data[q_col].sub(data[r_col]).abs() <= 0.01
            ) | (data[q_col].isna() & data[r_col].isna())
            if not same.all():
                raise ValueError(f"Identity check failed for {col}.")
        elif not data[q_col].equals(data[r_col]):
            raise ValueError(f"Identity check failed for {col}.")

    # Use original transaction values for the diagnostic calculations.
    tx_type = data["type_raw"]
    amount = data["amount_raw"]

    sender_delta = data["newbalanceOrig_raw"] - data["oldbalanceOrg_raw"]
    receiver_delta = data["newbalanceDest_raw"] - data["oldbalanceDest_raw"]

    tolerance = 0.01

    sender_decrease = tx_type.isin(["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])
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
        ) & (sender_error.abs() > tolerance)
    ) | (
        (
            receiver_increase | receiver_decrease
        ) & (receiver_error.abs() > tolerance)
    )

    data["is_alert"] = data["is_alert"].astype(bool)
    data["isFraud"] = data["isFraud"].astype(bool)

    # Compare strategies on exactly the same test rows.
    strategies = {
        "IsolationForest alert": data["is_alert"],
        "Balance mismatch diagnostic": data["balance_mismatch"],
        "Alert OR balance mismatch": (
            data["is_alert"] | data["balance_mismatch"]
        ),
        "Alert AND balance mismatch": (
            data["is_alert"] & data["balance_mismatch"]
        ),
    }

    rows = []
    for strategy_name, predictions in strategies.items():
        metrics = calculate_metrics(data["isFraud"], predictions)
        metrics["strategy"] = strategy_name
        rows.append(metrics)

    results = pd.DataFrame(rows)

    column_order = [
        "strategy",
        "rows",
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
    results = results[column_order]

    results_path = REPORT_DIR / "fraud_label_evaluation.csv"
    results.to_csv(results_path, index=False)

    # Keep a separate alert-level summary for interpretation.
    alert_summary = (
        data.groupby("is_alert")
        .agg(
            transactions=("source_row_id", "size"),
            fraud_cases=("isFraud", "sum"),
            balance_mismatch_count=("balance_mismatch", "sum"),
        )
        .reset_index()
    )

    alert_summary["fraud_rate"] = (
        alert_summary["fraud_cases"] / alert_summary["transactions"]
    )
    alert_summary["balance_mismatch_rate"] = (
        alert_summary["balance_mismatch_count"]
        / alert_summary["transactions"]
    )

    alert_summary_path = REPORT_DIR / "fraud_and_balance_by_alert.csv"
    alert_summary.to_csv(alert_summary_path, index=False)

    print("\n=== FRAUD LABEL EVALUATION ===")
    print(f"Test rows: {len(data):,}")
    print(f"Fraud labels: {int(data['isFraud'].sum()):,}")
    print(f"Matched queue rows: {len(queue):,}")
    print("\nStrategy comparison:")
    print(results.to_string(index=False))

    print("\n=== FRAUD AND BALANCE PATTERNS BY ALERT STATUS ===")
    print(alert_summary.to_string(index=False))

    print(f"\nSaved strategy comparison: {results_path}")
    print(f"Saved alert summary: {alert_summary_path}")

    print(
        "\nIMPORTANT: This is offline evaluation on a labeled test set. "
        "Balance mismatch rules are diagnostic heuristics, not proof of fraud."
    )


if __name__ == "__main__":
    main()
