
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "transactions.csv"
QUEUE_PATH = ROOT / "data" / "processed" / "anomaly_review_queue.parquet"
REPORT_DIR = ROOT / "reports"

TEST_START_STEP = 632
TEST_END_STEP = 743

# Allow tiny floating-point differences.
BALANCE_TOLERANCE = 0.01


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    if not QUEUE_PATH.exists():
        raise FileNotFoundError(f"Review queue not found: {QUEUE_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    queue = pd.read_parquet(QUEUE_PATH)

    required_queue_columns = {
        "source_row_id",
        "step",
        "amount",
        "type",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "is_alert",
    }

    missing = required_queue_columns - set(queue.columns)
    if missing:
        raise ValueError(
            f"Review queue is missing required columns: {sorted(missing)}"
        )

    if queue["source_row_id"].isna().any():
        raise ValueError("Queue contains missing source_row_id values.")

    if queue["source_row_id"].duplicated().any():
        raise ValueError("Queue contains duplicate source_row_id values.")

    con = duckdb.connect()

    raw_path_sql = str(RAW_PATH).replace("\\", "/").replace("'", "''")

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
                newbalanceDest
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

    con.close()

    if raw["source_row_id"].duplicated().any():
        raise ValueError("Raw test data contains duplicate source_row_id values.")

    merged = queue.merge(
        raw,
        on="source_row_id",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_queue", "_raw"),
    )

    print("\n=== JOIN VALIDATION ===")
    print(f"Queue rows: {len(queue):,}")
    print(f"Raw test rows: {len(raw):,}")
    print(f"Merged rows: {len(merged):,}")
    print("\nJoin status:")
    print(merged["_merge"].value_counts(dropna=False))

    if len(merged) != len(queue):
        raise ValueError("Join changed the review queue row count.")

    if not merged["_merge"].eq("both").all():
        raise ValueError("Some queue rows could not be matched to raw data.")

    identity_checks = [
        ("step_queue", "step_raw"),
        ("amount_queue", "amount_raw"),
        ("type_queue", "type_raw"),
        ("oldbalanceOrg_queue", "oldbalanceOrg_raw"),
        ("newbalanceOrig_queue", "newbalanceOrig_raw"),
        ("oldbalanceDest_queue", "oldbalanceDest_raw"),
        ("newbalanceDest_queue", "newbalanceDest_raw"),
    ]

    for left_col, right_col in identity_checks:
        if not merged[left_col].equals(merged[right_col]):
            # Numeric columns can occasionally differ in dtype.
            if pd.api.types.is_numeric_dtype(merged[left_col]) and (
                pd.api.types.is_numeric_dtype(merged[right_col])
            ):
                equal = (
                    merged[left_col].sub(merged[right_col]).abs()
                    <= BALANCE_TOLERANCE
                ) | (
                    merged[left_col].isna() & merged[right_col].isna()
                )
                if not equal.all():
                    raise ValueError(
                        f"Source-row identity check failed: {left_col}"
                    )
            else:
                raise ValueError(
                    f"Source-row identity check failed: {left_col}"
                )

    print("\nSource-row identity checks passed.")

    # Always calculate audit metrics from the matched raw columns.
    merged["sender_balance_delta"] = (
        merged["newbalanceOrig_raw"] - merged["oldbalanceOrg_raw"]
    )

    merged["receiver_balance_delta"] = (
        merged["newbalanceDest_raw"] - merged["oldbalanceDest_raw"]
    )

    amount = merged["amount_raw"]
    tx_type = merged["type_raw"]

    # Expected direction for common transaction types.
    # These rules are diagnostic conventions, not fraud labels.
    sender_should_decrease = tx_type.isin(["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])
    sender_should_increase = tx_type.eq("CASH_IN")

    receiver_should_increase = tx_type.isin(["TRANSFER", "CASH_OUT"])
    receiver_should_decrease = tx_type.eq("CASH_IN")

    merged["sender_direction_mismatch"] = (
        (
            sender_should_decrease
            & (merged["sender_balance_delta"] > BALANCE_TOLERANCE)
        )
        | (
            sender_should_increase
            & (merged["sender_balance_delta"] < -BALANCE_TOLERANCE)
        )
    )

    merged["receiver_direction_mismatch"] = (
        (
            receiver_should_increase
            & (merged["receiver_balance_delta"] < -BALANCE_TOLERANCE)
        )
        | (
            receiver_should_decrease
            & (merged["receiver_balance_delta"] > BALANCE_TOLERANCE)
        )
    )

    # Expected balance movement under simplified transaction assumptions.
    # Some dataset rows do not follow these assumptions, so interpret as
    # a diagnostic signal rather than proof of incorrect data.
    merged["sender_delta_error"] = (
        merged["sender_balance_delta"]
        + amount.where(sender_should_decrease, 0)
        - amount.where(sender_should_increase, 0)
    )

    merged["receiver_delta_error"] = (
        merged["receiver_balance_delta"]
        - amount.where(receiver_should_increase, 0)
        + amount.where(receiver_should_decrease, 0)
    )

    merged["sender_amount_mismatch"] = (
        sender_should_decrease | sender_should_increase
    ) & (merged["sender_delta_error"].abs() > BALANCE_TOLERANCE)

    merged["receiver_amount_mismatch"] = (
        receiver_should_increase | receiver_should_decrease
    ) & (merged["receiver_delta_error"].abs() > BALANCE_TOLERANCE)

    merged["any_balance_mismatch"] = (
        merged["sender_direction_mismatch"]
        | merged["receiver_direction_mismatch"]
        | merged["sender_amount_mismatch"]
        | merged["receiver_amount_mismatch"]
    )

    # Summary for each transaction type and alert status.
    summary = (
        merged.groupby(["type_raw", "is_alert"], dropna=False)
        .agg(
            rows=("amount_raw", "size"),
            median_amount=("amount_raw", "median"),
            sender_direction_mismatch_count=(
                "sender_direction_mismatch", "sum"
            ),
            receiver_direction_mismatch_count=(
                "receiver_direction_mismatch", "sum"
            ),
            sender_amount_mismatch_count=("sender_amount_mismatch", "sum"),
            receiver_amount_mismatch_count=("receiver_amount_mismatch", "sum"),
            any_balance_mismatch_count=("any_balance_mismatch", "sum"),
        )
        .reset_index()
        .rename(columns={"type_raw": "type"})
    )

    count_columns = [
        "sender_direction_mismatch_count",
        "receiver_direction_mismatch_count",
        "sender_amount_mismatch_count",
        "receiver_amount_mismatch_count",
        "any_balance_mismatch_count",
    ]

    for col in count_columns:
        summary[col.replace("_count", "_rate")] = (
            summary[col] / summary["rows"]
        )

    summary_path = REPORT_DIR / "balance_pattern_audit_v2.csv"
    summary.to_csv(summary_path, index=False)

    overall = (
        merged.groupby("is_alert", dropna=False)
        .agg(
            rows=("source_row_id", "size"),
            any_balance_mismatch_count=("any_balance_mismatch", "sum"),
            sender_amount_mismatch_count=("sender_amount_mismatch", "sum"),
            receiver_amount_mismatch_count=("receiver_amount_mismatch", "sum"),
        )
        .reset_index()
    )

    overall["any_balance_mismatch_rate"] = (
        overall["any_balance_mismatch_count"] / overall["rows"]
    )

    overall["sender_amount_mismatch_rate"] = (
        overall["sender_amount_mismatch_count"] / overall["rows"]
    )

    overall["receiver_amount_mismatch_rate"] = (
        overall["receiver_amount_mismatch_count"] / overall["rows"]
    )

    overall_path = REPORT_DIR / "balance_pattern_alert_comparison.csv"
    overall.to_csv(overall_path, index=False)

    print("\n=== BALANCE PATTERN AUDIT V2 ===")
    print(summary.to_string(index=False))

    print("\n=== ALERT VS NON-ALERT COMPARISON ===")
    print(overall.to_string(index=False))

    print(f"\nSaved type-level report: {summary_path}")
    print(f"Saved alert comparison: {overall_path}")
    print(
        "\nInterpretation: balance mismatches are diagnostic signals, "
        "not proof of fraud."
    )


if __name__ == "__main__":
    main()
