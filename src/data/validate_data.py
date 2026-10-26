
from pathlib import Path
import json

import pandas as pd

from src.data.validation_rules import (
    REQUIRED_COLUMNS,
    get_missing_columns,
    count_basic_quality_issues,
    count_cash_out_direction_mismatches,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "validation_report.json"

CHUNK_SIZE = 100_000
TOLERANCE = 0.01


def validate_transactions():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    report = {
        "file_name": INPUT_FILE.name,
        "total_rows": 0,
        "zero_amount_rows": 0,
        "negative_amount_rows": 0,
        "negative_balance_rows": 0,
        "invalid_step_rows": 0,
        "invalid_fraud_label_rows": 0,
        "invalid_flag_label_rows": 0,
        "sender_balance_mismatch_rows": 0,
        "receiver_balance_mismatch_rows": 0,
        "cash_out_direction_mismatch_rows": 0,
        "checks": {},
    }

    columns_checked = False

    for chunk in pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        # Check required columns once, on the first chunk.
        if not columns_checked:
            missing_columns = get_missing_columns(chunk.columns)

            if missing_columns:
                raise ValueError(
                    f"Required columns missing: {missing_columns}"
                )

            columns_checked = True

        report["total_rows"] += len(chunk)

        # Reuse tested quality rules instead of duplicating them.
        quality_issues = count_basic_quality_issues(chunk)

        for key, value in quality_issues.items():
            report[key] += value

        # Diagnostic: sender balance mismatch for TRANSFER/CASH_OUT.
        sender_mask = chunk["type"].isin(["TRANSFER", "CASH_OUT"])

        expected_sender_balance = (
            chunk["oldbalanceOrg"] - chunk["amount"]
        )

        sender_mismatch = (
            (
                chunk["newbalanceOrig"] - expected_sender_balance
            ).abs()
            > TOLERANCE
        )

        report["sender_balance_mismatch_rows"] += int(
            (sender_mask & sender_mismatch).sum()
        )

        # Diagnostic: receiver balance mismatch for CASH_IN.
        receiver_mask = chunk["type"].eq("CASH_IN")

        expected_receiver_balance = (
            chunk["oldbalanceDest"] + chunk["amount"]
        )

        receiver_mismatch = (
            (
                chunk["newbalanceDest"] - expected_receiver_balance
            ).abs()
            > TOLERANCE
        )

        report["receiver_balance_mismatch_rows"] += int(
            (receiver_mask & receiver_mismatch).sum()
        )

        # Investigative signal: sender balance increases during CASH_OUT.
        report["cash_out_direction_mismatch_rows"] += (
            count_cash_out_direction_mismatches(
                chunk,
                tolerance=TOLERANCE,
            )
        )

    report["checks"] = {
        "required_columns": "Passed",
        "zero_amounts": "Investigate; not automatically invalid",
        "balance_mismatches": (
            "Diagnostic only; investigate before drawing conclusions"
        ),
        "cash_out_direction_mismatches": (
            "Investigation candidates; not confirmed errors or fraud"
        ),
        "duplicate_rows": "Not checked yet",
    }

    report["validation_summary"] = {
        "negative_amounts_found": (
            report["negative_amount_rows"] > 0
        ),
        "negative_balances_found": (
            report["negative_balance_rows"] > 0
        ),
        "invalid_steps_found": report["invalid_step_rows"] > 0,
        "invalid_labels_found": (
            report["invalid_fraud_label_rows"] > 0
            or report["invalid_flag_label_rows"] > 0
        ),
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    REPORT_FILE.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== VALIDATION REPORT ===")
    print(f"Total rows: {report['total_rows']:,}")

    for key, value in report.items():
        if key.endswith("_rows") and key != "total_rows":
            print(f"{key}: {value:,}")

    print("\n=== VALIDATION SUMMARY ===")
    for key, value in report["validation_summary"].items():
        print(f"{key}: {value}")

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    validate_transactions()
