
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "transactions.csv"
REPORT_DIR = ROOT / "reports"

VALIDATION_START = 521
VALIDATION_END = 631


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_path_sql = str(RAW_PATH).replace("\\", "/").replace("'", "''")

    con = duckdb.connect()

    try:
        df = con.execute(
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
                    header=true,
                    parallel=false
                )
            )
            SELECT *
            FROM numbered
            WHERE step BETWEEN {VALIDATION_START} AND {VALIDATION_END}
              AND type = 'CASH_OUT'
            """
        ).df()
    finally:
        con.close()

    if df.empty:
        raise ValueError("No CASH_OUT validation rows were found.")

    df["isFraud"] = df["isFraud"].astype(bool)

    # Derived features
    df["amount_to_origin_balance_ratio"] = (
        df["amount"] / df["oldbalanceOrg"].replace(0, np.nan)
    )

    df["sender_delta_error"] = (
        df["newbalanceOrig"] - df["oldbalanceOrg"] + df["amount"]
    )

    df["receiver_delta_error"] = (
        df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
    )

    df["sender_balance_matches_amount"] = np.isclose(
        df["oldbalanceOrg"],
        df["amount"],
        rtol=0,
        atol=0.01,
    )

    df["sender_balance_change_matches_amount"] = np.isclose(
        df["sender_delta_error"],
        0,
        rtol=0,
        atol=0.01,
    )

    df["receiver_balance_change_matches_amount"] = np.isclose(
        df["receiver_delta_error"],
        0,
        rtol=0,
        atol=0.01,
    )

    # These are investigation hypotheses, not proven fraud rules.
    rules = {
        "amount_gt_200k": df["amount"] > 200_000,

        "amount_gt_400k": df["amount"] > 400_000,

        "amount_gt_200k_and_sender_balance_matches": (
            (df["amount"] > 200_000)
            & df["sender_balance_matches_amount"]
        ),

        "amount_gt_200k_and_sender_delta_matches": (
            (df["amount"] > 200_000)
            & df["sender_balance_change_matches_amount"]
        ),

        "amount_gt_200k_and_receiver_delta_matches": (
            (df["amount"] > 200_000)
            & df["receiver_balance_change_matches_amount"]
        ),

        "amount_gt_200k_and_ratio_ge_0_8": (
            (df["amount"] > 200_000)
            & (df["amount_to_origin_balance_ratio"] >= 0.8)
        ),

        "amount_gt_200k_and_both_deltas_match": (
            (df["amount"] > 200_000)
            & df["sender_balance_change_matches_amount"]
            & df["receiver_balance_change_matches_amount"]
        ),
    }

    y_true = df["isFraud"].to_numpy()

    print("\n=== DEBUG: LABEL AND ARRAY TYPES ===")
    print("df shape:", df.shape)
    print("isFraud dtype:", df["isFraud"].dtype)
    print("y_true dtype:", y_true.dtype)
    print("Fraud count:", np.count_nonzero(y_true))
    print("Non-fraud count:", np.count_nonzero(~y_true.astype(bool)))
    print("Sample labels:", y_true[:10])

    total_rows = len(df)
    total_fraud = int(y_true.sum())


    y_true = df["isFraud"].astype(bool).to_numpy()

    results = []

    for rule_name, rule_mask in rules.items():
        y_pred = (
            pd.Series(rule_mask, index=df.index)
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        # Explicit confusion matrix using integer counts
        tp = int(np.count_nonzero(y_true & y_pred))
        fp = int(np.count_nonzero((~y_true) & y_pred))
        tn = int(np.count_nonzero((~y_true) & (~y_pred)))
        fn = int(np.count_nonzero(y_true & (~y_pred)))

        if rule_name == "amount_gt_200k_and_sender_balance_matches":
            print("\n=== FIRST RULE CONFUSION MATRIX DEBUG ===")
            print("TP:", tp, "FP:", fp, "TN:", tn, "FN:", fn)
            print("Total from confusion matrix:", tp + fp + tn + fn)
            print("Total rows in df:", len(df))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        # Mandatory integrity checks
        assert tp + fp == int(y_pred.sum())
        assert tp + fn == int(y_true.sum())
        assert tp + fp + tn + fn == len(df)

        results.append({
            "rule": rule_name,
            "transactions": len(df),
            "fraud_cases": int(y_true.sum()),
            "alerts": tp + fp,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "alert_rate": (tp + fp) / len(df),
        })

    report = pd.DataFrame(results).sort_values(
        ["recall", "precision"],
        ascending=False,
    )

    print("\n=== REPORT DATAFRAME DEBUG ===")
    print(
        report[
            [
                "rule",
                "true_positives",
                "false_positives",
                "true_negatives",
                "false_negatives",
            ]
        ].to_string(index=False)
    )

    print("\nColumn data types:")
    print(report[
        [
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
        ]
    ].dtypes)


    output_path = REPORT_DIR / "cashout_rule_evaluation_validation.csv"
    report.to_csv(output_path, index=False)

    print("\n=== CASH_OUT RULE EVALUATION: VALIDATION ONLY ===")
    print(f"Transactions: {total_rows:,}")
    print(f"Fraud cases: {total_fraud:,}")
    print("\n=== RULE COMPARISON ===")

    print("\n=== RULE COMPARISON ===")
    print(report.to_csv(index=False))

    print(f"\nSaved report: {output_path}")
    print(
        "\nIMPORTANT: Rules are exploratory hypotheses. "
        "Do not evaluate or tune these rules on the reserved test set."
    )


if __name__ == "__main__":
    main()
