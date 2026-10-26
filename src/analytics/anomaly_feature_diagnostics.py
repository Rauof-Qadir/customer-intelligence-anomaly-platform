
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "anomaly_review_queue.parquet"
OUTPUT_FILE = ROOT / "reports" / "anomaly_feature_diagnostics.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    # Diagnostic features derived from available queue columns.
    df["amount_to_sender_balance"] = (
        df["amount"] / df["oldbalanceOrg"].where(
            df["oldbalanceOrg"] > 0
        )
    )

    df["sender_balance_is_zero"] = (
        df["oldbalanceOrg"] == 0
    ).astype(int)

    df["receiver_balance_is_zero"] = (
        df["oldbalanceDest"] == 0
    ).astype(int)

    numeric_features = [
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "amount_to_sender_balance",
        "sender_balance_is_zero",
        "receiver_balance_is_zero",
        "anomaly_score",
    ]

    report = (
        df.groupby(["type", "is_alert"])[numeric_features]
        .median()
        .reset_index()
        .sort_values(["type", "is_alert"])
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_FILE, index=False)

    print(report.to_string(index=False))
    print(f"\nSaved report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
