
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "raw" / "transactions.csv"
OUTPUT = ROOT / "reports" / "balance_diagnostics.json"

CHUNK_SIZE = 100_000
TOLERANCE = 0.01


def run_diagnostics():
    if not INPUT.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT}")

    results = {}

    for chunk in pd.read_csv(
        INPUT,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        # Group results by transaction type
        for tx_type, group in chunk.groupby("type", dropna=False):
            name = str(tx_type)

            stats = results.setdefault(
                name,
                {
                    "rows": 0,
                    "zero_amount_rows": 0,
                    "sender_balance_mismatches": 0,
                    "receiver_balance_mismatches": 0,
                    "sender_balance_unchanged": 0,
                    "receiver_balance_unchanged": 0,
                },
            )

            amount = group["amount"]
            old_org = group["oldbalanceOrg"]
            new_org = group["newbalanceOrig"]
            old_dest = group["oldbalanceDest"]
            new_dest = group["newbalanceDest"]

            sender_expected = old_org - amount
            receiver_expected = old_dest + amount

            stats["rows"] += len(group)
            stats["zero_amount_rows"] += int(
                amount.eq(0).sum()
            )

            # Diagnostic comparisons, not automatic error labels
            stats["sender_balance_mismatches"] += int(
                (
                    (new_org - sender_expected).abs()
                    > TOLERANCE
                ).sum()
            )

            stats["receiver_balance_mismatches"] += int(
                (
                    (new_dest - receiver_expected).abs()
                    > TOLERANCE
                ).sum()
            )

            stats["sender_balance_unchanged"] += int(
                new_org.eq(old_org).sum()
            )

            stats["receiver_balance_unchanged"] += int(
                new_dest.eq(old_dest).sum()
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(results, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== BALANCE DIAGNOSTICS BY TYPE ===")

    for tx_type, stats in results.items():
        print(f"\nTransaction type: {tx_type}")
        print(f"Rows: {stats['rows']:,}")
        print(f"Zero amounts: {stats['zero_amount_rows']:,}")
        print(
            "Sender mismatches: "
            f"{stats['sender_balance_mismatches']:,}"
        )
        print(
            "Receiver mismatches: "
            f"{stats['receiver_balance_mismatches']:,}"
        )
        print(
            "Sender unchanged: "
            f"{stats['sender_balance_unchanged']:,}"
        )
        print(
            "Receiver unchanged: "
            f"{stats['receiver_balance_unchanged']:,}"
        )

    print(f"\nReport saved to: {OUTPUT}")


if __name__ == "__main__":
    run_diagnostics()
