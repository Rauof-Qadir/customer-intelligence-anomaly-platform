
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "raw" / "transactions.csv"
OUTPUT = ROOT / "reports" / "balance_diagnostics_v2.json"

CHUNK_SIZE = 100_000


def run_diagnostics():
    if not INPUT.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT}")

    summary = {}

    for chunk in pd.read_csv(
        INPUT,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        # Actual change in the origin account balance
        chunk["origin_delta"] = (
            chunk["newbalanceOrig"] - chunk["oldbalanceOrg"]
        )

        for tx_type, group in chunk.groupby("type"):
            name = str(tx_type)

            stats = summary.setdefault(
                name,
                {
                    "rows": 0,
                    "origin_increased": 0,
                    "origin_decreased": 0,
                    "origin_unchanged": 0,
                    "expected_direction_mismatch": 0,
                    "fraud_labeled_rows": 0,
                },
            )

            delta = group["origin_delta"]
            amount = group["amount"]

            stats["rows"] += len(group)
            stats["origin_increased"] += int(delta.gt(0).sum())
            stats["origin_decreased"] += int(delta.lt(0).sum())
            stats["origin_unchanged"] += int(delta.eq(0).sum())
            stats["fraud_labeled_rows"] += int(
                group["isFraud"].eq(1).sum()
            )

            # These are investigative expectations, not error rules.
            if tx_type == "CASH_IN":
                mismatch = delta.lt(0) & amount.gt(0)
            elif tx_type in ("CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"):
                mismatch = delta.gt(0) & amount.gt(0)
            else:
                mismatch = pd.Series(False, index=group.index)

            stats["expected_direction_mismatch"] += int(
                mismatch.sum()
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== ORIGIN BALANCE BEHAVIOR ===")

    for tx_type, stats in summary.items():
        print(f"\n{tx_type}: {stats['rows']:,} rows")
        print(f"  Increased: {stats['origin_increased']:,}")
        print(f"  Decreased: {stats['origin_decreased']:,}")
        print(f"  Unchanged: {stats['origin_unchanged']:,}")
        print(
            "  Direction mismatches to investigate: "
            f"{stats['expected_direction_mismatch']:,}"
        )
        print(
            f"  Fraud-labeled rows: {stats['fraud_labeled_rows']:,}"
        )

    print(f"\nReport saved to: {OUTPUT}")


if __name__ == "__main__":
    run_diagnostics()
