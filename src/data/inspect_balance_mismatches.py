
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "raw" / "transactions.csv"

CHUNK_SIZE = 100_000
TOLERANCE = 0.01


def inspect_mismatches():
    matches = []

    for chunk in pd.read_csv(
        INPUT,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        # A CASH_OUT normally should not increase
        # the sender's balance when amount > 0.
        delta = chunk["newbalanceOrig"] - chunk["oldbalanceOrg"]

        mask = (
            chunk["type"].eq("CASH_OUT")
            & chunk["amount"].gt(0)
            & (delta > TOLERANCE)
        )

        if mask.any():
            cols = [
                "step",
                "type",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
                "isFraud",
                "isFlaggedFraud",
            ]

            matches.extend(
                chunk.loc[mask, cols].to_dict(orient="records")
            )

    print(f"\nDirection mismatches found: {len(matches)}")

    for i, row in enumerate(matches, start=1):
        print(f"\n--- Transaction {i} ---")
        for key, value in row.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    inspect_mismatches()
