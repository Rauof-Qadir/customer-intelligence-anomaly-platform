
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "transaction_features.parquet"
)


def main():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    path = str(FEATURE_FILE.resolve()).replace("\\", "/")
    con = duckdb.connect(database=":memory:")

    try:
        print("\n=== NUMERIC FEATURE AUDIT ===")

        result = con.execute(f"""
            SELECT
                COUNT(*) AS rows,

                COUNT(*) FILTER (
                    WHERE amount < 0
                ) AS negative_amounts,

                COUNT(*) FILTER (
                    WHERE amount = 0
                ) AS zero_amounts,

                COUNT(*) FILTER (
                    WHERE amount_to_origin_balance_ratio IS NULL
                ) AS undefined_ratios,

                COUNT(*) FILTER (
                    WHERE amount_to_origin_balance_ratio < 0
                ) AS negative_ratios,

                COUNT(*) FILTER (
                    WHERE amount_to_origin_balance_ratio > 1
                ) AS ratios_above_one,

                MIN(amount) AS min_amount,
                APPROX_QUANTILE(amount, 0.50) AS median_amount,
                APPROX_QUANTILE(amount, 0.95) AS p95_amount,
                APPROX_QUANTILE(amount, 0.99) AS p99_amount,
                MAX(amount) AS max_amount

            FROM read_parquet('{path}')
        """).fetchone()

        labels = [
            "Rows",
            "Negative amounts",
            "Zero amounts",
            "Undefined ratios",
            "Negative ratios",
            "Ratios above one",
            "Minimum amount",
            "Median amount",
            "95th percentile amount",
            "99th percentile amount",
            "Maximum amount",
        ]

        for label, value in zip(labels, result):
            if isinstance(value, (int, float)):
                print(f"{label}: {value:,.4f}")
            else:
                print(f"{label}: {value}")

        print("\n=== TRANSACTION TYPE COUNTS ===")

        # Read the source only for this diagnostic.
        raw_path = str(
            (PROJECT_ROOT / "data" / "raw" / "transactions.csv")
            .resolve()
        ).replace("\\", "/")

        type_counts = con.execute(f"""
            SELECT
                type_cash_in,
                type_cash_out,
                type_debit,
                type_payment,
                type_transfer,
                COUNT(*) AS row_count
            FROM read_parquet('{path}')
            GROUP BY ALL
            ORDER BY row_count DESC
        """).fetchall()

        for row in type_counts:
            print(row)

    finally:
        con.close()


if __name__ == "__main__":
    main()
