
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "customer_features.parquet"
)


def main():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    con = duckdb.connect(database=":memory:")

    try:
        feature_path = str(FEATURE_FILE.resolve()).replace("\\", "/")

        print("\n=== TRANSACTIONS PER CUSTOMER ===")

        distribution = con.execute(
            f"""
            SELECT
                transaction_count,
                COUNT(*) AS customer_count,
                ROUND(
                    100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
                    4
                ) AS customer_percentage
            FROM read_parquet('{feature_path}')
            GROUP BY transaction_count
            ORDER BY transaction_count
            """
        ).df()

        print(distribution.to_string(index=False))

        print("\n=== CUSTOMER FEATURE SUMMARY ===")

        summary = con.execute(
            f"""
            SELECT
                COUNT(*) AS total_customers,
                COUNT(*) FILTER (
                    WHERE transaction_count = 1
                ) AS one_transaction_customers,
                COUNT(*) FILTER (
                    WHERE transaction_count >= 2
                ) AS repeat_customers,
                COUNT(*) FILTER (
                    WHERE transaction_count >= 3
                ) AS three_or_more_customers
            FROM read_parquet('{feature_path}')
            """
        ).df()

        print(summary.to_string(index=False))

    finally:
        con.close()


if __name__ == "__main__":
    main()
