
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "customer_features.parquet"
)


def sql_path(path: Path) -> str:
    """Return a SQL-safe path for DuckDB."""
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def build_customer_features():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    input_path = sql_path(INPUT_FILE)
    output_path = sql_path(OUTPUT_FILE)

    con = duckdb.connect(database=":memory:")

    try:
        print("Building customer-level features...")
        print("Reading transactions in DuckDB. This may take a while.")

        query = f"""
        COPY (
            WITH transactions AS (
                SELECT
                    step,
                    type,
                    amount,
                    nameOrig
                FROM read_csv_auto(
                    '{input_path}',
                    header = true
                )
            ),

            dataset_stats AS (
                SELECT MAX(step) AS dataset_max_step
                FROM transactions
            ),

            customer_aggregates AS (
                SELECT
                    nameOrig AS customer_id,

                    COUNT(*) AS transaction_count,

                    MIN(step) AS first_step,
                    MAX(step) AS last_step,

                    SUM(amount) AS total_amount,
                    AVG(amount) AS avg_amount,
                    COALESCE(STDDEV_SAMP(amount), 0)
                        AS amount_std,
                    MIN(amount) AS min_amount,
                    MAX(amount) AS max_amount,

                    COUNT(DISTINCT type)
                        AS distinct_transaction_types,

                    SUM(CASE WHEN type = 'CASH_IN'
                        THEN 1 ELSE 0 END) AS cash_in_count,

                    SUM(CASE WHEN type = 'CASH_OUT'
                        THEN 1 ELSE 0 END) AS cash_out_count,

                    SUM(CASE WHEN type = 'DEBIT'
                        THEN 1 ELSE 0 END) AS debit_count,

                    SUM(CASE WHEN type = 'PAYMENT'
                        THEN 1 ELSE 0 END) AS payment_count,

                    SUM(CASE WHEN type = 'TRANSFER'
                        THEN 1 ELSE 0 END) AS transfer_count

                FROM transactions
                GROUP BY nameOrig
            )

            SELECT
                c.customer_id,

                c.transaction_count,

                s.dataset_max_step - c.last_step
                    AS recency_steps,

                c.first_step,
                c.last_step,

                c.last_step - c.first_step
                    AS active_span_steps,

                c.total_amount,
                c.avg_amount,
                c.amount_std,
                c.min_amount,
                c.max_amount,

                c.distinct_transaction_types,

                c.cash_in_count,
                c.cash_out_count,
                c.debit_count,
                c.payment_count,
                c.transfer_count,

                c.cash_in_count * 1.0
                    / c.transaction_count AS cash_in_share,

                c.cash_out_count * 1.0
                    / c.transaction_count AS cash_out_share,

                c.debit_count * 1.0
                    / c.transaction_count AS debit_share,

                c.payment_count * 1.0
                    / c.transaction_count AS payment_share,

                c.transfer_count * 1.0
                    / c.transaction_count AS transfer_share

            FROM customer_aggregates c
            CROSS JOIN dataset_stats s

        ) TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """

        con.execute(query)

        # Verify the saved output.
        result = con.execute(
            f"""
            SELECT
                COUNT(*) AS customer_rows,
                COUNT(DISTINCT customer_id)
                    AS unique_customer_ids,
                MIN(transaction_count) AS min_transactions,
                MAX(transaction_count) AS max_transactions,
                AVG(transaction_count) AS avg_transactions
            FROM read_parquet('{output_path}')
            """
        ).fetchone()

        print("\n=== CUSTOMER FEATURE SUMMARY ===")
        print(f"Customer rows: {result[0]:,}")
        print(f"Unique customer IDs: {result[1]:,}")
        print(f"Minimum transactions: {result[2]}")
        print(f"Maximum transactions: {result[3]}")
        print(f"Average transactions: {result[4]:.4f}")
        print(f"\nFeatures saved to: {OUTPUT_FILE}")

    finally:
        con.close()


if __name__ == "__main__":
    build_customer_features()
