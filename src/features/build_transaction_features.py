
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "transaction_features.parquet"
)


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    source = sql_path(INPUT_FILE)
    destination = sql_path(OUTPUT_FILE)

    con = duckdb.connect(database=":memory:")

    try:
        print("Building transaction-level features...")

        con.execute(f"""
            COPY (
                SELECT
                    step,

                    -- Transaction amount and its skew-reduced version
                    amount,
                    LN(1 + amount) AS log_amount,

                    -- Balances known before the transaction
                    oldbalanceOrg AS origin_balance_before,
                    LN(1 + oldbalanceOrg) AS log_origin_balance,

                    oldbalanceDest AS destination_balance_before,
                    LN(1 + oldbalanceDest) AS log_destination_balance,

                    -- Indicators for accounts with zero prior balance
                    CASE
                        WHEN oldbalanceOrg = 0 THEN 1
                        ELSE 0
                    END AS origin_balance_is_zero,

                    CASE
                        WHEN oldbalanceDest = 0 THEN 1
                        ELSE 0
                    END AS destination_balance_is_zero,

                    -- Amount relative to prior origin balance
                    CASE
                        WHEN oldbalanceOrg > 0
                        THEN amount / oldbalanceOrg
                        ELSE NULL
                    END AS amount_to_origin_balance_ratio,

                    -- Transaction type encoded as numeric features
                    CASE WHEN type = 'CASH_IN'
                        THEN 1 ELSE 0 END AS type_cash_in,

                    CASE WHEN type = 'CASH_OUT'
                        THEN 1 ELSE 0 END AS type_cash_out,

                    CASE WHEN type = 'DEBIT'
                        THEN 1 ELSE 0 END AS type_debit,

                    CASE WHEN type = 'PAYMENT'
                        THEN 1 ELSE 0 END AS type_payment,

                    CASE WHEN type = 'TRANSFER'
                        THEN 1 ELSE 0 END AS type_transfer

                FROM read_csv_auto('{source}', header = true)
            )
            TO '{destination}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        stats = con.execute(f"""
            SELECT
                COUNT(*) AS row_count,
                COUNT(*) FILTER (
                    WHERE amount_to_origin_balance_ratio IS NULL
                ) AS missing_ratio_count,
                COUNT(*) FILTER (
                    WHERE log_amount IS NULL
                ) AS missing_log_amount_count
            FROM read_parquet('{destination}')
        """).fetchone()

        print("\n=== TRANSACTION FEATURE SUMMARY ===")
        print(f"Transaction rows: {stats[0]:,}")
        print(f"Undefined origin ratios: {stats[1]:,}")
        print(f"Missing log amounts: {stats[2]:,}")
        print(f"\nSaved to: {OUTPUT_FILE}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
