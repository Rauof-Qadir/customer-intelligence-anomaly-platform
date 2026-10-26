
from pathlib import Path
import json
import pandas as pd

# Project ke root folder ko locate karo
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input aur output file paths
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality_report.json"


def inspect_transactions():
    """Inspect a CSV in chunks and create a data quality report."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    total_rows = 0
    missing_counts = {}
    numeric_stats = {}
    categorical_counts = {}
    columns = None
    detected_dtypes = {}

    # Read 100,000 rows at a time to limit memory usage
    for chunk in pd.read_csv(
        INPUT_FILE,
        chunksize=100_000,
        low_memory=False
    ):
        total_rows += len(chunk)

        if columns is None:
            columns = chunk.columns.tolist()
            detected_dtypes = {
                col: str(dtype)
                for col, dtype in chunk.dtypes.items()
            }

            missing_counts = {col: 0 for col in columns}

            for col in columns:
                if (
                    pd.api.types.is_object_dtype(chunk[col])
                    or pd.api.types.is_string_dtype(chunk[col])
                    or pd.api.types.is_bool_dtype(chunk[col])
                ):
                    categorical_counts[col] = {}

        # Count missing values in every chunk
        for col in columns:
            missing_counts[col] += int(chunk[col].isna().sum())

        # Summarize numeric columns without storing all rows
        for col in chunk.select_dtypes(include="number").columns:
            series = chunk[col].dropna()

            if series.empty:
                continue

            stats = numeric_stats.setdefault(
                col,
                {
                    "count": 0,
                    "sum": 0.0,
                    "min": None,
                    "max": None,
                },
            )

            stats["count"] += int(series.count())
            stats["sum"] += float(series.sum())
            chunk_min = float(series.min())
            chunk_max = float(series.max())

            stats["min"] = (
                chunk_min if stats["min"] is None
                else min(stats["min"], chunk_min)
            )
            stats["max"] = (
                chunk_max if stats["max"] is None
                else max(stats["max"], chunk_max)
            )

        # Keep the 10 most frequent values per categorical column
        for col in categorical_counts:
            counts = chunk[col].value_counts(dropna=False)

            for value, count in counts.items():
                key = "<MISSING>" if pd.isna(value) else str(value)
                categorical_counts[col][key] = (
                    categorical_counts[col].get(key, 0)
                    + int(count)
                )

    for col, stats in numeric_stats.items():
        stats["mean"] = (
            stats["sum"] / stats["count"]
            if stats["count"] else None
        )
        del stats["sum"]

    categorical_summary = {
        col: dict(
            sorted(
                counts.items(),
                key=lambda item: item[1],
                reverse=True
            )[:10]
        )
        for col, counts in categorical_counts.items()
    }

    report = {
        "file_name": INPUT_FILE.name,
        "file_size_mb": round(
            INPUT_FILE.stat().st_size / (1024 ** 2), 2
        ),
        "total_rows": total_rows,
        "total_columns": len(columns or []),
        "columns": columns,
        "detected_dtypes": detected_dtypes,
        "missing_values": missing_counts,
        "numeric_summary": numeric_stats,
        "top_categorical_values": categorical_summary,
        "duplicate_check": (
            "Not performed yet; will be implemented separately."
        ),
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8"
    )

    print("\n=== DATASET INSPECTION ===")
    print(f"File: {INPUT_FILE.name}")
    print(f"Rows: {total_rows:,}")
    print(f"Columns: {len(columns or []):,}")
    print(f"Size: {report['file_size_mb']} MB")

    print("\n=== COLUMNS AND DATA TYPES ===")
    for col in columns or []:
        print(f"{col}: {detected_dtypes[col]}")

    print("\n=== MISSING VALUES ===")
    for col, count in missing_counts.items():
        print(f"{col}: {count:,}")

    print("\n=== NUMERIC SUMMARY ===")
    for col, stats in numeric_stats.items():
        print(
            f"{col}: min={stats['min']}, "
            f"max={stats['max']}, mean={stats['mean']}"
        )

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    inspect_transactions()
