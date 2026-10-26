
from pathlib import Path
import json

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "transactions.csv"
MODEL_PATH = ROOT / "data" / "features" / "isolation_forest.joblib"
THRESHOLD_PATH = ROOT / "reports" / "threshold_analysis.json"

REPORT_DIR = ROOT / "reports"
CSV_REPORT = REPORT_DIR / "final_test_business_cost.csv"
JSON_REPORT = REPORT_DIR / "final_test_business_cost.json"

TEST_START = 632
TEST_END = 743

# Illustrative assumptions, not verified real-world costs.
FP_COST = 5.0
FN_COST = 150.0

FEATURES = [
    "log_amount",
    "log_origin_balance",
    "log_destination_balance",
    "origin_balance_is_zero",
    "destination_balance_is_zero",
    "amount_to_origin_balance_ratio",
    "type_cash_in",
    "type_cash_out",
    "type_debit",
    "type_payment",
    "type_transfer",
]


def sql_path(path):
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def main():
    for path in (RAW_PATH, MODEL_PATH, THRESHOLD_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Freeze the threshold chosen previously using validation.
    threshold_report = json.loads(
        THRESHOLD_PATH.read_text(encoding="utf-8")
    )
    threshold = float(threshold_report["selected_threshold"])

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    medians = pd.Series(saved["training_medians"])

    if list(saved["features"]) != FEATURES:
        raise ValueError("Saved model features do not match this script.")

    # Load only the held-out test period.
    source = sql_path(RAW_PATH)

    query = f"""
        SELECT
            step,
            type,
            amount,
            oldbalanceOrg,
            isFraud,

            LN(1 + amount) AS log_amount,
            LN(1 + oldbalanceOrg) AS log_origin_balance,
            LN(1 + oldbalanceDest) AS log_destination_balance,

            CASE WHEN oldbalanceOrg = 0
                THEN 1 ELSE 0 END AS origin_balance_is_zero,

            CASE WHEN oldbalanceDest = 0
                THEN 1 ELSE 0 END AS destination_balance_is_zero,

            CASE WHEN oldbalanceOrg > 0
                THEN amount / oldbalanceOrg
                ELSE NULL END AS amount_to_origin_balance_ratio,

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

        FROM read_csv_auto(
            '{source}',
            header = true,
            parallel = false
        )
        WHERE step BETWEEN {TEST_START} AND {TEST_END}
    """

    con = duckdb.connect(database=":memory:")
    try:
        df = con.execute(query).df()
    finally:
        con.close()

    if df.empty:
        raise ValueError("No test rows loaded.")

    df["isFraud"] = df["isFraud"].astype(int)

    # Score using the existing model. Do not retrain.
    X = (
        df[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
    )

    scores = -model.decision_function(X)

    # Apply the threshold selected before this final test.
    model_alerts = scores >= threshold

    # Apply the previously defined CASH_OUT rule unchanged.
    rule_alerts = (
        (df["type"] == "CASH_OUT")
        & (df["amount"] > 200_000)
        & np.isclose(
            df["oldbalanceOrg"].fillna(-999999),
            df["amount"].fillna(-999999),
            rtol=0,
            atol=0.01,
        )
    )

    strategies = {
        "IsolationForest": model_alerts,
        "CASH_OUT rule": rule_alerts,
        "Combined OR": model_alerts | rule_alerts,
        "Combined AND": model_alerts & rule_alerts,
    }

    y = df["isFraud"].to_numpy()
    results = []

    for name, alerts in strategies.items():
        predicted = np.asarray(alerts, dtype=bool)

        tp = int(((y == 1) & predicted).sum())
        fp = int(((y == 0) & predicted).sum())
        tn = int(((y == 0) & ~predicted).sum())
        fn = int(((y == 1) & ~predicted).sum())

        alert_count = tp + fp
        precision = tp / alert_count if alert_count else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        results.append({
            "strategy": name,
            "test_rows": len(df),
            "fraud_cases": int(y.sum()),
            "alerts": alert_count,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "alert_rate": alert_count / len(df),
            "false_positive_cost": fp * FP_COST,
            "missed_fraud_cost": fn * FN_COST,
            "total_estimated_cost": fp * FP_COST + fn * FN_COST,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(CSV_REPORT, index=False)

    overlap = int((model_alerts & rule_alerts).sum())

    report = {
        "evaluation": "final held-out test",
        "test_step_start": TEST_START,
        "test_step_end": TEST_END,
        "test_rows": int(len(df)),
        "fraud_cases": int(y.sum()),
        "model": "IsolationForest",
        "threshold": threshold,
        "threshold_source": "Previously selected validation threshold",
        "model_rule_overlap": overlap,
        "average_precision": float(
            average_precision_score(y, scores)
        ),
        "cost_assumptions": {
            "false_positive_cost": FP_COST,
            "missed_fraud_cost": FN_COST,
            "note": "Illustrative assumptions, not verified business costs.",
        },
        "strategies": results_df.to_dict(orient="records"),
        "test_usage": (
            "Test labels were used only to calculate final evaluation metrics. "
            "Do not tune the model or threshold using these results."
        ),
    }

    JSON_REPORT.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== FINAL HELD-OUT TEST EVALUATION ===")
    print(f"Test rows: {len(df):,}")
    print(f"Fraud cases: {int(y.sum()):,}")
    print(f"Frozen threshold: {threshold:.10f}")
    print(f"Model alerts: {int(model_alerts.sum()):,}")
    print(f"CASH_OUT rule alerts: {int(rule_alerts.sum()):,}")
    print(f"Alert overlap: {overlap:,}")
    print(f"Average precision: {report['average_precision']:.6f}")

    print("\n=== TEST BUSINESS-COST COMPARISON ===")
    print(
        results_df[
            [
                "strategy",
                "alerts",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "alert_rate",
                "total_estimated_cost",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}",
        )
    )

    print(f"\nCSV saved: {CSV_REPORT}")
    print(f"JSON saved: {JSON_REPORT}")


if __name__ == "__main__":
    main()
