
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
THRESHOLD_REPORT_PATH = ROOT / "reports" / "threshold_analysis.json"

REPORT_DIR = ROOT / "reports"
CSV_REPORT = REPORT_DIR / "validation_business_cost.csv"
JSON_REPORT = REPORT_DIR / "validation_business_cost.json"

VALIDATION_START = 521
VALIDATION_END = 631

# Illustrative assumptions only; not verified real-world costs.
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


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def main():
    # ---------------------------------------------------------
    # 1. Validate required files
    # ---------------------------------------------------------
    for path in (RAW_PATH, MODEL_PATH, THRESHOLD_REPORT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Use the previously selected 2% alert-budget threshold.
    threshold_report = json.loads(
        THRESHOLD_REPORT_PATH.read_text(encoding="utf-8")
    )
    threshold = float(threshold_report["selected_threshold"])

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    training_medians = pd.Series(saved["training_medians"])

    if list(saved["features"]) != FEATURES:
        raise ValueError(
            "Saved model feature list does not match this script."
        )

    # ---------------------------------------------------------
    # 2. Load validation data and recreate model features
    # ---------------------------------------------------------
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
        WHERE step BETWEEN {VALIDATION_START} AND {VALIDATION_END}
    """

    con = duckdb.connect(database=":memory:")
    try:
        df = con.execute(query).df()
    finally:
        con.close()

    if df.empty:
        raise ValueError("No validation rows were loaded.")

    df["isFraud"] = df["isFraud"].astype(int)

    print(f"Validation rows: {len(df):,}")
    print(f"Validation fraud cases: {int(df['isFraud'].sum()):,}")

    # ---------------------------------------------------------
    # 3. Generate Isolation Forest anomaly scores
    # ---------------------------------------------------------
    X = (
        df[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(training_medians)
    )

    # Higher score means more anomalous.
    scores = -model.decision_function(X)

    df["anomaly_score"] = scores
    df["isolation_forest_alert"] = scores >= threshold

    # ---------------------------------------------------------
    # 4. Generate CASH_OUT rule alerts
    # ---------------------------------------------------------
    df["cashout_rule_alert"] = (
        (df["type"] == "CASH_OUT")
        & (df["amount"] > 200_000)
        & np.isclose(
            df["oldbalanceOrg"].fillna(-999999),
            df["amount"].fillna(-999999),
            rtol=0,
            atol=0.01,
        )
    )

    model_alerts = df["isolation_forest_alert"].astype(bool)
    rule_alerts = df["cashout_rule_alert"].astype(bool)

    overlap = model_alerts & rule_alerts
    combined_or = model_alerts | rule_alerts
    combined_and = model_alerts & rule_alerts

    # ---------------------------------------------------------
    # 5. Audit alert overlap
    # ---------------------------------------------------------
    print("\n=== ALERT OVERLAP AUDIT ===")
    print(f"Isolation Forest alerts: {int(model_alerts.sum()):,}")
    print(f"CASH_OUT rule alerts: {int(rule_alerts.sum()):,}")
    print(f"Both strategies alert: {int(overlap.sum()):,}")
    print(f"Combined OR alerts: {int(combined_or.sum()):,}")
    print(f"Combined AND alerts: {int(combined_and.sum()):,}")

    print("\n=== BREAKDOWN BY TRANSACTION TYPE ===")

    print(
        df.groupby("type")
        .agg(
            rows=("isFraud", "size"),
            fraud_cases=("isFraud", "sum"),
            model_alerts=("isolation_forest_alert", "sum"),
            rule_alerts=("cashout_rule_alert", "sum"),
        )
        .to_string()
    )

    # ---------------------------------------------------------
    # 6. Inspect rule-alert anomaly scores
    # ---------------------------------------------------------
    print("\n=== CASH_OUT RULE ALERT SCORE CHECK ===")

    rule_rows = df.loc[rule_alerts]

    print(f"Rule alert rows: {len(rule_rows):,}")
    print(
        "Rule alerts also flagged by Isolation Forest:",
        int((model_alerts & rule_alerts).sum()),
    )
    print(f"Selected model threshold: {threshold:.10f}")

    if not rule_rows.empty:
        print("\nAnomaly score summary:")
        print(rule_rows["anomaly_score"].describe().to_string())

        print(
            "\nMaximum rule-alert anomaly score:",
            f"{rule_rows['anomaly_score'].max():.10f}",
        )

        print(
            "Rule alerts with score at or above threshold:",
            int((rule_rows["anomaly_score"] >= threshold).sum()),
        )

    # ---------------------------------------------------------
    # 7. Threshold sensitivity diagnostic
    #    Diagnostic only: does not change the selected threshold.
    # ---------------------------------------------------------
    print("\n=== THRESHOLD SENSITIVITY CHECK ===")

    for percentile in [0.95, 0.98, 0.99]:
        candidate_threshold = float(
            np.quantile(scores, percentile)
        )

        candidate_alerts = scores >= candidate_threshold
        candidate_overlap = candidate_alerts & rule_alerts

        print(
            f"Percentile: {percentile:.0%} | "
            f"Threshold: {candidate_threshold:.6f} | "
            f"Model alerts: {int(candidate_alerts.sum()):,} | "
            f"Rule overlap: {int(candidate_overlap.sum()):,}"
        )

    # ---------------------------------------------------------
    # 8. Compare business costs at the selected threshold
    # ---------------------------------------------------------
    strategies = {
        "IsolationForest": model_alerts,
        "CASH_OUT rule": rule_alerts,
        "Combined OR": combined_or,
        "Combined AND": combined_and,
    }

    y = df["isFraud"].to_numpy()
    rows = []

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
            if precision + recall
            else 0.0
        )

        fp_cost = fp * FP_COST
        fn_cost = fn * FN_COST
        total_cost = fp_cost + fn_cost

        rows.append({
            "strategy": name,
            "rows": len(df),
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
            "false_positive_cost": fp_cost,
            "missed_fraud_cost": fn_cost,
            "total_estimated_cost": total_cost,
            "fp_cost_assumption": FP_COST,
            "fn_cost_assumption": FN_COST,
        })

    results = (
        pd.DataFrame(rows)
        .sort_values("total_estimated_cost")
        .reset_index(drop=True)
    )

    results.to_csv(CSV_REPORT, index=False)

    # ---------------------------------------------------------
    # 9. Save validation report
    # ---------------------------------------------------------
    report = {
        "evaluation_period": {
            "step_start": VALIDATION_START,
            "step_end": VALIDATION_END,
            "rows": int(len(df)),
            "fraud_cases": int(y.sum()),
        },
        "model": "IsolationForest",
        "threshold": threshold,
        "threshold_source": (
            "Previously selected using validation anomaly scores "
            "for the 2% alert-budget target"
        ),
        "cost_assumptions": {
            "false_positive_investigation_cost": FP_COST,
            "missed_fraud_cost": FN_COST,
            "note": (
                "Illustrative assumptions; replace with "
                "business-approved costs."
            ),
        },
        "average_precision": float(
            average_precision_score(y, scores)
        ),
        "strategies": results.to_dict(orient="records"),
        "warning": (
            "Validation results are for strategy analysis only. "
            "Do not use validation results as final test performance."
        ),
    }

    JSON_REPORT.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 10. Print final comparison
    # ---------------------------------------------------------
    print("\n=== VALIDATION BUSINESS-COST COMPARISON ===")

    print(
        results[
            [
                "strategy",
                "alerts",
                "true_positives",
                "false_positives",
                "precision",
                "recall",
                "alert_rate",
                "false_positive_cost",
                "missed_fraud_cost",
                "total_estimated_cost",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}",
        )
    )

    print(f"\nThreshold used: {threshold:.10f}")
    print(f"Average precision: {report['average_precision']:.6f}")
    print(f"\nCSV saved: {CSV_REPORT}")
    print(f"JSON saved: {JSON_REPORT}")


if __name__ == "__main__":
    main()
