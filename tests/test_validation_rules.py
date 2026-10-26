
import pandas as pd

from src.data.validation_rules import (
    REQUIRED_COLUMNS,
    get_missing_columns,
    count_basic_quality_issues,
    count_cash_out_direction_mismatches,
)


def make_sample():
    return pd.DataFrame({
        "step": [1, 2, 0],
        "type": ["PAYMENT", "CASH_OUT", "CASH_OUT"],
        "amount": [100.0, 0.0, -5.0],
        "nameOrig": ["A", "B", "C"],
        "oldbalanceOrg": [500.0, 200.0, -1.0],
        "newbalanceOrig": [400.0, 250.0, 10.0],
        "nameDest": ["X", "Y", "Z"],
        "oldbalanceDest": [0.0, 10.0, 20.0],
        "newbalanceDest": [100.0, 10.0, 15.0],
        "isFraud": [0, 1, 2],
        "isFlaggedFraud": [0, 0, 3],
    })


def test_required_columns_are_present():
    assert get_missing_columns(REQUIRED_COLUMNS) == []


def test_missing_columns_are_detected():
    columns = REQUIRED_COLUMNS - {"amount"}
    assert get_missing_columns(columns) == ["amount"]


def test_basic_quality_issue_counts():
    result = count_basic_quality_issues(make_sample())

    assert result["zero_amount_rows"] == 1
    assert result["negative_amount_rows"] == 1
    assert result["negative_balance_rows"] == 1
    assert result["invalid_step_rows"] == 1
    assert result["invalid_fraud_label_rows"] == 1
    assert result["invalid_flag_label_rows"] == 1


def test_cash_out_direction_mismatch():
    df = make_sample()

    # Only the second row is CASH_OUT with a positive
    # sender balance change and positive transaction amount.
    df.loc[1, "amount"] = 50.0

    assert count_cash_out_direction_mismatches(df) == 1


def test_normal_cash_out_is_not_flagged():
    df = make_sample().iloc[[1]].copy()
    df["amount"] = 50.0
    df["oldbalanceOrg"] = 200.0
    df["newbalanceOrig"] = 150.0

    assert count_cash_out_direction_mismatches(df) == 0
