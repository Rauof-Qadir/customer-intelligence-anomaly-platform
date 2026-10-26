
import pandas as pd


REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}


def get_missing_columns(columns):
    """Return required columns that are absent."""
    return sorted(REQUIRED_COLUMNS - set(columns))


def count_basic_quality_issues(df):
    """Count basic quality issues in a DataFrame."""

    balance_columns = [
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    return {
        "zero_amount_rows": int(df["amount"].eq(0).sum()),
        "negative_amount_rows": int(df["amount"].lt(0).sum()),
        "negative_balance_rows": int(
            df[balance_columns].lt(0).any(axis=1).sum()
        ),
        "invalid_step_rows": int(
            (df["step"].isna() | df["step"].le(0)).sum()
        ),
        "invalid_fraud_label_rows": int(
            (~df["isFraud"].isin([0, 1])).sum()
        ),
        "invalid_flag_label_rows": int(
            (~df["isFlaggedFraud"].isin([0, 1])).sum()
        ),
    }


def count_cash_out_direction_mismatches(df, tolerance=0.01):
    """Flag CASH_OUT rows where a positive amount accompanies
    an increase in the sender's balance.
    These are investigation candidates, not confirmed errors.
    """

    delta = df["newbalanceOrig"] - df["oldbalanceOrg"]

    mask = (
        df["type"].eq("CASH_OUT")
        & df["amount"].gt(0)
        & (delta > tolerance)
    )

    return int(mask.sum())
