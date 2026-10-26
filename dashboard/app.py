
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
QUEUE_FILE = ROOT / "data" / "processed" / "anomaly_review_queue.parquet"
REPORT_FILE = ROOT / "reports" / "threshold_analysis.json"

st.set_page_config(
    page_title="Customer Intelligence | Risk Dashboard",
    page_icon="🔎",
    layout="wide",
)

st.title("Customer Intelligence & Anomaly Discovery")
st.caption(
    "Transaction anomaly monitoring | Isolation Forest | "
    "Synthetic PaySim-style dataset"
)

if not QUEUE_FILE.exists() or not REPORT_FILE.exists():
    st.error(
        "Required output files are missing. "
        "Run the threshold evaluation script first."
    )
    st.stop()

df = pd.read_parquet(QUEUE_FILE)

with REPORT_FILE.open("r", encoding="utf-8") as file:
    report = json.load(file)

metrics = report["final_test_metrics"]
threshold = report["selected_threshold"]

alerts = df[df["is_alert"] == 1].copy()

# Sidebar filters
st.sidebar.header("Review Filters")

transaction_types = sorted(df["type"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect(
    "Transaction type",
    transaction_types,
    default=transaction_types,
)

min_score = float(df["anomaly_score"].min())
max_score = float(df["anomaly_score"].max())

score_range = st.sidebar.slider(
    "Anomaly score range",
    min_value=min_score,
    max_value=max_score,
    value=(threshold, max_score),
    format="%.3f",
)

filtered = df[
    df["type"].isin(selected_types)
    & df["anomaly_score"].between(score_range[0], score_range[1])
].copy()

filtered_alerts = filtered[filtered["is_alert"] == 1]

# KPI cards
c1, c2, c3, c4 = st.columns(4)

c1.metric("Test Transactions", f"{len(df):,}")
c2.metric("Flagged Alerts", f"{int(df['is_alert'].sum()):,}")
c3.metric("Test Alert Rate", f"{metrics['alert_rate']:.2%}")
c4.metric("Average Precision", f"{metrics['average_precision']:.3f}")

c5, c6, c7, c8 = st.columns(4)

c5.metric("Precision", f"{metrics['precision']:.2%}")
c6.metric("Recall", f"{metrics['recall']:.2%}")
c7.metric("Detected Fraud Labels", f"{metrics['true_positives']:,}")
c8.metric("Missed Fraud Labels", f"{metrics['false_negatives']:,}")

st.divider()

st.subheader("Anomaly Overview")

left, right = st.columns(2)

with left:
    score_fig = px.histogram(
        df,
        x="anomaly_score",
        color="is_alert",
        nbins=60,
        title="Anomaly Score Distribution",
        labels={
            "anomaly_score": "Anomaly score",
            "is_alert": "Flagged alert",
        },
    )
    score_fig.add_vline(
        x=threshold,
        line_dash="dash",
        annotation_text=f"Threshold: {threshold:.3f}",
    )
    st.plotly_chart(score_fig, use_container_width=True)

with right:
    type_summary = (
        df.groupby("type", as_index=False)
        .agg(
            transactions=("type", "size"),
            alerts=("is_alert", "sum"),
        )
    )

    type_summary["alert_rate"] = (
        type_summary["alerts"] / type_summary["transactions"]
    )

    type_fig = px.bar(
        type_summary,
        x="type",
        y="alert_rate",
        title="Alert Rate by Transaction Type",
        labels={
            "type": "Transaction type",
            "alert_rate": "Alert rate",
        },
    )
    type_fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(type_fig, use_container_width=True)

st.subheader("Transaction Amount Analysis")

amount_fig = px.box(
    df[df["is_alert"] == 1],
    x="type",
    y="amount",
    title="Transaction Amounts Among Flagged Alerts",
    points=False,
    labels={
        "type": "Transaction type",
        "amount": "Amount",
    },
)
st.plotly_chart(amount_fig, use_container_width=True)

st.divider()

st.subheader("Analyst Review Queue")

st.write(
    f"Showing **{len(filtered_alerts):,} flagged transactions** "
    "matching the selected filters."
)

display_columns = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "anomaly_score",
    "review_priority",
]

review_table = filtered_alerts[display_columns].copy()

st.dataframe(
    review_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "amount": st.column_config.NumberColumn(
            "Transaction Amount", format="%,.2f"
        ),
        "oldbalanceOrg": st.column_config.NumberColumn(
            "Sender Balance Before", format="%,.2f"
        ),
        "oldbalanceDest": st.column_config.NumberColumn(
            "Receiver Balance Before", format="%,.2f"
        ),
        "anomaly_score": st.column_config.NumberColumn(
            "Anomaly Score", format="%.5f"
        ),
    },
)

csv_data = review_table.to_csv(index=False).encode("utf-8")

st.download_button(
    "Export Filtered Alerts as CSV",
    data=csv_data,
    file_name="anomaly_review_alerts.csv",
    mime="text/csv",
)

st.caption(
    "Important: Anomaly scores indicate unusual transaction patterns, "
    "not confirmed fraud. Metrics are from the held-out test period. "
    "The dataset uses synthetic time steps, not actual timestamps."
)
