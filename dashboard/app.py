
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
    initial_sidebar_state="expanded",
)

# ============================================================
# PROFESSIONAL UI THEME — PRESENTATION ONLY
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #10243A;
        --navy-light: #193750;
        --teal: #14B8A6;
        --teal-light: #CCFBF1;
        --page-bg: #F4F7FB;
        --card-bg: #FFFFFF;
        --text-main: #182B42;
        --text-muted: #64748B;
        --border: #E2E8F0;
    }

    .stApp {
        background: var(--page-bg);
        color: var(--text-main);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, 0.92);
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1600px;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--text-main) !important;
        letter-spacing: -0.035em;
    }

    h1 {
        font-weight: 800 !important;
    }

    h2, h3 {
        font-weight: 700 !important;
    }

    p, label, [data-testid="stMarkdownContainer"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(120deg, #10243A 0%, #193750 70%, #145B67 100%);
        padding: 30px 34px;
        border-radius: 20px;
        margin-bottom: 22px;
        box-shadow: 0 12px 30px rgba(16, 36, 58, 0.12);
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 230px;
        height: 230px;
        border-radius: 50%;
        right: -55px;
        top: -100px;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow:
            0 0 0 30px rgba(255,255,255,0.025),
            0 0 0 60px rgba(255,255,255,0.02);
    }

    .hero-eyebrow {
        color: #5EEAD4;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.17em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        color: #FFFFFF;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(25px, 3vw, 38px);
        font-weight: 800;
        line-height: 1.2;
        letter-spacing: -0.045em;
        margin-bottom: 10px;
        position: relative;
        z-index: 1;
    }

    .hero-description {
        color: #CBD9E7;
        font-size: 14px;
        line-height: 1.7;
        max-width: 760px;
        margin-bottom: 20px;
        position: relative;
        z-index: 1;
    }

    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        position: relative;
        z-index: 1;
    }

    .hero-tag {
        color: #E2F8F5;
        background: rgba(255,255,255,0.09);
        border: 1px solid rgba(255,255,255,0.14);
        padding: 6px 11px;
        border-radius: 7px;
        font-size: 11px;
        font-weight: 600;
    }

    /* Section headings */
    .section-heading {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 12px;
        margin-bottom: 5px;
        color: var(--text-main);
        font-family: 'Manrope', sans-serif;
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.035em;
    }

    .section-dot {
        width: 9px;
        height: 9px;
        background: var(--teal);
        border-radius: 3px;
        display: inline-block;
        flex-shrink: 0;
    }

    .section-subtitle {
        color: var(--text-muted);
        font-size: 12px;
        margin-bottom: 17px;
        line-height: 1.6;
    }

    /* KPI cards */
    [data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 19px 20px;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.025);
        min-height: 112px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: #99DCD4;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.055);
    }

    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #10243A !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: clamp(22px, 2vw, 30px) !important;
        font-weight: 800 !important;
        letter-spacing: -0.04em;
    }

    [data-testid="stMetricDelta"] {
        font-size: 11px !important;
    }

    /* Chart and content surfaces */
    [data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 9px 8px 0;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.025);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 13px;
        overflow: hidden;
        background: #FFFFFF;
    }

    [data-testid="stDownloadButton"] button {
        background: var(--navy);
        color: #FFFFFF;
        border: 1px solid var(--navy);
        border-radius: 9px;
        padding: 9px 17px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    [data-testid="stDownloadButton"] button:hover {
        background: #1D435D;
        color: #FFFFFF;
        border-color: #1D435D;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 17px !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #64748B;
        font-size: 12px;
        line-height: 1.6;
    }

    [data-testid="stSidebar"] [data-testid="stMultiSelect"] > div,
    [data-testid="stSidebar"] [data-testid="stSlider"] {
        margin-bottom: 12px;
    }

    /* Supporting notes */
    .info-strip {
        display: flex;
        gap: 11px;
        align-items: flex-start;
        background: #ECFDF9;
        border: 1px solid #B8EFE5;
        border-radius: 12px;
        padding: 13px 16px;
        margin: 12px 0 18px;
        color: #115E59;
        font-size: 12px;
        line-height: 1.7;
    }

    .info-strip strong {
        color: #134E4A;
    }

    .table-meta {
        color: #64748B;
        font-size: 12px;
        margin: 2px 0 13px;
    }

    .footer-note {
        border-top: 1px solid #DCE5EE;
        margin-top: 28px;
        padding-top: 18px;
        color: #64748B;
        font-size: 11px;
        line-height: 1.8;
    }

    /* Keep the interface usable on narrow screens */
    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 0.8rem 2rem;
        }

        .hero {
            padding: 23px 20px;
            border-radius: 15px;
        }

        [data-testid="stMetric"] {
            padding: 13px;
            min-height: 96px;
        }
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING — ORIGINAL LOGIC PRESERVED
# ============================================================

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

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">FINANCIAL RISK INTELLIGENCE · ANALYTICS</div>
        <div class="hero-title">Customer Intelligence<br> &amp; Anomaly Discovery</div>
        <div class="hero-description">
            Explore unusual transaction patterns, monitor model alerts,
            analyze transaction behavior, and prioritize records for analyst review.
        </div>
        <div class="hero-tags">
            <span class="hero-tag">Isolation Forest</span>
            <span class="hero-tag">Anomaly Detection</span>
            <span class="hero-tag">Risk Analytics</span>
            <span class="hero-tag">Synthetic PaySim-style Data</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR FILTERS — ORIGINAL FILTER LOGIC PRESERVED
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding: 4px 0 14px;">
        <div style="font-size:10px;letter-spacing:.15em;font-weight:800;
                    color:#0D9488;margin-bottom:7px;">
            ANALYST WORKSPACE
        </div>
        <div style="font-family:Manrope,sans-serif;font-size:22px;
                    font-weight:800;color:#10243A;letter-spacing:-.04em;">
            Review Controls
        </div>
        <div style="font-size:12px;color:#64748B;margin-top:5px;">
            Filter the transaction review queue.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

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

st.sidebar.markdown(
    f"""
    <div style="background:#F1F5F9;border:1px solid #E2E8F0;
                border-radius:10px;padding:12px;margin-top:15px;">
        <div style="font-size:10px;color:#64748B;font-weight:700;
                    letter-spacing:.08em;text-transform:uppercase;">
            Selected Threshold
        </div>
        <div style="font-family:Manrope,sans-serif;font-size:23px;
                    font-weight:800;color:#10243A;margin-top:3px;">
            {threshold:.5f}
        </div>
        <div style="font-size:11px;color:#64748B;margin-top:3px;">
            Validation-selected threshold
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

filtered = df[
    df["type"].isin(selected_types)
    & df["anomaly_score"].between(score_range[0], score_range[1])
].copy()

filtered_alerts = filtered[filtered["is_alert"] == 1]

# ============================================================
# KPI OVERVIEW — SAME METRICS, IMPROVED PRESENTATION
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-dot"></span>
        Risk Performance Overview
    </div>
    <div class="section-subtitle">
        Model evaluation metrics from the held-out test period.
    </div>
    """,
    unsafe_allow_html=True,
)

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

st.markdown(
    """
    <div class="info-strip">
        <span>ⓘ</span>
        <div>
            <strong>How to read these metrics:</strong>
            Precision describes the share of flagged records that are fraud labels;
            recall describes the share of fraud labels detected. Average precision
            summarizes precision-recall performance across score thresholds.
            These are evaluation metrics, not guarantees of future performance.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ============================================================
# ANOMALY OVERVIEW — SAME CHARTS, POLISHED VISUALS
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-dot"></span>
        Anomaly Overview
    </div>
    <div class="section-subtitle">
        Understand score patterns and how alert rates vary across transaction types.
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")

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
        color_discrete_map={
            0: "#A9B8CA",
            1: "#14B8A6",
            "0": "#A9B8CA",
            "1": "#14B8A6",
        },
        category_orders={"is_alert": [0, 1]},
        template="plotly_white",
        opacity=0.88,
    )

    score_fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#E07A35",
        line_width=2,
        annotation_text=f"Threshold: {threshold:.3f}",
        annotation_position="top right",
        annotation_font_color="#B45309",
    )

    score_fig.update_layout(
        height=390,
        margin=dict(l=18, r=18, t=70, b=20),
        font=dict(family="DM Sans, sans-serif", color="#475569", size=11),
        title=dict(font=dict(size=15, color="#10243A"), x=0.04, xanchor="left"),
        legend=dict(
            title_text="Flagged alert",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        bargap=0.06,
    )
    score_fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        title_font=dict(size=11),
    )
    score_fig.update_yaxes(
        gridcolor="#EDF2F7",
        zeroline=False,
        title_font=dict(size=11),
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
        template="plotly_white",
        color="alert_rate",
        color_continuous_scale=["#C7D9E8", "#5EC9BC", "#0F766E"],
    )

    type_fig.update_layout(
        height=390,
        margin=dict(l=18, r=18, t=70, b=20),
        font=dict(family="DM Sans, sans-serif", color="#475569", size=11),
        title=dict(font=dict(size=15, color="#10243A"), x=0.04, xanchor="left"),
        coloraxis_showscale=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    type_fig.update_traces(
        marker_line_width=0,
        hovertemplate=(
            "Type: %{x}<br>"
            "Alert rate: %{y:.2%}<extra></extra>"
        ),
    )
    type_fig.update_yaxes(
        tickformat=".1%",
        gridcolor="#EDF2F7",
        zeroline=False,
        title_font=dict(size=11),
    )
    type_fig.update_xaxes(
        showgrid=False,
        title_font=dict(size=11),
    )

    st.plotly_chart(type_fig, use_container_width=True)

# ============================================================
# TRANSACTION AMOUNT ANALYSIS — ORIGINAL DATA AND PURPOSE
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-dot"></span>
        Transaction Amount Analysis
    </div>
    <div class="section-subtitle">
        Compare the distribution of transaction amounts among flagged alerts.
    </div>
    """,
    unsafe_allow_html=True,
)

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
    template="plotly_white",
    color="type",
    color_discrete_sequence=[
        "#0F766E",
        "#2878A5",
        "#6D79B8",
        "#D18A42",
        "#5B8F75",
    ],
)

amount_fig.update_layout(
    height=430,
    margin=dict(l=20, r=20, t=70, b=25),
    font=dict(family="DM Sans, sans-serif", color="#475569", size=11),
    title=dict(font=dict(size=15, color="#10243A"), x=0.025, xanchor="left"),
    showlegend=False,
    paper_bgcolor="white",
    plot_bgcolor="white",
)
amount_fig.update_traces(
    marker_size=4,
    line_width=1.5,
    hovertemplate=(
        "Type: %{x}<br>"
        "Amount: %{y:,.2f}<extra></extra>"
    ),
)
amount_fig.update_xaxes(showgrid=False, title_font=dict(size=11))
amount_fig.update_yaxes(
    gridcolor="#EDF2F7",
    zeroline=False,
    tickformat=",.0f",
    title_font=dict(size=11),
)

st.plotly_chart(amount_fig, use_container_width=True)

st.divider()

# ============================================================
# ANALYST REVIEW QUEUE — SAME FILTERS AND DISPLAY COLUMNS
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-dot"></span>
        Analyst Review Queue
    </div>
    <div class="section-subtitle">
        Review flagged transactions matching your selected transaction types
        and anomaly-score range.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="table-meta">
        Showing <strong>{len(filtered_alerts):,}</strong> flagged transactions
        matching the selected filters.
    </div>
    """,
    unsafe_allow_html=True,
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
    height=420,
    column_config={
        "step": st.column_config.NumberColumn(
            "Time Step",
            format="%d",
        ),
        "type": st.column_config.TextColumn(
            "Transaction Type",
        ),
        "amount": st.column_config.NumberColumn(
            "Transaction Amount",
            format="%,.2f",
        ),
        "oldbalanceOrg": st.column_config.NumberColumn(
            "Sender Balance Before",
            format="%,.2f",
        ),
        "oldbalanceDest": st.column_config.NumberColumn(
            "Receiver Balance Before",
            format="%,.2f",
        ),
        "anomaly_score": st.column_config.NumberColumn(
            "Anomaly Score",
            format="%.5f",
        ),
        "review_priority": st.column_config.TextColumn(
            "Review Priority",
        ),
    },
)

csv_data = review_table.to_csv(index=False).encode("utf-8")

download_col, download_note_col = st.columns([1, 2])

with download_col:
    st.download_button(
        "↓  Export Filtered Alerts as CSV",
        data=csv_data,
        file_name="anomaly_review_alerts.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_note_col:
    st.caption(
        "Export includes the currently filtered alert rows and the same "
        "review columns displayed in the table."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        <strong style="color:#334155;">CUSTOMER INTELLIGENCE · RISK ANALYTICS</strong>
        <br>
        Important: Anomaly scores indicate unusual transaction patterns,
        not confirmed fraud. Metrics are from the held-out test period.
        The dataset uses synthetic time steps, not actual timestamps.
        <br><br>
        Built for analytical exploration and portfolio demonstration.
        Not intended for autonomous financial decisions or production use
        without additional validation.
    </div>
    """,
    unsafe_allow_html=True,
)
