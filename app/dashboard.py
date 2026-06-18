import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from config.config import RAW_DATA_PATH, MODEL_PATH, FEATURE_COLUMNS
from src.preprocessing import load_dataset
from src.predict import predict_defect
from src.logger import log_prediction
from src.monitor import load_metadata, load_logs, monitoring_summary


st.set_page_config(
    page_title="QA Sentinel AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
#MainMenu, footer, header, [data-testid="stSidebar"] {display:none !important;}

.stApp {
    background: #f6f8fb !important;
    color: #111827 !important;
}

.block-container {
    max-width: 1520px !important;
    padding: 18px 30px 36px 30px !important;
}

* {
    font-family: Arial, Helvetica, sans-serif !important;
}

.qa-header {
    background: linear-gradient(135deg, #101828 0%, #1d2939 58%, #2563eb 100%);
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    gap: 20px;
    align-items: center;
    box-shadow: 0 16px 36px rgba(16, 24, 40, 0.18);
}

.qa-title {
    color: #ffffff !important;
    font-size: 30px;
    font-weight: 900;
    margin: 0 0 8px 0;
    line-height: 1.15;
}

.qa-subtitle {
    color: #dbeafe !important;
    font-size: 15px;
    line-height: 1.55;
    margin: 0;
}

.qa-logo {
    width: 62px;
    height: 62px;
    border-radius: 16px;
    background: #f59e0b;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size: 34px;
    box-shadow: 0 10px 25px rgba(245, 158, 11, 0.35);
    flex: 0 0 auto;
}

.qa-header-left {
    display:flex;
    align-items:center;
    gap: 18px;
}

.qa-pills {
    display:flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content:flex-end;
}

.qa-pill {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.28);
    color: #ffffff !important;
    border-radius: 999px;
    padding: 10px 14px;
    font-size: 13px;
    font-weight: 800;
    white-space: nowrap;
}

.nav-shell {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 10px 12px;
    margin-bottom: 18px;
    box-shadow: 0 8px 22px rgba(16,24,40,.06);
}

div[data-testid="stRadio"] {margin:0 !important;}

div[role="radiogroup"] {
    display:flex;
    gap: 10px;
    flex-wrap: wrap;
}

div[role="radiogroup"] label {
    background: #eef4ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 12px !important;
    color: #1e3a8a !important;
    min-width: 170px;
    height: 44px;
    padding: 8px 14px !important;
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    font-size: 14px !important;
    font-weight: 800 !important;
}

div[role="radiogroup"] label * {
    color: #1e3a8a !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}

div[role="radiogroup"] label:hover {
    background: #dbeafe !important;
    border-color: #2563eb !important;
}

.kpi-card {
    background:#ffffff;
    border:1px solid #e5e7eb;
    border-radius:16px;
    padding:20px 22px;
    min-height:128px;
    box-shadow:0 10px 26px rgba(16,24,40,.07);
    border-top:5px solid #2563eb;
}

.kpi-label {
    color:#667085 !important;
    font-size:12px;
    font-weight:900;
    letter-spacing:.08em;
    text-transform:uppercase;
    margin-bottom:16px;
}

.kpi-value {
    color:#111827 !important;
    font-size:32px;
    font-weight:900;
    line-height:1.05;
    margin-bottom:8px;
}

.kpi-note {
    color:#475467 !important;
    font-size:14px;
    line-height:1.45;
}

.panel {
    background:#ffffff;
    border:1px solid #e5e7eb;
    border-radius:16px;
    padding:22px 24px;
    box-shadow:0 10px 26px rgba(16,24,40,.07);
    margin-bottom:18px;
}

.panel h2 {
    color:#111827 !important;
    font-size:24px;
    font-weight:900;
    line-height:1.2;
    margin:0 0 10px 0;
}

.panel p, .panel li {
    color:#344054 !important;
    font-size:15px;
    line-height:1.65;
}

.section-title {
    color:#111827 !important;
    font-size:20px;
    font-weight:900;
    margin: 8px 0 12px 0;
}

.blue {color:#2563eb !important;}
.red {color:#dc2626 !important;}
.green {color:#16a34a !important;}
.orange {color:#f59e0b !important;}

div[data-testid="stPlotlyChart"] {
    background:#ffffff;
    border:1px solid #e5e7eb;
    border-radius:16px;
    padding:8px;
    box-shadow:0 8px 22px rgba(16,24,40,.05);
}

.stButton button, .stDownloadButton button, .stFormSubmitButton button {
    height:48px;
    border-radius:12px !important;
    background:#2563eb !important;
    color:#ffffff !important;
    border:none !important;
    font-size:15px !important;
    font-weight:900 !important;
}

label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color:#111827 !important;
    font-weight:800 !important;
}

.workflow {
    display:grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}

.step {
    background:#ffffff;
    border:1px solid #e5e7eb;
    border-left:5px solid #f59e0b;
    border-radius:14px;
    padding:16px;
    min-height:110px;
}

.step-num {
    background:#2563eb;
    color:#ffffff !important;
    width:28px;
    height:28px;
    border-radius:9px;
    display:flex;
    justify-content:center;
    align-items:center;
    font-weight:900;
    margin-bottom:10px;
}

.step-title {
    color:#111827 !important;
    font-size:15px;
    font-weight:900;
    margin-bottom:6px;
}

.step-text {
    color:#475467 !important;
    font-size:13px;
    line-height:1.45;
}

@media(max-width:1366px){
    .block-container {padding:14px 22px 28px 22px !important;}
    .qa-title {font-size:26px;}
    .qa-subtitle {font-size:14px;}
    .qa-logo {width:54px;height:54px;font-size:30px;}
    .qa-pill {font-size:12px;padding:8px 10px;}
    div[role="radiogroup"] label {min-width:145px;height:40px;font-size:13px !important;}
    .kpi-card {min-height:116px;padding:17px 18px;}
    .kpi-value {font-size:27px;}
    .panel {padding:19px 20px;}
    .workflow {grid-template-columns: repeat(2, 1fr);}
}
</style>
""", unsafe_allow_html=True)


df = load_dataset(RAW_DATA_PATH)
metadata = load_metadata()
logs = load_logs()
summary = monitoring_summary()
model_ready = MODEL_PATH.exists()

total_records = len(df)
defective = int(df["defects"].sum())
non_defective = int(total_records - defective)
defect_rate = round((defective / total_records) * 100, 2)
best_model = metadata.get("best_model", "Not trained")
accuracy = metadata.get("metrics", {}).get("accuracy", "N/A")


st.markdown(f"""
<div class="qa-header">
    <div class="qa-header-left">
        <div class="qa-logo">🧪</div>
        <div>
            <div class="qa-title">QA Sentinel AI Test Automation Framework</div>
            <p class="qa-subtitle">
                AI-driven software defect prediction, test risk scoring, intelligent QA prioritization,
                automated monitoring, and software quality analytics.
            </p>
        </div>
    </div>
    <div class="qa-pills">
        <div class="qa-pill">Dataset: {total_records:,} Modules</div>
        <div class="qa-pill">Model: {"Ready" if model_ready else "Train Required"}</div>
        <div class="qa-pill">Target: Defects</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="nav-shell">', unsafe_allow_html=True)
page = st.radio(
    "Navigation",
    [
        "🧭 QA Overview",
        "🤖 Defect Predictor",
        "📊 Quality Analytics",
        "🧪 Test Prioritization",
        "📈 Monitoring",
        "📘 Project Guide"
    ],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)


if page == "🧭 QA Overview":
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Software Modules</div><div class="kpi-value">{total_records:,}</div><div class="kpi-note">NASA JM1 module records</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Defective Modules</div><div class="kpi-value red">{defective:,}</div><div class="kpi-note">Modules marked as defective</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Defect Rate</div><div class="kpi-value orange">{defect_rate}%</div><div class="kpi-note">Observed defect ratio</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Model Accuracy</div><div class="kpi-value blue">{accuracy}</div><div class="kpi-note">Best model: {best_model}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.25, 0.95], gap="large")
    with left:
        st.markdown('<div class="panel"><h2>Software Defect Distribution</h2><p>This chart shows defective and non-defective software modules in the dataset.</p></div>', unsafe_allow_html=True)
        dist = pd.DataFrame({"Class": ["Non-Defective", "Defective"], "Count": [non_defective, defective]})
        fig = px.bar(dist, x="Class", y="Count", text="Count", color="Class", height=420,
                     color_discrete_map={"Non-Defective": "#16a34a", "Defective": "#dc2626"})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("""
        <div class="panel">
            <h2>QA Automation Purpose</h2>
            <p>
                The system predicts risky software modules before release. QA teams can use this output to
                prioritize regression tests, code reviews, unit tests, integration tests, and release checks.
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("View Model Metadata", expanded=False):
            st.json(metadata if metadata else {"status": "Train model first"})


elif page == "🤖 Defect Predictor":
    st.markdown("""
    <div class="panel">
        <h2>Real-Time Defect Risk Predictor</h2>
        <p>Enter static code metrics to predict whether a software module is defective and what testing priority it requires.</p>
    </div>
    """, unsafe_allow_html=True)

    form_area, guide = st.columns([1.45, 0.8], gap="large")
    with form_area:
        with st.form("defect_form"):
            st.markdown('<div class="section-title">Code Size & Complexity Metrics</div>', unsafe_allow_html=True)
            a1, a2, a3, a4 = st.columns(4)
            with a1: loc = st.number_input("LOC", 0.0, 2000.0, 42.0)
            with a2: vg = st.number_input("Cyclomatic Complexity v(g)", 0.0, 200.0, 5.0)
            with a3: evg = st.number_input("Essential Complexity ev(g)", 0.0, 200.0, 2.0)
            with a4: ivg = st.number_input("Design Complexity iv(g)", 0.0, 200.0, 4.0)

            st.markdown('<div class="section-title">Halstead Software Metrics</div>', unsafe_allow_html=True)
            b1, b2, b3, b4 = st.columns(4)
            with b1: n = st.number_input("Halstead Length n", 0.0, 5000.0, 120.0)
            with b2: v = st.number_input("Halstead Volume v", 0.0, 50000.0, 700.0)
            with b3: l = st.number_input("Program Level l", 0.0, 5.0, 0.08)
            with b4: d = st.number_input("Difficulty d", 0.0, 300.0, 15.0)

            b5, b6, b7, b8 = st.columns(4)
            with b5: i = st.number_input("Intelligence i", 0.0, 1000.0, 45.0)
            with b6: e = st.number_input("Effort e", 0.0, 500000.0, 12000.0)
            with b7: b = st.number_input("Estimated Bugs b", 0.0, 20.0, 0.25)
            with b8: t = st.number_input("Time t", 0.0, 50000.0, 650.0)

            st.markdown('<div class="section-title">Operator, Operand & Branch Metrics</div>', unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: lOCode = st.number_input("Lines of Code", 0, 2000, 30)
            with c2: lOComment = st.number_input("Comment Lines", 0, 500, 4)
            with c3: lOBlank = st.number_input("Blank Lines", 0, 500, 5)
            with c4: locCodeAndComment = st.number_input("Code + Comment", 0, 300, 1)
            with c5: branchCount = st.number_input("Branch Count", 0.0, 500.0, 9.0)

            d1, d2, d3, d4 = st.columns(4)
            with d1: uniq_Op = st.number_input("Unique Operators", 0.0, 500.0, 12.0)
            with d2: uniq_Opnd = st.number_input("Unique Operands", 0.0, 1000.0, 28.0)
            with d3: total_Op = st.number_input("Total Operators", 0.0, 5000.0, 70.0)
            with d4: total_Opnd = st.number_input("Total Operands", 0.0, 5000.0, 50.0)

            submitted = st.form_submit_button("Analyze Defect Risk")

    with guide:
        st.markdown("""
        <div class="panel">
            <h2>Risk Guide</h2>
            <p>
                <b>High Risk:</b> Run full regression testing and manual review.<br><br>
                <b>Medium Risk:</b> Add targeted unit and integration tests.<br><br>
                <b>Low Risk:</b> Standard QA testing is acceptable.
            </p>
        </div>
        """, unsafe_allow_html=True)

    if submitted:
        if not model_ready:
            st.error("Model not trained. Run: python src\\train_model.py")
        else:
            payload = {
                "loc": loc, "v(g)": vg, "ev(g)": evg, "iv(g)": ivg,
                "n": n, "v": v, "l": l, "d": d, "i": i, "e": e, "b": b, "t": t,
                "lOCode": lOCode, "lOComment": lOComment, "lOBlank": lOBlank,
                "locCodeAndComment": locCodeAndComment,
                "uniq_Op": uniq_Op, "uniq_Opnd": uniq_Opnd,
                "total_Op": total_Op, "total_Opnd": total_Opnd,
                "branchCount": branchCount
            }
            result = predict_defect(payload)
            log_prediction(payload, result)
            risk_class = "green" if result["risk_level"] == "Low" else "orange" if result["risk_level"] == "Medium" else "red"

            r1, r2, r3 = st.columns(3, gap="large")
            with r1:
                st.markdown(f'<div class="kpi-card"><div class="kpi-label">Prediction</div><div class="kpi-value {risk_class}">{result["defect_prediction"]}</div><div class="kpi-note">Model decision</div></div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="kpi-card"><div class="kpi-label">Defect Probability</div><div class="kpi-value">{result["defect_probability"]*100:.2f}%</div><div class="kpi-note">Risk probability</div></div>', unsafe_allow_html=True)
            with r3:
                st.markdown(f'<div class="kpi-card"><div class="kpi-label">Testing Priority</div><div class="kpi-value {risk_class}">{result["risk_level"]}</div><div class="kpi-note">QA priority level</div></div>', unsafe_allow_html=True)

            st.success(result["recommendation"])


elif page == "📊 Quality Analytics":
    st.markdown('<div class="panel"><h2>Software Quality Analytics</h2><p>Analyze code complexity, size, and defect relationships using visual quality engineering insights.</p></div>', unsafe_allow_html=True)

    a, b = st.columns(2, gap="large")
    with a:
        fig = px.box(df, x="defects", y="loc", color="defects", height=430, title="LOC Distribution by Defect Class",
                     color_discrete_map={0:"#16a34a", 1:"#dc2626"})
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig2 = px.scatter(df.sample(min(1600, len(df)), random_state=4), x="v(g)", y="loc", color="defects",
                          height=430, title="Complexity vs Lines of Code",
                          color_discrete_map={0:"#16a34a", 1:"#dc2626"})
        st.plotly_chart(fig2, use_container_width=True)

    corr = df[FEATURE_COLUMNS + ["defects"]].corr(numeric_only=True)
    fig3 = px.imshow(corr, text_auto=False, height=560, color_continuous_scale="Blues", title="Feature Correlation Heatmap")
    st.plotly_chart(fig3, use_container_width=True)


elif page == "🧪 Test Prioritization":
    st.markdown("""
    <div class="panel">
        <h2>AI-Based Test Prioritization</h2>
        <p>
            The framework converts defect risk into QA actions. High-risk modules should receive earlier testing,
            deeper review, and stronger automation coverage.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sample = df.sample(min(500, len(df)), random_state=7).copy()
    sample["priority_score"] = (
        sample["loc"].rank(pct=True) * 0.25 +
        sample["v(g)"].rank(pct=True) * 0.25 +
        sample["e"].rank(pct=True) * 0.20 +
        sample["branchCount"].rank(pct=True) * 0.20 +
        sample["defects"].astype(int) * 0.10
    )
    sample["Test Priority"] = pd.cut(sample["priority_score"], bins=[0, .45, .70, 1.1], labels=["Low", "Medium", "High"])
    sample_show = sample[["loc", "v(g)", "e", "branchCount", "defects", "priority_score", "Test Priority"]].sort_values("priority_score", ascending=False).head(20)

    c1, c2 = st.columns([1.2, 0.8], gap="large")
    with c1:
        st.dataframe(sample_show, use_container_width=True, height=460)
    with c2:
        fig = px.histogram(sample, x="Test Priority", color="Test Priority", height=460,
                           color_discrete_map={"Low":"#16a34a","Medium":"#f59e0b","High":"#dc2626"})
        st.plotly_chart(fig, use_container_width=True)


elif page == "📈 Monitoring":
    st.markdown('<div class="panel"><h2>QA Prediction Monitoring</h2><p>Track dashboard predictions, risk trends, and software quality decisions.</p></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Predictions</div><div class="kpi-value">{summary["total_predictions"]}</div><div class="kpi-note">Logged QA analyses</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">High Risk Modules</div><div class="kpi-value red">{summary["high_risk_modules"]}</div><div class="kpi-note">Critical test targets</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Defect Probability</div><div class="kpi-value orange">{summary["average_defect_probability"]}%</div><div class="kpi-note">Mean risk score</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Last Prediction</div><div class="kpi-note">{summary["last_prediction_time"]}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if not logs.empty:
        a, b = st.columns(2, gap="large")
        with a:
            fig = px.histogram(logs, x="risk_level", color="risk_level", height=390,
                               color_discrete_map={"Low":"#16a34a","Medium":"#f59e0b","High":"#dc2626"})
            st.plotly_chart(fig, use_container_width=True)
        with b:
            logs = logs.copy()
            logs["request_no"] = range(1, len(logs)+1)
            fig2 = px.line(logs, x="request_no", y="defect_probability", height=390, title="Defect Probability Trend")
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(logs, use_container_width=True, height=340)
        st.download_button("Download QA Monitoring Logs", logs.to_csv(index=False).encode("utf-8"), "qa_prediction_logs.csv", "text/csv")
    else:
        st.info("No prediction logs yet. Use the Defect Predictor page first.")


elif page == "📘 Project Guide":
    st.markdown("""
    <div class="panel">
        <h2>Project Workflow</h2>
        <p>
            This project demonstrates how AI supports software quality assurance by predicting defective modules
            and prioritizing automated testing before software release.
        </p>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("Dataset Loading", "NASA JM1 software metrics dataset is loaded."),
        ("Data Cleaning", "Invalid object values are converted into numeric software metrics."),
        ("Feature Engineering", "Code complexity, Halstead, branch, and LOC metrics are prepared."),
        ("Model Training", "Multiple ML models are trained and compared automatically."),
        ("Model Selection", "The best model is selected using F1-score and ROC-AUC."),
        ("Defect Prediction", "The system predicts defective or non-defective modules."),
        ("Test Prioritization", "Risk levels guide automated testing order and QA effort."),
        ("Monitoring", "Prediction logs are stored for QA tracking and reporting.")
    ]

    st.markdown('<div class="workflow">', unsafe_allow_html=True)
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f'<div class="step"><div class="step-num">{i}</div><div class="step-title">{title}</div><div class="step-text">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div class="panel">
        <h2>Why This Project Helps</h2>
        <p>
            Manual QA cannot test every module with equal depth when release time is limited.
            This system identifies risky software modules early, helps QA teams prioritize testing,
            reduces defect leakage, improves release confidence, and supports intelligent test automation.
        </p>
    </div>
    """, unsafe_allow_html=True)
