from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ARTIFACT_NAMES = {
    "model": "fraud_model.pkl",
    "scaler": "scaler.pkl",
    "selector": "selector.pkl",
    "features": "features.pkl",
    "labels": "labels.pkl",
}
ARTIFACT_DIR_CANDIDATES = [
    APP_DIR,
    APP_DIR.parent / "Pattern",
    Path.cwd(),
]
DEFAULT_DATA_PATH = APP_DIR / "creditcard.csv"
TARGET_COLUMN = "Class"
FRAUD_LABEL = 1


st.set_page_config(
    page_title="FraudLens AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --page: #090712;
        --panel: rgba(20, 18, 34, 0.86);
        --panel-2: rgba(31, 27, 52, 0.74);
        --line: rgba(255, 255, 255, 0.11);
        --text: #fff7ff;
        --muted: #b8b0ca;
        --pink: #ff4fa3;
        --violet: #a855f7;
        --lavender: #d8b4fe;
        --red: #fb7185;
        --green: #34d399;
        --amber: #fbbf24;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(255, 79, 163, 0.24), transparent 24rem),
            radial-gradient(circle at 78% 0%, rgba(168, 85, 247, 0.20), transparent 24rem),
            linear-gradient(135deg, #090712 0%, #111020 52%, #080711 100%);
        color: var(--text);
    }

    .block-container {
        padding: 1.15rem 1.5rem 2rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121022 0%, #090814 100%);
        border-right: 1px solid var(--line);
    }

    .hero {
        min-height: 220px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.25rem;
        padding: 1.45rem;
        border: 1px solid var(--line);
        border-radius: 22px;
        background:
            linear-gradient(135deg, rgba(255, 79, 163, 0.18), rgba(168, 85, 247, 0.12)),
            rgba(14, 13, 25, 0.82);
        box-shadow: 0 26px 100px rgba(0, 0, 0, 0.40);
        overflow: hidden;
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2.1rem, 4.6vw, 4.5rem);
        line-height: 1;
        letter-spacing: 0;
    }

    .hero p {
        color: var(--muted);
        max-width: 760px;
        margin: 0.75rem 0 0;
        font-size: 1.02rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.45rem 0.7rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.13);
        background: rgba(255, 255, 255, 0.07);
        color: var(--lavender);
        font-weight: 800;
        font-size: 0.84rem;
        margin-bottom: 0.75rem;
    }

    .section-title {
        margin: 1.05rem 0 0.6rem;
        font-size: 1.2rem;
        font-weight: 900;
    }

    .glass-card {
        height: 100%;
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid var(--line);
        background: var(--panel);
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.25);
    }

    .mini-card {
        height: 100%;
        padding: 0.95rem;
        border-radius: 14px;
        border: 1px solid var(--line);
        background: var(--panel-2);
    }

    .mini-card b {
        color: white;
        font-size: 1rem;
    }

    .mini-card p {
        margin: 0.35rem 0 0;
        color: var(--muted);
        font-size: 0.91rem;
    }

    .result-high {
        padding: 1.1rem;
        border-radius: 16px;
        border: 1px solid rgba(251, 113, 133, 0.48);
        background: linear-gradient(135deg, rgba(251, 113, 133, 0.26), rgba(255, 79, 163, 0.11));
        color: #fff2f5;
        font-size: 1.35rem;
        font-weight: 900;
    }

    .result-low {
        padding: 1.1rem;
        border-radius: 16px;
        border: 1px solid rgba(52, 211, 153, 0.38);
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.21), rgba(168, 85, 247, 0.10));
        color: #f0fff9;
        font-size: 1.35rem;
        font-weight: 900;
    }

    div[data-testid="stMetric"] {
        padding: 1rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(20, 18, 34, 0.88);
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.22);
    }

    div[data-testid="stMetricValue"] {
        color: white;
        font-weight: 900;
    }

    div[data-testid="stMetricDelta"] {
        color: var(--lavender);
    }

    .stButton > button,
    .stDownloadButton > button {
        min-height: 2.65rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.17);
        background: linear-gradient(135deg, #ff4fa3 0%, #a855f7 100%);
        color: white;
        font-weight: 900;
        box-shadow: 0 14px 34px rgba(168, 85, 247, 0.25);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    @media (max-width: 860px) {
        .hero {
            min-height: 190px;
            align-items: flex-start;
            flex-direction: column;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def artifact_paths():
    for folder in ARTIFACT_DIR_CANDIDATES:
        paths = {key: folder / name for key, name in ARTIFACT_NAMES.items()}
        if all(path.exists() for path in paths.values()):
            return folder, paths, []

    paths = {key: APP_DIR / name for key, name in ARTIFACT_NAMES.items()}
    missing = [name for name in ARTIFACT_NAMES.values() if not (APP_DIR / name).exists()]
    return APP_DIR, paths, missing


@st.cache_resource(show_spinner=False)
def load_artifacts():
    folder, paths, missing = artifact_paths()
    if missing:
        return {
            "model": None,
            "scaler": None,
            "selector": None,
            "features": None,
            "labels": {0: "Normal", 1: "Fraud"},
            "folder": folder,
            "missing": missing,
            "error": None,
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(paths["model"])
            scaler = joblib.load(paths["scaler"])
            selector = joblib.load(paths["selector"])
            features = list(joblib.load(paths["features"]))
            labels = joblib.load(paths["labels"])
    except Exception as exc:
        return {
            "model": None,
            "scaler": None,
            "selector": None,
            "features": None,
            "labels": {0: "Normal", 1: "Fraud"},
            "folder": folder,
            "missing": [],
            "error": str(exc),
        }

    return {
        "model": model,
        "scaler": scaler,
        "selector": selector,
        "features": features,
        "labels": labels,
        "folder": folder,
        "missing": [],
        "error": None,
    }


@st.cache_data(show_spinner=False)
def load_dataset(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def read_uploaded_csv(uploaded_file):
    if uploaded_file is None:
        return None, None
    try:
        return pd.read_csv(uploaded_file), None
    except Exception as exc:
        return None, str(exc)


def plot_theme(fig, title=None):
    fig.update_layout(
        template="plotly_dark",
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fff7ff"),
        colorway=["#ff4fa3", "#a855f7", "#d8b4fe", "#34d399", "#fbbf24"],
        margin=dict(l=20, r=20, t=48, b=25),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)")
    return fig


def pct(value):
    return f"{value:.2%}" if pd.notna(value) else "N/A"


def money(value):
    return f"${value:,.2f}" if pd.notna(value) else "$0.00"


def model_metrics():
    return pd.DataFrame(
        {
            "Model": ["Random Forest", "XGBoost", "Logistic Regression"],
            "Accuracy": [0.999192, 0.998086, 0.986833],
            "Recall": [0.846939, 0.887755, 0.918367],
            "F1-score": [0.783019, 0.614841, 0.193548],
        }
    )


def alert(message, kind="info"):
    style = {
        "info": ("#a855f7", "rgba(168, 85, 247, 0.14)"),
        "success": ("#34d399", "rgba(52, 211, 153, 0.14)"),
        "warning": ("#fbbf24", "rgba(251, 191, 36, 0.14)"),
        "error": ("#fb7185", "rgba(251, 113, 133, 0.16)"),
    }[kind]
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid {style[0]}; background: {style[1]};">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def sample_for_charts(df, size):
    if df.empty:
        return df
    return df.sample(min(len(df), size), random_state=7) if len(df) > size else df


def align_features(df, expected_features):
    working = df.copy()
    if TARGET_COLUMN in working.columns:
        working = working.drop(columns=[TARGET_COLUMN])

    missing = [feature for feature in expected_features if feature not in working.columns]
    extra = [feature for feature in working.columns if feature not in expected_features]
    if missing:
        return None, missing, extra

    aligned = working[expected_features].apply(pd.to_numeric, errors="coerce")
    aligned = aligned.fillna(aligned.median(numeric_only=True)).fillna(0)
    return aligned, missing, extra


def score_transactions(df, artifacts):
    expected_features = artifacts["features"]
    aligned, missing, extra = align_features(df, expected_features)
    if missing:
        return None, missing, extra

    scaled = artifacts["scaler"].transform(aligned)
    selected = artifacts["selector"].transform(scaled)
    prediction = artifacts["model"].predict(selected)

    if hasattr(artifacts["model"], "predict_proba"):
        probability = artifacts["model"].predict_proba(selected)[:, 1]
    elif hasattr(artifacts["model"], "decision_function"):
        scores = artifacts["model"].decision_function(selected)
        probability = 1 / (1 + np.exp(-scores))
    else:
        probability = prediction.astype(float)

    labels = artifacts["labels"]
    result = df.copy()
    result["prediction"] = prediction
    result["prediction_label"] = [labels.get(int(item), str(item)) for item in prediction]
    result["fraud_probability"] = probability
    result["risk_band"] = pd.cut(
        result["fraud_probability"],
        bins=[-0.01, 0.30, 0.70, 1.01],
        labels=["Low", "Review", "Critical"],
    )
    return result.sort_values("fraud_probability", ascending=False), missing, extra


def risk_gauge(probability):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 42}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#b8b0ca"},
                "bar": {"color": "#ff4fa3"},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.10)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(52, 211, 153, 0.22)"},
                    {"range": [30, 70], "color": "rgba(251, 191, 36, 0.20)"},
                    {"range": [70, 100], "color": "rgba(251, 113, 133, 0.24)"},
                ],
            },
        )
    )
    return plot_theme(fig, "Fraud probability")


artifacts = load_artifacts()
default_data = load_dataset(DEFAULT_DATA_PATH)

with st.sidebar:
    st.markdown("## 💳 FraudLens AI")
    st.caption("Fintech fraud intelligence cockpit")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Overview", "Model Lab", "Data Explorer", "Predict", "Risk Monitor"],
        label_visibility="collapsed",
    )
    st.divider()
    uploaded_data = st.file_uploader("Dataset for dashboard", type=["csv"], key="analysis_csv")
    st.markdown("### Pipeline")
    if artifacts["error"]:
        st.error("Model load error")
        st.caption(artifacts["error"])
    elif artifacts["missing"]:
        st.warning("Artifacts missing")
        st.caption(", ".join(artifacts["missing"]))
    else:
        st.success("Model ready")
        st.caption(str(artifacts["folder"]))
    st.divider()
    risk_threshold = st.slider("Fraud threshold", 0.05, 0.95, 0.50, 0.01)
    chart_sample = st.slider("Chart sample", 1000, 50000, 8000, 1000)


uploaded_df, upload_error = read_uploaded_csv(uploaded_data)

if upload_error:
    alert(f"CSV upload error: {upload_error}", "error")


if uploaded_df is not None:
    data = uploaded_df
else:
    data = load_dataset(Path("creditcard.csv"))
    
st.markdown(
    """
    <div class="hero">
        <div>
            <div class="status-pill">● Real-time ML risk scoring</div>
            <h1>FraudLens AI</h1>
            <p>
                Interactive credit card fraud detection dashboard for model comparison,
                transaction exploration, batch scoring, manual scoring, and risk triage.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if data.empty:
    alert("No dataset loaded. Add creditcard.csv beside app.py or upload a CSV from the sidebar.", "warning")

has_target = not data.empty and TARGET_COLUMN in data.columns
total = len(data) if not data.empty else 0
fraud_count = int(data[TARGET_COLUMN].sum()) if has_target else 0
normal_count = total - fraud_count if has_target else 0
fraud_rate = fraud_count / total if total and has_target else 0
amount_total = float(data["Amount"].sum()) if not data.empty and "Amount" in data.columns else 0.0


if page == "Overview":
    st.markdown('<div class="section-title">📊 Command Overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Transactions", f"{total:,}", "loaded rows")
    k2.metric("Fraud Cases", f"{fraud_count:,}", "confirmed labels")
    k3.metric("Normal Cases", f"{normal_count:,}", "legitimate")
    k4.metric("Fraud Rate", pct(fraud_rate), money(amount_total))

    c1, c2, c3 = st.columns([1.15, 1, 1])
    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="margin:0;">Model Health</h3>
                <p style="color:#b8b0ca;margin-top:.5rem;">
                    Random Forest is selected as the operational model because it gives the
                    best F1-score in your notebook and keeps accuracy extremely high on the
                    imbalanced fraud dataset.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        if artifacts["missing"] or artifacts["error"]:
            alert("Prediction pipeline is not fully ready. Check the Pipeline box in the sidebar.", "warning")
        else:
            st.markdown(
                """
                <div class="mini-card">
                    <b>✅ Pipeline Ready</b>
                    <p>Model, scaler, selector, features, and labels loaded successfully.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with c3:
        st.markdown(
            f"""
            <div class="mini-card">
                <b>🎚️ Active Threshold</b>
                <p>Transactions above <b>{risk_threshold:.0%}</b> fraud probability are routed to review.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if has_target:
        chart_df = pd.DataFrame({"Label": ["Normal", "Fraud"], "Count": [normal_count, fraud_count]})
        fig = px.pie(chart_df, names="Label", values="Count", hole=0.62, title="Class distribution")
        st.plotly_chart(plot_theme(fig), use_container_width=True)


elif page == "Model Lab":
    st.markdown('<div class="section-title">🧠 Model Lab</div>', unsafe_allow_html=True)
    perf = model_metrics()
    best = perf.sort_values("F1-score", ascending=False).iloc[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Best Model", best["Model"], "by F1-score")
    m2.metric("Best F1-score", pct(best["F1-score"]), "fraud balance")
    m3.metric("Recall Leader", perf.sort_values("Recall", ascending=False).iloc[0]["Model"], "catches more fraud")

    left, right = st.columns([1, 1.25])
    with left:
        perf_display = perf.copy()
        for column in ["Accuracy", "Recall", "F1-score"]:
            perf_display[column] = perf_display[column].map(lambda value: f"{value:.2%}")
        st.dataframe(
            perf_display,
            use_container_width=True,
            hide_index=True,
        )
        alert(
            "Random Forest is the production choice from your notebook because it has the strongest F1-score. "
            "Logistic Regression has higher recall, but its very low F1-score means many more false alarms.",
            "info",
        )
    with right:
        metric_choice = st.radio("Compare by", ["Accuracy", "Recall", "F1-score"], horizontal=True)
        fig = px.bar(
            perf.sort_values(metric_choice, ascending=False),
            x="Model",
            y=metric_choice,
            color="Model",
            text=metric_choice,
            title=f"{metric_choice} comparison",
        )
        fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.08])
        st.plotly_chart(plot_theme(fig), use_container_width=True)

    radar = go.Figure()
    for _, row in perf.iterrows():
        radar.add_trace(
            go.Scatterpolar(
                r=[row["Accuracy"], row["Recall"], row["F1-score"], row["Accuracy"]],
                theta=["Accuracy", "Recall", "F1-score", "Accuracy"],
                fill="toself",
                name=row["Model"],
            )
        )
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(plot_theme(radar, "Model score radar"), use_container_width=True)


elif page == "Data Explorer":
    st.markdown('<div class="section-title">📈 Interactive Data Explorer</div>', unsafe_allow_html=True)
    if data.empty:
        alert("Upload a dataset first to explore distributions and correlations.", "warning")
    else:
        numeric_cols = [col for col in data.select_dtypes(include=np.number).columns if col != TARGET_COLUMN]
        if not numeric_cols:
            alert("No numeric columns found for plotting.", "error")
        else:
            controls = st.columns([1, 1, 1])
            feature = controls[0].selectbox("Feature", numeric_cols, index=numeric_cols.index("Amount") if "Amount" in numeric_cols else 0)
            chart_type = controls[1].selectbox("View", ["Histogram", "Boxplot", "Violin"])
            color_target = controls[2].toggle("Color by fraud class", value=has_target)
            chart_df = sample_for_charts(data, chart_sample)
            color_col = TARGET_COLUMN if color_target and has_target else None

            if chart_type == "Histogram":
                fig = px.histogram(chart_df, x=feature, color=color_col, nbins=70, marginal="box", title=f"{feature} distribution")
            elif chart_type == "Boxplot":
                fig = px.box(chart_df, x=color_col, y=feature, color=color_col, points="outliers", title=f"{feature} outlier profile")
            else:
                fig = px.violin(chart_df, x=color_col, y=feature, color=color_col, box=True, points="outliers", title=f"{feature} density shape")
            st.plotly_chart(plot_theme(fig), use_container_width=True)

            corr_defaults = [col for col in ["Time", "Amount", "V1", "V2", "V3", "V4", "V10", "V14", "V17"] if col in numeric_cols]
            corr_cols = st.multiselect("Heatmap features", numeric_cols, default=corr_defaults[:10])
            if len(corr_cols) >= 2:
                corr = chart_df[corr_cols].corr(numeric_only=True)
                heat = go.Figure(
                    go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.columns,
                        colorscale=[[0, "#22112f"], [0.5, "#a855f7"], [1, "#ff4fa3"]],
                        hovertemplate="%{x} × %{y}<br>corr=%{z:.3f}<extra></extra>",
                    )
                )
                st.plotly_chart(plot_theme(heat, "Correlation heatmap"), use_container_width=True)
            else:
                alert("Choose at least two features to build the heatmap.", "warning")


elif page == "Predict":
    st.markdown('<div class="section-title">⚡ Real-Time Prediction</div>', unsafe_allow_html=True)
    if artifacts["missing"] or artifacts["error"]:
        alert("Model artifacts could not be loaded, so prediction is disabled. The app now searches both Pattern Project and the sibling Pattern folder.", "error")
    else:
        mode = st.radio("Scoring mode", ["CSV batch", "Manual transaction"], horizontal=True)
        if mode == "CSV batch":
            batch_file = st.file_uploader("Upload transactions CSV", type=["csv"], key="batch_csv")
            if batch_file is not None:
                batch_df, batch_error = read_uploaded_csv(batch_file)
                if batch_error:
                    alert(f"Could not read file: {batch_error}", "error")
                else:
                    scored, missing, extra = score_transactions(batch_df, artifacts)
                    if missing:
                        alert("Column mismatch. Missing: " + ", ".join(missing[:14]) + (" ..." if len(missing) > 14 else ""), "error")
                    else:
                        if extra:
                            alert("Extra columns were ignored: " + ", ".join(extra[:10]) + (" ..." if len(extra) > 10 else ""), "info")
                        reviewed = scored[scored["fraud_probability"] >= risk_threshold]
                        st.session_state["latest_results"] = scored

                        p1, p2, p3, p4 = st.columns(4)
                        p1.metric("Scored", f"{len(scored):,}")
                        p2.metric("Route to Review", f"{len(reviewed):,}", pct(len(reviewed) / len(scored)))
                        p3.metric("Top Risk", pct(scored["fraud_probability"].max()))
                        p4.metric("Avg Risk", pct(scored["fraud_probability"].mean()))

                        left, right = st.columns([1, 1])
                        with left:
                            st.plotly_chart(risk_gauge(float(scored["fraud_probability"].max())), use_container_width=True)
                        with right:
                            fig = px.histogram(scored, x="fraud_probability", color="risk_band", nbins=45, title="Probability distribution")
                            fig.update_xaxes(tickformat=".0%")
                            st.plotly_chart(plot_theme(fig), use_container_width=True)

                        st.markdown("#### 🔥 Top risky transactions")
                        st.dataframe(scored.head(25), use_container_width=True, hide_index=True)
                        st.download_button(
                            "⬇️ Download scored CSV",
                            scored.to_csv(index=False).encode("utf-8"),
                            "fraud_predictions.csv",
                            "text/csv",
                            use_container_width=True,
                        )
        else:
            feature_values = {}
            features = artifacts["features"]
            defaults = {"Amount": 75.0, "Time": 0.0}
            groups = st.tabs(["Core", "V1 - V10", "V11 - V20", "V21 - V28"])
            for idx, feature in enumerate(features):
                if feature in ["Time", "Amount"]:
                    tab = groups[0]
                elif feature.startswith("V") and int(feature[1:]) <= 10:
                    tab = groups[1]
                elif feature.startswith("V") and int(feature[1:]) <= 20:
                    tab = groups[2]
                else:
                    tab = groups[3]
                with tab:
                    feature_values[feature] = st.number_input(
                        feature,
                        value=float(defaults.get(feature, 0.0)),
                        step=0.01,
                        format="%.6f",
                    )

            if st.button("Score this transaction", use_container_width=True):
                manual_df = pd.DataFrame([feature_values])
                scored, missing, _ = score_transactions(manual_df, artifacts)
                if missing:
                    alert("Manual input is missing model features.", "error")
                else:
                    prob = float(scored["fraud_probability"].iloc[0])
                    pred = int(scored["prediction"].iloc[0])
                    label = artifacts["labels"].get(pred, str(pred))
                    if pred == FRAUD_LABEL or prob >= risk_threshold:
                        st.markdown(f'<div class="result-high">🚨 {label} | Fraud probability {prob:.2%}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-low">✅ {label} | Fraud probability {prob:.2%}</div>', unsafe_allow_html=True)
                    st.plotly_chart(risk_gauge(prob), use_container_width=True)


elif page == "Risk Monitor":
    st.markdown('<div class="section-title">🔎 Risk Monitor</div>', unsafe_allow_html=True)
    latest = st.session_state.get("latest_results")
    if latest is None or latest.empty:
        alert("Score a CSV in the Predict page to unlock live monitoring, suspicious queues, and probability analytics.", "info")
    else:
        flagged = latest[latest["fraud_probability"] >= risk_threshold]
        r1, r2, r3 = st.columns(3)
        r1.metric("Flagged Queue", f"{len(flagged):,}")
        r2.metric("Critical Risk", f"{(latest['risk_band'] == 'Critical').sum():,}")
        r3.metric("Review Threshold", pct(risk_threshold))

        a, b = st.columns([1, 1])
        with a:
            dist = latest["prediction_label"].value_counts().reset_index()
            dist.columns = ["Prediction", "Count"]
            fig = px.pie(dist, names="Prediction", values="Count", hole=0.58, title="Prediction mix")
            st.plotly_chart(plot_theme(fig), use_container_width=True)
        with b:
            band = latest["risk_band"].value_counts().reset_index()
            band.columns = ["Risk Band", "Count"]
            fig = px.bar(band, x="Risk Band", y="Count", color="Risk Band", title="Risk band queue")
            st.plotly_chart(plot_theme(fig), use_container_width=True)

        st.markdown("#### 🕵️ Suspicious transaction queue")
        st.dataframe(flagged.sort_values("fraud_probability", ascending=False).head(50), use_container_width=True, hide_index=True)
