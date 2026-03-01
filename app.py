import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Student Burnout AI System",
    layout="wide",
    page_icon="🎓"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1 style='text-align: center;'>AI-Based Student Burnout & Dropout Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Behavioral Clustering • Risk Scoring • Explainable AI Dashboard</p>", unsafe_allow_html=True)
st.markdown("---")

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
rf_model = joblib.load("rf_model.pkl")
log_model = joblib.load("log_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.title("Student Behavioral Inputs")

with st.sidebar.expander("📘 Academic Engagement", expanded=True):

    lms = st.slider(
        "LMS Logins per Week",
        0, 50, 20,
        help="Average weekly learning platform activity"
    )

    login_trend = st.selectbox(
        "Login Trend",
        ["increase", "stable", "decrease"],
        help="Recent change in LMS activity pattern"
    )

with st.sidebar.expander("📝 Assignment Behavior", expanded=True):

    col1, col2 = st.columns(2)

    with col1:
        delay = st.slider(
            "Avg Delay (Days)",
            0.0, 10.0, 2.0,
            help="Average submission delay"
        )

    with col2:
        missed = st.slider(
            "Missed Assignments",
            0, 10, 1,
            help="Total missed assignments"
        )

with st.sidebar.expander("📅 Attendance Monitoring", expanded=True):

    attendance = st.slider(
        "Attendance Percentage",
        0.0, 100.0, 75.0,
        help="Overall attendance level"
    )

    attendance_trend = st.selectbox(
        "Attendance Trend",
        ["increase", "stable", "decrease"],
        help="Recent change in attendance pattern"
    )

with st.sidebar.expander("🧠 Behavioral Signals", expanded=False):

    sentiment = st.slider(
        "Feedback Sentiment Score",
        -1.0, 1.0, 0.0,
        help="Sentiment from feedback forms (-1 negative, +1 positive)"
    )

    variance = st.slider(
        "Activity Variance",
        0.0, 15.0, 5.0,
        help="Irregularity in study pattern"
    )

    late = st.slider(
        "Late Night Activity Ratio",
        0.0, 1.0, 0.2,
        help="Proportion of late-night study behavior"
    )

# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------
trend_map = {"increase":2, "stable":1, "decrease":0}

input_data = pd.DataFrame([[
    lms,
    trend_map[login_trend],
    delay,
    missed,
    attendance,
    trend_map[attendance_trend],
    sentiment,
    variance,
    late
]], columns=[
    "lms_logins_per_week",
    "login_trend_change",
    "avg_submission_delay_days",
    "missed_assignments_count",
    "attendance_percent",
    "attendance_trend_change",
    "feedback_sentiment_score",
    "activity_variance",
    "late_night_activity_ratio"
])

# --------------------------------------------------
# MODEL PREDICTIONS
# --------------------------------------------------
scaled_input = scaler.transform(input_data)

cluster = kmeans.predict(scaled_input)[0]
cluster_map = {0:"Low Risk Segment", 1:"Medium Risk Segment", 2:"High Risk Segment"}
segment = cluster_map.get(cluster, "Unknown")

dropout_prob = log_model.predict_proba(input_data)[0][1]
risk_score = float(np.clip(rf_model.predict(input_data)[0], 0, 100))

if risk_score < 33:
    category = "Low"
elif risk_score < 66:
    category = "Medium"
else:
    category = "High"

# --------------------------------------------------
# TOP METRICS
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Risk Score", f"{risk_score:.2f}/100")
col2.metric("Risk Category", category)
col3.metric("Dropout Probability", f"{dropout_prob*100:.2f}%")
col4.metric("Behavior Segment", segment)

st.markdown("---")

# --------------------------------------------------
# GAUGE + FEATURE IMPORTANCE
# --------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Burnout Risk Gauge")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "salmon"}
            ],
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

with right:
    st.subheader("Key Behavioral Triggers")

    importance_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(fig_importance, use_container_width=True)

st.markdown("---")

# --------------------------------------------------
# SHAP EXPLANATION
# --------------------------------------------------
st.subheader("Explainable AI – Individual Risk Breakdown")

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer(input_data)

shap_df = pd.DataFrame({
    "Feature": input_data.columns,
    "Impact on Risk Score": shap_values.values[0]
}).sort_values(by="Impact on Risk Score", key=abs, ascending=False)

fig_shap = px.bar(
    shap_df,
    x="Impact on Risk Score",
    y="Feature",
    orientation="h"
)

st.plotly_chart(fig_shap, use_container_width=True)

st.markdown("---")

# --------------------------------------------------
# INTERVENTION STRATEGY
# --------------------------------------------------
st.subheader("Recommended Intervention Strategy")

if risk_score < 33:
    st.success("""
    • Maintain academic mentoring  
    • Encourage leadership activities  
    • Monitor engagement trends monthly  
    """)
elif risk_score < 66:
    st.warning("""
    • Schedule faculty mentoring  
    • Monitor assignment submissions  
    • Encourage peer group participation  
    • Conduct stress management workshop  
    """)
else:
    st.error("""
    • Immediate counseling referral  
    • Personalized academic recovery plan  
    • Weekly attendance monitoring  
    • Temporary workload adjustment  
    """)

st.markdown("---")
st.caption("AI Behavioral Analytics System | KMeans + Logistic Regression + Random Forest + SHAP + Streamlit")