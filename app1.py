import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import boxcox
import joblib
import plotly.express as px

# ==============================
# Load model, scaler & lambda
# ==============================
scaler = joblib.load("scaler.pkl")
model = joblib.load("xgb_model.pkl")
lambda_ = joblib.load("boxcox_lambda.pkl")

st.set_page_config(page_title="Predictive Maintenance App", layout="wide")

# ==============================
# 🎨 Premium CSS
# ==============================
st.markdown("""
<style>
.main {background-color: #f4f6f9;}
h1 {text-align:center; color:#2E8B57;}
.creator {font-weight:bold;}

.card {
    padding:12px;
    border-radius:12px;
    margin-bottom:10px;
    color:white;
    transition:0.3s;
}
.card:hover {transform:scale(1.05);}

.card1 {background:linear-gradient(135deg,#667eea,#764ba2);}
.card2 {background:linear-gradient(135deg,#ff7e5f,#feb47b);}
.card3 {background:linear-gradient(135deg,#43cea2,#185a9d);}
.card4 {background:linear-gradient(135deg,#11998e,#38ef7d);}
.card5 {background:linear-gradient(135deg,#fc466b,#3f5efb);}
.card6 {background:linear-gradient(135deg,#f7971e,#ffd200);}

.stButton>button {
    background:linear-gradient(90deg,#4CAF50,#2E8B57);
    color:white; border-radius:10px; height:3em;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Header
# ==============================
st.markdown(""" <div style="display:flex; justify-content:space-between;"> 
            <div class="creator">Created by Archit Tomar</div> <div style="width:100%;"> 
            <h1>🚀 Predictive Maintenance App</h1> </div> </div> """, unsafe_allow_html=True)

# ==============================
# Sidebar
# ==============================
st.sidebar.title("🎯 Navigation")
section = st.sidebar.radio(
    "Choose Section:",
    ["Dashboard & Prediction", "Model Performance", "Confusion Matrix", "Dataset Info", "About"]
)

# ==============================
# 🔥 Sidebar Cards
# ==============================
st.sidebar.markdown("### 📌 Quick Info")

st.sidebar.markdown('<div class="card card1">📊 Model: XGBoost</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="card card2">⚖️ SMOTE Applied</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="card card3">📦 Dataset: 10,000 Samples</div>', unsafe_allow_html=True)

# 🔥 NEW: KPI tiles moved to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Metrics")

st.sidebar.markdown('<div class="card card4">🎯 Accuracy: 97.76%</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="card card5">📊 ROC-AUC: 0.8976</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="card card6">⚙️ Model Type: XGBoost</div>', unsafe_allow_html=True)

# ==============================
# Dashboard + Prediction
# ==============================
if section == "Dashboard & Prediction":

    st.subheader("🔍 Enter Machine Details")

    col1, col2 = st.columns(2)

    with col1:
        Type_label = st.selectbox("Type", ["L","M","H"])
        type_map = {"L":0,"M":1,"H":2}
        Type = type_map[Type_label]

        air_temp = st.number_input("Air Temperature (K)", value=300.0)
        process_temp = st.number_input("Process Temperature (K)", value=310.0)

    with col2:
        torque = st.number_input("Torque (Nm)", value=40.0)
        tool_wear = st.number_input("Tool Wear (min)", value=100.0)
        rotational_speed = st.number_input("Rotational Speed (rpm)", value=1500.0)

    st.markdown("---")

    if st.button("🚀 Predict Machine Status"):

        with st.spinner("Analyzing machine data..."):

            rot_bc = boxcox([rotational_speed], lmbda=lambda_)[0]

            numeric_data = pd.DataFrame([[ 
                air_temp, process_temp, torque, tool_wear, rot_bc
            ]], columns=[
                'Air temperature [K]',
                'Process temperature [K]',
                'Torque [Nm]',
                'Tool wear [min]',
                'Rotational speed [rpm]_log'
            ])

            scaled = scaler.transform(numeric_data)
            final_input = np.concatenate([[Type], scaled[0]]).reshape(1, -1)

            prediction = model.predict(final_input)[0]
            probability = model.predict_proba(final_input)[0][1]

        st.markdown("### 🧾 Input Summary")
        st.write(numeric_data)

        st.markdown("### 📊 Prediction Result")

        st.progress(float(probability))
        st.caption(f"Confidence: {probability:.2f}")

        if probability > 0.7:
            st.error("🔴 High Risk")
        elif probability > 0.4:
            st.warning("🟡 Medium Risk")
        else:
            st.success("🟢 Low Risk")

        if prediction == 1:
            st.error("⚠️ Machine Failure Likely")
        else:
            st.success("✅ Machine Healthy")

        st.info("Prediction based on temperature, torque, and tool wear patterns.")

# ==============================
# Model Performance
# ==============================
elif section == "Model Performance":

    st.subheader("📈 Model Performance")

    report_df = pd.DataFrame({
        "Precision":[0.99,0.63],
        "Recall":[0.98,0.81],
        "F1":[0.99,0.71]
    }, index=["No Failure","Failure"])

    st.dataframe(report_df, use_container_width=True)

# ==============================
# Confusion Matrix
# ==============================
elif section == "Confusion Matrix":

    st.subheader("🎯 Confusion Matrix")

    cm = np.array([[2375,40],[16,69]])

    fig = px.imshow(cm,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    labels=dict(x="Predicted", y="Actual"))

    st.plotly_chart(fig)

# ==============================
# Dataset Info
# ==============================
elif section == "Dataset Info":

    st.subheader("📄 Dataset Info")

    st.info("Contains 10,000 machine records with sensor readings.")

    df = pd.DataFrame({
        "Feature":["Type","Air Temp","Process Temp","RPM","Torque","Tool Wear"],
        "Description":[
            "Machine category",
            "Ambient temperature",
            "Process temp",
            "Speed",
            "Torque",
            "Wear duration"
        ]
    })

    st.dataframe(df)

# ==============================
# About
# ==============================
elif section == "About":

    st.subheader("ℹ️ About")

    st.write("""
    Predictive Maintenance using Machine Learning.
    
    - Model: XGBoost  
    - Preprocessing: StandardScaler + Box-Cox  
    - Imbalance: SMOTE  
    """)

# ==============================
# Footer
# ==============================
st.markdown("""
<hr>
<center>🚀 Built with Streamlit | ML Project by Archit Tomar</center>
""", unsafe_allow_html=True)