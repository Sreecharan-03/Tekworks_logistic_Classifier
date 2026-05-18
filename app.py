import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 40px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
        text-align: center;
    }
    
    .hero-section h1 {
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }
    
    .hero-section p {
        font-size: 1.1em;
        opacity: 0.95;
        margin-bottom: 5px;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
    }
    
    .card h2 {
        color: #667eea;
        margin-bottom: 20px;
        font-size: 1.5em;
    }
    
    /* Risk indicators */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 20px 0;
        font-weight: bold;
        font-size: 1.1em;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 20px 0;
        font-weight: bold;
        font-size: 1.1em;
        box-shadow: 0 4px 15px rgba(81, 207, 102, 0.3);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffd93d 0%, #ffb700 100%);
        color: #333;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 20px 0;
        font-weight: bold;
        font-size: 1.1em;
        box-shadow: 0 4px 15px rgba(255, 183, 0, 0.3);
    }
    
    /* Input Section Styling */
    .input-section {
        background: white;
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border-top: 4px solid #667eea;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-box .metric-value {
        font-size: 2em;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .metric-box .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 15px 40px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSidebar [data-testid="stSidebarNav"] {
        background-color: transparent;
    }
    
    /* Table styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #667eea !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 30px;
        color: #666;
        margin-top: 40px;
        border-top: 2px solid rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "customer_churn_model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

# Main content
st.markdown("""
<div class="hero-section">
    <h1>📊 Customer Churn Intelligence</h1>
    <p>Advanced ML-Powered Prediction Dashboard</p>
    <p style="font-size: 0.9em; margin-top: 15px;">Identify at-risk customers and boost retention with data-driven insights</p>
</div>
""", unsafe_allow_html=True)

model = load_model()

if model is None:
    st.error("❌ Model file not found! Please ensure 'customer_churn_model.pkl' exists in the app directory.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Control Panel")
    st.markdown("---")
    prediction_mode = st.radio(
        "Select Input Mode:",
        ["Interactive Form", "Upload CSV"],
        help="Choose how to provide customer data"
    )
    st.markdown("---")
    st.markdown(
        "<small>🔧 Built with Streamlit & Scikit-Learn | v1.0</small>",
        unsafe_allow_html=True
    )

# Main content area
if prediction_mode == "Interactive Form":
    # Input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 👤 Customer Profile Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=45)
        income = st.number_input("Annual Income ($)", min_value=0.0, max_value=200000.0, value=50000.0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        transactions_month = st.number_input("Transactions per Month", min_value=0, max_value=100, value=20)
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    with col2:
        avg_purchase_value = st.number_input("Avg Purchase Value ($)", min_value=0.0, max_value=500.0, value=100.0)
        days_since_last_login = st.number_input("Days Since Last Login", min_value=0, max_value=365, value=30)
        tenure_months = st.number_input("Tenure (months)", min_value=0, max_value=200, value=24)
        num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=2)
        region = st.selectbox("Region", ["North", "South", "East", "West"])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction button
    if st.button("🔮 Predict Churn Risk", key="predict_btn"):
        # Prepare input data with correct feature names
        input_data = pd.DataFrame({
            "age": [age],
            "income": [income],
            "credit_score": [credit_score],
            "transactions_month": [transactions_month],
            "avg_purchase_value": [avg_purchase_value],
            "days_since_last_login": [days_since_last_login],
            "tenure_months": [tenure_months],
            "num_products": [num_products],
            "gender": [gender],
            "region": [region]
        })
        
        # One-hot encode categorical features to match model training
        input_data_encoded = pd.get_dummies(input_data, columns=['gender', 'region'], drop_first=False)
        
        # Ensure all expected feature columns exist (model expects specific one-hot features)
        expected_features = ['age', 'income', 'credit_score', 'transactions_month', 'avg_purchase_value',
                            'days_since_last_login', 'tenure_months', 'num_products']
        
        # Add one-hot encoded features
        for col in input_data_encoded.columns:
            if col.startswith('gender_') or col.startswith('region_'):
                expected_features.append(col)
        
        # Reorder and ensure all features are present
        for feature in model.feature_names_in_:
            if feature not in input_data_encoded.columns:
                input_data_encoded[feature] = 0
        
        # Select only the features the model was trained on, in the correct order
        input_data_encoded = input_data_encoded[model.feature_names_in_]
        
        # Make prediction
        prediction = model.predict(input_data_encoded)[0]
        
        # Display results
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Prediction Results")
        
        if prediction == 1 or prediction == "Yes":
            st.markdown(
                '<div class="risk-high">⚠️ HIGH CHURN RISK - Customer is likely to leave</div>',
                unsafe_allow_html=True
            )
            risk_level = "HIGH"
            risk_color = "#ff6b6b"
        else:
            st.markdown(
                '<div class="risk-low">✅ LOW CHURN RISK - Customer is likely to stay</div>',
                unsafe_allow_html=True
            )
            risk_level = "LOW"
            risk_color = "#51cf66"
        
        # Probability metrics
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data_encoded)[0]
            churn_prob = proba[-1] * 100
            retain_prob = proba[0] * 100
            
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Churn Probability</div>
                    <div class="metric-value">{churn_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                <div class="metric-box" style="background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);">
                    <div class="metric-label">Retention Probability</div>
                    <div class="metric-value">{retain_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Progress bar
            st.progress(min(churn_prob / 100, 1.0))
            st.caption(f"Risk Level: {risk_level}")
        
        # Customer summary
        st.markdown("#### 📋 Customer Summary")
        summary_df = pd.DataFrame({
            "Attribute": ["Age", "Annual Income", "Credit Score", "Tenure", "Region"],
            "Value": [f"{age} years", f"${income:,.2f}", f"{credit_score}", f"{tenure_months} months", region]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

else:  # CSV Upload mode
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Customer Data (CSV)")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded {len(df)} records**")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("🔮 Predict Churn for All Customers"):
            # One-hot encode the CSV data to match model training
            df_encoded = pd.get_dummies(df, columns=['gender', 'region'], drop_first=False)
            
            # Ensure all expected features exist
            for feature in model.feature_names_in_:
                if feature not in df_encoded.columns:
                    df_encoded[feature] = 0
            
            # Select only features the model was trained on, in correct order
            df_encoded = df_encoded[model.feature_names_in_]
            
            predictions = model.predict(df_encoded)
            
            df["Churn Prediction"] = predictions
            df["Churn Risk"] = df["Churn Prediction"].apply(
                lambda x: "HIGH" if x == 1 else "LOW"
            )
            
            # Display results
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Batch Prediction Results")
            
            high_risk = (df["Churn Prediction"] == 1).sum()
            low_risk = (df["Churn Prediction"] == 0).sum()
            high_risk_pct = (high_risk / len(df)) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Total Customers</div>
                    <div class="metric-value">{len(df)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-box" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);">
                    <div class="metric-label">At-Risk Customers</div>
                    <div class="metric-value">{high_risk}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-box" style="background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);">
                    <div class="metric-label">Healthy Customers</div>
                    <div class="metric-value">{low_risk}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.progress(high_risk_pct / 100)
            st.caption(f"Churn Risk Rate: {high_risk_pct:.1f}%")
            
            st.markdown("#### 🎯 Detailed Predictions")
            st.dataframe(df[["Churn Risk", "Churn Prediction"]].head(20), use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>🚀 Customer Churn Intelligence Dashboard | Powered by Advanced Machine Learning</p>
    <p style="font-size: 0.85em; color: #999;">© 2024 | Data-Driven Business Intelligence</p>
</div>
""", unsafe_allow_html=True)
