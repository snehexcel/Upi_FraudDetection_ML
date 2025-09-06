import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import geocoder

# Load model and data
model = joblib.load("LinearRegression.pkl")
data = pd.read_csv("Indian_UPI_Data.csv")

# Encode categorical variables
le_merchant = LabelEncoder().fit(data['MerchantCategory'])
le_transaction = LabelEncoder().fit(data['TransactionType'])
le_ip = LabelEncoder().fit(data['IPAddress'])
le_bankname = LabelEncoder().fit(data['BankName'])

# Streamlit page configuration
st.set_page_config(
    page_title="Sneha's UPI Fraud Detection",
    page_icon="👀",
    layout="wide"
)

# Load user location
myloc = geocoder.ip('me')
latitude = myloc.latlng[0] if myloc.ok and myloc.latlng else 20.5937
longitude = myloc.latlng[1] if myloc.ok and myloc.latlng else 78.9629

# Inject CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f4f4f4;
    }
    .main {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
    }
    h1 {
        text-align: center;
        color: #1f77b4;
    }
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover {
        background-color: #d62828;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section with image ---
st.markdown("<h1>🔐 UPI Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("Use AI to detect potentially fraudulent UPI transactions in real-time 🚨")

# --- Sidebar with dataset info ---
with st.sidebar:
    st.header("📊 Dataset Summary")
    st.write("Total Transactions:", len(data))
    st.write("Unique Merchants:", data['MerchantCategory'].nunique())
    st.write("Transaction Types:", data['TransactionType'].unique().tolist())
    st.write("Banks:", data['BankName'].nunique())
    st.markdown("---")
    st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}), zoom=4)

# --- Input Fields ---
st.markdown("### 📝 Enter Transaction Details")
amount = st.slider("💸 Transaction Amount", float(data['Amount'].min()), float(data['Amount'].max()))
merchant_category = st.selectbox("🏪 Merchant Category", sorted(data['MerchantCategory'].unique()))
transaction_type = st.selectbox("🔄 Transaction Type", sorted(data['TransactionType'].unique()))
bank_name = st.selectbox("🏦 Bank Name", sorted(data['BankName'].unique()))

# Encode inputs
encoded_merchant = le_merchant.transform([merchant_category])[0]
encoded_transaction = le_transaction.transform([transaction_type])[0]
encoded_bank = le_bankname.transform([bank_name])[0]

# Prepare input for prediction
input_data = np.array([[amount, encoded_merchant, encoded_transaction, latitude, encoded_bank]])

# Predict button
if st.button("🔍 Predict Fraud?"):
    prediction = model.predict(input_data)[0]
    score = np.clip(prediction, 0, 1)  # Treat output as fraud probability (pseudo score)

    st.markdown(f"### 🔎 Fraud Risk Score: {score:.2f}")
    
    if score >= 0.5:
        st.error("🚨 Fraudulent Transaction Detected!")
        
    else:
        st.success("✅ Transaction Seems Legitimate")
        

# --- Footer ---
st.markdown("---")
st.markdown(
    "<center><small>Made with ❤ by Sneha | Powered by AI & Streamlit</small></center>",
    unsafe_allow_html=True
)