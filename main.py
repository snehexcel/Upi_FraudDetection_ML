''' 
['Amount', 'Timestamp', 'MerchantCategory', 'TransactionType', 'IPAddress', 'Latitude']
'''

import streamlit as st 
import pandas as pd 
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

model = joblib.load("logisticRegression.pkl")
data = pd.read_csv("UPI_Fraud.csv")

le_city = LabelEncoder()
le_city.fit(data['Transaction_City'])

le_status = LabelEncoder()
le_status.fit(data['Transaction_Status'])

le_channel = LabelEncoder()
le_channel.fit(data['Transaction_Channel'])

st.set_page_config(
    page_title="Sneha's UPI Fraud Detection", 
    page_icon = "👀",
    layout = "wide"
)
st.title(":blue[UPI] :green[Fraud] :red[Detection]")
merchant_ID = st.slider(label="Select the merchant ID", min_value=0, max_value=647)
device_ID = st.slider(label="Select the device ID", min_value=0, max_value=647)
location = st.selectbox(label="Select the transaction city", options=list(data['Transaction_City'].unique()))
transaction_status = st.selectbox(label="Select the transaction status", options=list(data['Transaction_Status'].unique()))
transaction_channel = st.selectbox(label="Select the transaction channel", options=list(data['Transaction_Channel'].unique()))
amount = st.slider(label="Select the amount", min_value = data["amount"].min(), max_value = data["amount"].max())

encoded_city = le_city.transform([location])[0]
encoded_status = le_status.transform([transaction_status])[0]
encoded_channel = le_channel.transform([transaction_channel])[0]

input_data = np.array([[merchant_ID, device_ID, encoded_city, encoded_status, encoded_channel, amount]])

if st.button("Predict Fraud?"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction Seems Legitimate")