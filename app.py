import os
import urllib.request
import gzip
import shutil
import joblib
import streamlit as st
import pandas as pd

MODEL_URL = "https://github.com/Berl-cloud/bank-term_deposit-prediction/raw/refs/heads/main/final_model_streamlit.pkl.gz"
COMPRESSED_MODEL_PATH = "final_model_streamlit.pkl.gz"
MODEL_PATH = "final_model_streamlit.pkl"

# Download the compressed model if not already present
if not os.path.exists(COMPRESSED_MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, COMPRESSED_MODEL_PATH)

# Unzip the .gz file
if not os.path.exists(MODEL_PATH):
    with gzip.open(COMPRESSED_MODEL_PATH, 'rb') as f_in:
        with open(MODEL_PATH, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# Load the model
model = joblib.load(MODEL_PATH)




# Streamlit UI
st.set_page_config(page_title="Term Deposit Predictor", layout="centered")
st.title("Term Deposit Subscription Predictor")
st.write("Use this tool to predict whether a bank client is likely to subscribe to a term deposit based on their information.")

# User inputs
age = st.slider("Age", 18, 95, 30)
job = st.selectbox("Job Type", ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'student', 'self-employed', 'unknown'])
marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
education = st.selectbox("Education Level", ['secondary', 'tertiary', 'primary', 'unknown'])
default = st.selectbox("Has Credit in Default?", ['no', 'yes'])
housing = st.selectbox("Has Housing Loan?", ['yes', 'no'])
loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone'])
month = st.selectbox("Last Contact Month", ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'])
day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
duration = st.number_input("Call Duration (in seconds)", min_value=0, value=100)
campaign = st.number_input("Number of Contacts in Current Campaign", min_value=1, value=1)
previous = st.number_input("Number of Previous Contacts", min_value=0, value=0)
poutcome = st.selectbox("Outcome of Previous Campaign", ['failure', 'nonexistent', 'success'])

input_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'month': [month],
    'day_of_week': [day_of_week],
    'duration': [duration],
    'campaign': [campaign],
    'previous': [previous],
    'poutcome': [poutcome]
})

if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1]
    
    if prediction[0] == 1:
        st.success(f"The client is likely to subscribe! (Confidence: {prediction_proba:.2%})")
    else:
        st.error(f"The client is unlikely to subscribe. (Confidence: {prediction_proba:.2%})")

    st.markdown("---")
    st.subheader("Client Input Summary")
    st.write(input_data)
