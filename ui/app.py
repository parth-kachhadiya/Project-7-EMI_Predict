import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="\U0001F4B0",
    layout="wide",
)

st.title("\U0001F4B0 EMIPredict AI")
st.markdown("""
Intelligent Financial Risk Assessment Platform — predicts EMI eligibility
(classification) and maximum safe monthly EMI amount (regression).

Use the sidebar to navigate:
- **Predict** — real-time EMI eligibility & max EMI amount for an applicant
- **Admin Dashboard** — model performance, MLflow experiment tracking, trigger retraining
""")