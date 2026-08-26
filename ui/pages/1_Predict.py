import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Predict", page_icon="\U0001F3AF")
st.header("Real-Time EMI Prediction")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 25, 60, 35)
        gender = st.selectbox("Gender", ["Female", "Male"])
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
        monthly_salary = st.number_input("Monthly Salary (INR)", 15000, 200000, 50000, step=1000)
        employment_type = st.selectbox("Employment Type", ["Self-employed", "Government", "Private"])
        years_of_employment = st.number_input("Years of Employment", 0, 40, 5)
        company_type = st.selectbox("Company Type", ["Mid-size", "Large Indian", "Small", "MNC", "Startup"])

    with col2:
        house_type = st.selectbox("House Type", ["Family", "Rented", "Own"])
        monthly_rent = st.number_input("Monthly Rent (INR)", 0, 100000, 10000, step=500)
        family_size = st.selectbox("Family Size", [1, 2, 3, 4, 5])
        dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4])
        school_fees = st.number_input("School Fees (INR)", 0, 50000, 0, step=500)
        college_fees = st.number_input("College Fees (INR)", 0, 100000, 0, step=500)
        travel_expenses = st.number_input("Travel Expenses (INR)", 0, 50000, 2000, step=500)
        groceries_utilities = st.number_input("Groceries & Utilities (INR)", 0, 50000, 5000, step=500)

    with col3:
        other_monthly_expenses = st.number_input("Other Monthly Expenses (INR)", 0, 50000, 1000, step=500)
        existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
        current_emi_amount = st.number_input("Current EMI Amount (INR)", 0, 100000, 0, step=500)
        credit_score = st.slider("Credit Score", 300, 850, 700)
        bank_balance = st.number_input("Bank Balance (INR)", 0, 5000000, 20000, step=1000)
        emergency_fund = st.number_input("Emergency Fund (INR)", 0, 1000000, 20000, step=1000)
        emi_scenario = st.selectbox("EMI Scenario", [
            "Education EMI", "Home Appliances EMI", "E-commerce Shopping EMI",
            "Vehicle EMI", "Personal Loan EMI"
        ])
        requested_amount = st.number_input("Requested Loan Amount (INR)", 10000, 1500000, 100000, step=1000)
        requested_tenure = st.number_input("Requested Tenure (months)", 1, 84, 24)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "age": age, "gender": gender, "marital_status": marital_status, "education": education,
        "monthly_salary": monthly_salary, "employment_type": employment_type,
        "years_of_employment": years_of_employment, "company_type": company_type,
        "house_type": house_type, "monthly_rent": monthly_rent, "family_size": family_size,
        "dependents": dependents, "school_fees": school_fees, "college_fees": college_fees,
        "travel_expenses": travel_expenses, "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses, "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount, "credit_score": credit_score,
        "bank_balance": bank_balance, "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario, "requested_amount": requested_amount,
        "requested_tenure": requested_tenure
    }

    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()["prediction"]
            st.success(f"EMI Eligibility: **{result['emi_eligibility']}**")
            st.info(f"Max Safe Monthly EMI: **₹{result['max_monthly_emi']:,.0f}**")
        elif response.status_code == 503:
            st.error("Models not loaded on the API yet. Ask admin to load resources.")
        else:
            st.error(f"Prediction failed: {response.json().get('detail', 'Unknown error')}")

    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach the API at {API_BASE_URL}. Is it running?")
    except Exception as e:
        st.error(f"Unexpected error: {e}")