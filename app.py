
import streamlit as st
import pandas as pd
import joblib


model = joblib.load("loan_model.pkl")


st.title("Loan Default Risk Prediction")

st.write(
    "Enter the applicant's details below "
    "to estimate loan default risk."
)



with st.form("loan_form"):

    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    person_gender = st.selectbox(
        "Gender",
        ["female", "male"]
    )

    person_education = st.selectbox(
        "Education",
        ["High School", "Associate", "Bachelor",
         "Master", "Doctorate"]
    )

    person_income = st.number_input(
        "Annual Income",
        min_value=0,
        value=50000
    )

    person_emp_exp = st.number_input(
        "Employment Experience (years)",
        min_value=0,
        value=2
    )

    person_home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0,
        value=10000
    )

    loan_intent = st.selectbox(
        "Loan Purpose",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        value=10.0
    )

    loan_percent_income = st.number_input(
        "Loan as a Proportion of Income",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (years)",
        min_value=0.0,
        value=3.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=0,
        max_value=1000,
        value=650
    )

    previous_loan_defaults_on_file = st.selectbox(
        "Previous Loan Defaults",
        ["No", "Yes"]
    )

    
    submitted = st.form_submit_button(
        "Predict Default Risk"
    )


if submitted:

   
    input_data = pd.DataFrame([{
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length":
            cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file":
            previous_loan_defaults_on_file
    }])

    prediction = model.predict(input_data)[0]

    
    probabilities = model.predict_proba(input_data)[0]

   
    class_index = list(model.classes_).index(1)

    default_probability = probabilities[class_index] * 100


    st.subheader("Prediction Results")

    st.metric(
        label="Estimated Default Probability",
        value=f"{default_probability:.2f}%"
    )

    if default_probability < 20:
        risk_level = "Lower estimated risk"

    elif default_probability < 50:
        risk_level = "Moderate estimated risk"

    else:
        risk_level = "Higher estimated risk"

    st.write(f"**Risk indicator:** {risk_level}")


    if prediction == 1:
        st.warning(
            "The model predicts potential loan default."
        )

    else:
        st.success(
            "The model predicts no loan default."
        )

 
    st.caption(
        "This is a machine learning estimate, not a guarantee. "
        "The risk bands are illustrative and have not been "
        "validated for real-world lending decisions."
    )
