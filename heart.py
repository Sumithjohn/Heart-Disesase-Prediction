import streamlit as st
import pickle
import numpy as np

# Load trained model and scaler
model_path = r"C:\Users\Smile\Downloads\python\mini project\heart_disease.pkl"
scaler_path = r"C:\Users\Smile\Downloads\python\mini project\scaler.pkl"
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Ensure scaler is a valid sklearn Scaler
if not hasattr(scaler, "transform"):
    st.error("⚠️ Error: The loaded scaler is incorrect! Please retrain and save the correct scaler.")
    st.stop()

# Streamlit UI
st.title("Heart Disease Prediction")
st.write("Enter your details below and click **Predict**.")

# User Input Form
with st.form("user_input_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    sex = st.radio("Sex", [0, 1])  # If label encoding was used (Male=1, Female=0)
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    bp = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    ekg = st.selectbox("ECG Results (restecg)", [0, 1, 2])
    max_hr = st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=220, value=150)
    exang = st.radio("Exercise-Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.2, value=1.0)
    slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
    ca = st.number_input("Number of Major Vessels (fluoroscopy)", min_value=0, max_value=3, value=1)
    thal = st.selectbox("Thallium Stress Test Result", [0, 1, 2, 3])

    # Form submission button
    submitted = st.form_submit_button("Predict")

# Predict only after form submission
if submitted:
    # Prepare input data
    data = np.array([age, sex, cp, bp, chol, fbs, ekg, max_hr, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    
    # ✅ FIX: Ensure scaler is applied correctly
    try:
        scaled_data = scaler.transform(data)
    except ValueError as e:
        st.error(f"⚠️ Error while scaling data: {e}")
        st.stop()

    # ✅ FIX: Ensure correct shape before prediction
    prediction = model.predict(scaled_data)

    # Display Results
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠️ High Risk: You may have heart disease. Please consult a doctor.")
    else:
        st.success("✅ Low Risk: Your heart health looks good!")