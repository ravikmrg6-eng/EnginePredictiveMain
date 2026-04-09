import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="ravikmrg6/CapStnProjMlopsPred", filename="cs_pred_maintenance.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Engine Predictive Maintenance")
st.write("""
This application predicts engine failures in automotive application
based on its engine sensor inputs such as RPM, Temperature, Pressure and other sensore readings ...
Please enter the app details below to get a prediction of engine health.
""")

# User input

Engine_RPM = st.number_input("Indicating the Engine Speed in RPM ", min_value=1, max_value=2000, value=1, step=1)
Lub_Oil_Pressure = st.number_input("The pressure of the Lubricating oil in the engine in kPa", min_value=1.0000000000, max_value=20.0000000000, value=1.0000000000, step=0.0000000001 , format="%.10f" )
Fuel_Pressure = st.number_input("The Fuel Pressure in kPa", min_value=1.000000, max_value=20.000000, value=1.000000, step=0.000001, format="%.6f" )
Coolant_Pressure = st.number_input("The Pressure of the Engine Coolant in kPa", min_value=1.000000, max_value=20.000000, value=1.000000, step=0.000001, format="%.6f")
Lub_Oil_Temperature = st.number_input("The Temparture of the Lub Oil", min_value=1.000000, max_value=100.000000, value=1.000000, step=0.000001, format="%.6f")
Coolant_Temperature = st.number_input("The Temparture of the Engine Coolant",  min_value=1.000000, max_value=100.000000, value=1.000000, step=0.000001, format="%.6f")

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine rpm': Engine_RPM,
    'Lub oil pressure': Lub_Oil_Pressure,
    'Fuel pressure': Fuel_Pressure,
    'Coolant pressure': Coolant_Pressure,
    'lub oil temp': Lub_Oil_Temperature,
    'Coolant temp': Coolant_Temperature
    }])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Active" if prediction ==1 else "Faulty"
    st.subheader("Prediction Result:")
    st.success(f": Modle predict :: Engine Condition **{result}**")
