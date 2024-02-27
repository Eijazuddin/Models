import streamlit as st

import pandas as pd
import joblib

st.title("Stress Model")

posts = st.text_area("Enter your concern")

if st.button("Predict"):
    vector = joblib.load("stress.h5")
   
    predictions = vector.transform([posts])

    model = joblib.load("stress_model.h5")

    pred = model.predict(predictions)

    st.success(pred)

# We need not use conditional statements as its in labels and we didn;t encode it
    
