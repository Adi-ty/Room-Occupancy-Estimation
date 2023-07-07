import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("occupancy.pkl", "rb"))

st.title("Room Occupancy Predictor")

cols = {}

for column in ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light',
               'S2_Light', 'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound',
               'S4_Sound', 'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']:
    cols[column] = st.number_input(column, 0, 1000000)

if st.button("Predict Occupancy"):
    input_data = []
    for column in ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light',
                   'S2_Light', 'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound',
                   'S4_Sound', 'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']:
        input_data.append(cols[column])

    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)

    st.write("Room Occupancy Count:", prediction[0])
