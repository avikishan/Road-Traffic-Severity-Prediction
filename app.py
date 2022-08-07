import streamlit as st
import pandas as pd
import numpy as np
import joblib
import bz2file as bz2
from sklearn.ensemble import ExtraTreesClassifier
from prediction import get_prediction, ordinal_encoder, decompress_pickle

model=decompress_pickle(r"Model/model1.pbz2")

st.set_page_config(page_title="Accident Severity Prediction App",page_icon="🚙",layout="wide")

#Creating the option list
option_light_conditions=['Daylight','Darkness - lights lit','Darkness - no lighting','Darkness - lights unlit']
#option_number_of_casualties=[1,2,3,4,5,6,7,8]
#option_number_of_vehicle_involved=[1,2,3,4,5,6,7]
option_age_band_of_driver=['18-30', '31-50', 'Under 18', 'Over 51', 'Unknown']
#option_minute=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
#option_hour=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
option_day_of_week=['Sunday','Monday', 'Tuesday','Wednesday','Thursday','Friday','Saturday']
option_type_of_junction=['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown', 'T Shape', 'X Shape']
option_road_surface_condition=['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep']
option_driving_experience=['1-2yr', 'Above 10yr','5-10yr','2-5yr','No Licence','Below 1yr','unknown']
option_sex_of_casualty=['na', 'Male', 'Female']

#Features
features=['hour','minute','Days of Week','Light Condition','Number of Casualties','Number of vehicle involved','Age band of driver','Type of Junction','Road Surface Condition','Driving Experience','Lanes Or Medians']

st.markdown("<h1 style='Text align: center;'>Accident Severity Prediction Application 🚙</h1>",unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):
        st.subheader("Enter the Input for the following features: ")

        hour=st.slider("Pickup Hour: ",0,23,value=0,format="%d")
        minute=st.slider("Pickup Minute: ",0,59,value=0,format="%d")
        day_of_week=st.selectbox("Select Day of the Week: ",options=option_day_of_week)
        light_condition=st.selectbox("Select the Conditions of Light: ",options=option_light_conditions)
        number_of_casualties=st.slider("Select the Number of Casualties: ",1,8,value=1,format="%d")
        number_of_vehicle_involved=st.slider("Select the Number of Vehicles Involved: ",1,7,value=1,format="%d")
        age_band_of_driver=st.selectbox("Select the Age band of Driver: ",options=option_age_band_of_driver)
        type_of_junction=st.selectbox("Select the type of Junction: ",options=option_type_of_junction)
        road_surface_condition=st.selectbox("Select the type of Road Surface Condition: ",options=option_road_surface_condition)
        driving_experience=st.selectbox("Select the Driving Experience of Driver: ",options=option_driving_experience)
        sex_of_casualty=st.selectbox("Select the Sex of Casualty: ",options=option_sex_of_casualty)

        submit=st.form_submit_button("Predict")
    
    if submit:
        
        day_of_week=ordinal_encoder(day_of_week,['Monday', 'Sunday', 'Friday', 'Wednesday', 'Saturday', 'Thursday', 'Tuesday'])
        light_condition=ordinal_encoder(light_condition,option_light_conditions)
        #number_of_casualties=ordinal_encoder(number_of_casualties)
        #number_of_vehicle_involved=ordinal_encoder(number_of_vehicle_involved)
        age_band_of_driver=ordinal_encoder(age_band_of_driver,option_age_band_of_driver)
        type_of_junction=ordinal_encoder(type_of_junction,option_type_of_junction)
        road_surface_condition=ordinal_encoder(road_surface_condition,option_road_surface_condition)
        driving_experience=ordinal_encoder(driving_experience,option_driving_experience)
        sex_of_casualty=ordinal_encoder(sex_of_casualty,option_sex_of_casualty)

        data={'Light_conditions':[light_condition],
              'Number_of_casualties':[number_of_casualties],
              'Number_of_vehicles_involved':[number_of_vehicle_involved], 
              'Age_band_of_driver':[age_band_of_driver], 
              'minute':[minute],
              'Day_of_week':[day_of_week], 
              'Types_of_Junction':[type_of_junction], 
              'Driving_experience':[driving_experience],
              'Road_surface_conditions':[road_surface_condition], 
              'hour':[hour], 
              'Sex_of_casualty':[sex_of_casualty]}
        data=pd.DataFrame(data)
        #data=np.array([light_condition,number_of_casualties,number_of_vehicle_involved,age_band_of_driver,minute,day_of_week,type_of_junction,driving_experience,road_surface_condition,hour,sex_of_casualty]).reshape(1,-1)

        pred=get_prediction(data,model=model)

        st.write(f"The Predicted Severity of Accident is: {pred[0]}")

if __name__=='__main__':
    main()


