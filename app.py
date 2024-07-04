import streamlit as st
import pandas as pd
import traceback
import joblib
import sys
import numpy as np

def load_model(file_path):
    try:
        model = joblib.load(file_path)
        st.success(f"Successfully loaded model from {file_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model from {file_path}: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

# Load the saved model
model = load_model('model.joblib')

# Define the input features
features = [
    'Age_Oldest_TL', 'enq_L3m', 'time_since_recent_enq', 'num_std',
    'time_since_recent_payment', 'Time_With_Curr_Empr', 'NETMONTHLYINCOME',
    'Age_Newest_TL', 'num_std_6mts', 'tot_enq', 'pct_PL_enq_L6m_of_ever', 'recent_level_of_deliq'
]

# Function to get user input
def user_input_features():
    data = {}
    data['Age_Oldest_TL'] = st.number_input('Age of Oldest Trade Line', min_value=0, max_value=100, value=10)
    data['enq_L3m'] = st.number_input('Number of Enquiries in Last 3 Months', min_value=0, max_value=100, value=1)
    data['time_since_recent_enq'] = st.number_input('Time Since Recent Enquiry', min_value=0, max_value=100, value=5)
    data['num_std'] = st.number_input('Number of Standard Deviation', min_value=0, max_value=100, value=1)
    data['time_since_recent_payment'] = st.number_input('Time Since Recent Payment', min_value=0, max_value=100, value=5)
    data['Time_With_Curr_Empr'] = st.number_input('Time with Current Employer', min_value=0, max_value=100, value=5)
    data['NETMONTHLYINCOME'] = st.number_input('Net Monthly Income', min_value=0, max_value=100000, value=50000)
    data['Age_Newest_TL'] = st.number_input('Age of Newest Trade Line', min_value=0, max_value=100, value=1)
    data['num_std_6mts'] = st.number_input('Number of Standard Deviation in Last 6 Months', min_value=0, max_value=100, value=1)
    data['tot_enq'] = st.number_input('Total Enquiries', min_value=0, max_value=100, value=5)
    data['pct_PL_enq_L6m_of_ever'] = st.number_input('Percentage of Personal Loan Enquiries in Last 6 Months', min_value=0, max_value=100, value=10)
    data['recent_level_of_deliq'] = st.number_input('Recent Level of Delinquency', min_value=0, max_value=10, value=0)
    features_df = pd.DataFrame(data, index=[0])
    return features_df

# Main Streamlit app
def main():
    st.title("Credit Risk Prediction App")
    st.write("""
    ### Enter the following details to predict the credit risk:
    """)
    
    st.write("""
    - **Age of Oldest Trade Line**: Age of the oldest trade line in years.
    - **Number of Enquiries in Last 3 Months**: Number of credit enquiries in the last 3 months.
    - **Time Since Recent Enquiry**: Time (in months) since the last credit enquiry.
    - **Number of Standard Deviation**: Standard deviation of the number of trade lines.
    - **Time Since Recent Payment**: Time (in months) since the most recent payment.
    - **Time with Current Employer**: Time (in months) with the current employer.
    - **Net Monthly Income**: Net monthly income in the local currency.
    - **Age of Newest Trade Line**: Age of the newest trade line in years.
    - **Number of Standard Deviation in Last 6 Months**: Standard deviation of the number of trade lines in the last 6 months.
    - **Total Enquiries**: Total number of credit enquiries.
    - **Percentage of Personal Loan Enquiries in Last 6 Months**: Percentage of personal loan enquiries in the last 6 months out of all enquiries.
    - **Recent Level of Delinquency**: Level of recent delinquency on a scale of 0 to 10.
    """)
    
    input_df = user_input_features()
    
    st.write("### User Input:")
    st.write(input_df)
    
    # Prediction
    if st.button("Predict with XGBoost"):
        xgb_prediction = model.predict(input_df)
        st.write(f"XGBoost Prediction: {xgb_prediction[0]}")

if __name__ == '__main__':
    main()