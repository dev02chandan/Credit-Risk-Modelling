import streamlit as st
import pandas as pd
import traceback
import joblib

# Set the page configuration
st.set_page_config(page_title="Credit Risk Prediction App", layout="wide")

def load_model(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {file_path}: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

# Load the saved XGBoost model
xgb_model = load_model('model.joblib')

# Define the input features
features = [
    'Age_Oldest_TL', 'enq_L3m', 'time_since_recent_enq', 'num_std',
    'time_since_recent_payment', 'Time_With_Curr_Empr', 'NETMONTHLYINCOME',
    'Age_Newest_TL', 'num_std_6mts', 'tot_enq', 'pct_PL_enq_L6m_of_ever', 'recent_level_of_deliq'
]

# Function to get user input
def user_input_features():
    st.sidebar.header("Input Features")
    data = {
        'Age_Oldest_TL': st.sidebar.slider('Age of Oldest Trade Line (years)', 0, 100, 10),
        'enq_L3m': st.sidebar.slider('Number of Enquiries in Last 3 Months', 0, 100, 1),
        'time_since_recent_enq': st.sidebar.slider('Time Since Recent Enquiry (months)', 0, 100, 5),
        'num_std': st.sidebar.slider('Number of Standard Deviation', 0, 100, 1),
        'time_since_recent_payment': st.sidebar.slider('Time Since Recent Payment (months)', 0, 100, 5),
        'Time_With_Curr_Empr': st.sidebar.slider('Time with Current Employer (months)', 0, 100, 5),
        'NETMONTHLYINCOME': st.sidebar.number_input('Net Monthly Income', min_value=0, max_value=100000, value=50000),
        'Age_Newest_TL': st.sidebar.slider('Age of Newest Trade Line (years)', 0, 100, 1),
        'num_std_6mts': st.sidebar.slider('Number of Standard Deviation in Last 6 Months', 0, 100, 1),
        'tot_enq': st.sidebar.slider('Total Enquiries', 0, 100, 5),
        'pct_PL_enq_L6m_of_ever': st.sidebar.slider('Percentage of Personal Loan Enquiries in Last 6 Months (%)', 0, 100, 10),
        'recent_level_of_deliq': st.sidebar.slider('Recent Level of Delinquency', 0, 10, 0)
    }
    features_df = pd.DataFrame(data, index=[0])
    return features_df

# Function to interpret prediction results
def interpret_prediction(prediction):
    mapping = {
        1: "Low Risk",
        2: "Medium Risk",
        3: "High Risk",
    }
    return mapping.get(prediction, "Unknown Risk Level")

# Main Streamlit app
def main():
    st.title("Credit Risk Prediction App")
    st.write("""
    Enter the details to predict the credit risk using the XGBoost model.
    """)
    
    input_df = user_input_features()
    
    st.write("## User Input:")
    st.write(input_df)
    
    if xgb_model is not None and st.button("Predict"):
        try:
            prediction = xgb_model.predict(input_df)[0]
            risk_level = interpret_prediction(prediction)
            st.success(f"XGBoost Prediction: {prediction} ({risk_level})")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == '__main__':
    main()
