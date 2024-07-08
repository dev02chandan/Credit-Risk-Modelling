import streamlit as st
import pandas as pd
import traceback
import joblib

# Set the page configuration
st.set_page_config(page_title="CrediSense")

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

# Feature explanations
explanations = {
    'Age_Oldest_TL': 'The age of the oldest trade line (credit account) in years.',
    'enq_L3m': 'The number of credit enquiries made in the last 3 months.',
    'time_since_recent_enq': 'The time since the most recent credit enquiry, measured in months.',
    'num_std': 'The number of standard deviations of trade lines. This measures the variability or diversity in the number of trade lines.',
    'time_since_recent_payment': 'The time since the most recent payment, measured in months.',
    'Time_With_Curr_Empr': 'The duration of time the borrower has been with their current employer, measured in months.',
    'NETMONTHLYINCOME': 'The net monthly income of the borrower, measured in currency units.',
    'Age_Newest_TL': 'The age of the newest trade line (credit account) in years.',
    'num_std_6mts': 'The number of standard deviations of trade lines in the last 6 months. This measures the variability or diversity in the number of trade lines in the recent 6 months.',
    'tot_enq': 'The total number of credit enquiries made by the borrower.',
    'pct_PL_enq_L6m_of_ever': 'The percentage of personal loan enquiries made in the last 6 months compared to all enquiries ever made.',
    'recent_level_of_deliq': 'The recent level of delinquency, measured on a scale from 0 to 10, where 0 indicates no delinquency and 10 indicates high delinquency.'
}

# Function to get user input
def user_input_features():
    input_data = []

    st.header("Input Features")
    for feature, explanation in explanations.items():
        st.write(f'#### {explanation}')
        if feature == 'NETMONTHLYINCOME':
            value = st.number_input(f"{feature}", min_value=0, max_value=100000, value=50000)
        else:
            value = st.slider(f"{feature}", 0, 100, 1)
        input_data.append((feature, value))

    features_df = pd.DataFrame(input_data, columns=['Feature', 'Value'])
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
    st.image('images/logo.png')
    st.title("CrediSense")
    st.write("""
    Enter the details to predict the credit risk using the AI model.
    """)
    
    input_df = user_input_features()
    
    st.write("## User Input Summary:")
    st.table(input_df)
    
    input_data = {row['Feature']: row['Value'] for index, row in input_df.iterrows()}
    input_df_formatted = pd.DataFrame(input_data, index=[0])
    
    if xgb_model is not None and st.button("Predict"):
        try:
            prediction = xgb_model.predict(input_df_formatted)[0]
            risk_level = interpret_prediction(prediction)
            st.markdown(f"## AI Prediction: **{prediction} ({risk_level})**", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == '__main__':
    main()
