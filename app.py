import streamlit as st
import pandas as pd
import traceback
import joblib
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

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
xgb_model = load_model('models/model.joblib')

# Define the input features
features = [
    'Age_Oldest_TL', 'enq_L3m', 'enq_L6m', 'time_since_recent_enq', 
    'num_std_12mts', 'num_std', 'num_std_6mts', 'enq_L12m', 
    'pct_PL_enq_L6m_of_ever', 'pct_PL_enq_L6m_of_L12m', 
    'PL_enq_L6m', 'PL_enq_L12m'
]

# Feature explanations
explanations = {
    'Age_Oldest_TL': 'The age of the oldest trade line (credit account) in years.',
    'enq_L3m': 'The number of credit enquiries made in the last 3 months.',
    'enq_L6m': 'The number of credit enquiries made in the last 6 months.',
    'time_since_recent_enq': 'The time since the most recent credit enquiry, measured in months.',
    'num_std_12mts': 'The number of standard deviations of trade lines in the last 12 months. This measures the variability or diversity in the number of trade lines.',
    'num_std': 'The number of standard deviations of trade lines. This measures the variability or diversity in the number of trade lines.',
    'num_std_6mts': 'The number of standard deviations of trade lines in the last 6 months. This measures the variability or diversity in the number of trade lines in the recent 6 months.',
    'enq_L12m': 'The number of credit enquiries made in the last 12 months.',
    'pct_PL_enq_L6m_of_ever': 'The percentage of personal loan enquiries made in the last 6 months compared to all enquiries ever made.',
    'pct_PL_enq_L6m_of_L12m': 'The percentage of personal loan enquiries made in the last 6 months compared to the last 12 months.',
    'PL_enq_L6m': 'The number of personal loan enquiries made in the last 6 months.',
    'PL_enq_L12m': 'The number of personal loan enquiries made in the last 12 months.'
}

# Function to get user input
def user_input_features():
    input_data = []

    st.header("Input Features")
    for feature, explanation in explanations.items():
        st.write(f'#### {explanation}')
        value = st.slider(feature, 0, 100, 1, key=feature)
        input_data.append((feature, value))

    features_df = pd.DataFrame(input_data, columns=['Feature', 'Value'])
    return features_df

# Function to generate polynomial features
def generate_polynomial_features(df):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(df)
    return poly_features

# Function to interpret prediction results
def interpret_prediction(prediction):
    mapping = {
        0: "You have a very low risk of default. You are an ideal candidate for a loan.",
        1: "You have a low risk of default. You are a good candidate for a loan.",
        2: "You have a moderate risk of default. There are some concerns, but you may still qualify for a loan.",
        3: "You have a high risk of default. There are significant concerns about your ability to repay the loan."
    }
    return mapping.get(prediction, "Unknown Risk Level")

# Function to generate HTML for the slider
def get_slider_html(prediction):
    if prediction == 0:
        position = 0
    elif prediction == 1:
        position = 33
    elif prediction == 2:
        position = 66
    else:
        position = 100

    html = f"""
    <div style="width: 100%; display: flex; justify-content: center; align-items: center;">
        <div style="width: 80%; height: 30px; position: relative; background: linear-gradient(90deg, green 0%, yellow 33%, orange 66%, red 100%); border-radius: 5px;">
            <div style="position: absolute; left: {position}%; top: -10px; transform: translateX(-50%);">
                <div style="width: 10px; height: 40px; background-color: black; border-radius: 5px;"></div>
            </div>
        </div>
    </div>
    """
    return html

# Main Streamlit app
def main():
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image('images/logo.png')
    st.markdown("<h1 style='text-align: center;'>CrediSense</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enter the details to predict the credit risk using the AI model.</p>", unsafe_allow_html=True)
    
    input_df = user_input_features()
    
    input_data = {row['Feature']: row['Value'] for index, row in input_df.iterrows()}
    input_df_formatted = pd.DataFrame(input_data, index=[0])
    
    # Generate polynomial features
    poly_features = generate_polynomial_features(input_df_formatted)
    
    if xgb_model is not None and st.button("Predict"):
        try:
            prediction = xgb_model.predict(poly_features)[0]
            risk_level = interpret_prediction(prediction)
            slider_html = get_slider_html(prediction)
            st.markdown(f"<h2 style='text-align: center;'>AI Prediction: {risk_level}</h2>", unsafe_allow_html=True)
            st.markdown(slider_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == '__main__':
    main()
