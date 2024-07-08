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
            value = st.number_input(f"{feature}", min_value=0, max_value=10000000, value=50000)
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

# Function to generate HTML for the slider
def get_slider_html(prediction):
    if prediction == 1:
        position = 0
    elif prediction == 2:
        position = 50
    else:
        position = 100
        
    html = f"""
    <div style="width: 100%; display: flex; justify-content: center; align-items: center;">
        <div style="width: 80%; height: 30px; position: relative; background: linear-gradient(90deg, green 0%, yellow 50%, red 100%); border-radius: 5px;">
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
    # st.image('images/logo.png', width=400)
    st.markdown("<h1 style='text-align: center;'>CrediSense</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enter the details to predict the credit risk using the AI model.</p>", unsafe_allow_html=True)
    
    input_df = user_input_features()
    
    st.write("## User Input Summary:")
    st.table(input_df)
    
    input_data = {row['Feature']: row['Value'] for index, row in input_df.iterrows()}
    input_df_formatted = pd.DataFrame(input_data, index=[0])
    
    if xgb_model is not None and st.button("Predict"):
        try:
            prediction = xgb_model.predict(input_df_formatted)[0]
            risk_level = interpret_prediction(prediction)
            slider_html = get_slider_html(prediction)
            st.markdown(f"<h2 style='text-align: center;'>AI Prediction: {prediction} ({risk_level})</h2>", unsafe_allow_html=True)
            st.markdown(slider_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == '__main__':
    main()
