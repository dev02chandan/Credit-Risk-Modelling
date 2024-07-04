# Credit Risk Prediction App

This Streamlit app predicts credit risk using an XGBoost model. Users can input various financial and personal details to get a credit risk prediction.

## Features

- Interactive web interface built with Streamlit
- Credit risk prediction using an XGBoost model
- Input validation and error handling

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have the `best_xgb_model.joblib` file in the same directory as the app.
2. Run the Streamlit app:
   ```
   python -m streamlit run app.py
   ```
3. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`)
4. Enter the required information and click "Predict with XGBoost" to get a credit risk prediction

## Input Features

- Age of Oldest Trade Line
- Number of Enquiries in Last 3 Months
- Time Since Recent Enquiry
- Number of Standard Deviation
- Time Since Recent Payment
- Time with Current Employer
- Net Monthly Income
- Age of Newest Trade Line
- Number of Standard Deviation in Last 6 Months
- Total Enquiries
- Percentage of Personal Loan Enquiries in Last 6 Months
- Recent Level of Delinquency

## Dependencies

- streamlit
- pandas
- joblib
- numpy
- xgboost

## Note

This app is for educational purposes only and should not be used for actual credit risk assessment without proper validation and regulatory compliance.
