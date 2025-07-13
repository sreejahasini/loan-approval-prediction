
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_data():
    # Load the dataset
    data = pd.read_csv("data/LoanApprovalPrediction.csv")

    # Drop Loan_ID (not useful for prediction)
    if 'Loan_ID' in data.columns:
        data.drop(columns=['Loan_ID'], inplace=True)

    # Feature engineering: Total_Income & EMI
    data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    data['EMI'] = data['LoanAmount'] / data['Loan_Amount_Term']

    # Drop ApplicantIncome and CoapplicantIncome after combining
    data.drop(columns=['ApplicantIncome', 'CoapplicantIncome'], inplace=True)

    # Fill missing values
    data.fillna(data.mean(numeric_only=True), inplace=True)

    # Encode categorical variables
    label_enc = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = label_enc.fit_transform(data[col].astype(str))

    return data

def train_and_save_model():
    data = load_data()
    X = data.drop("Loan_Status", axis=1)
    y = data["Loan_Status"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    os.makedirs("saved_model", exist_ok=True)
    joblib.dump(model, "saved_model/model.pkl")
    print("✅ Model trained and saved successfully to 'saved_model/model.pkl'.")

if __name__ == "__main__":
    train_and_save_model()
