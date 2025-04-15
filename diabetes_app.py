# diabetes_app_decision_tree_no_pregnancy.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load Dataset
def load_data():
    data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    df = pd.read_csv(data_url, header=None, names=column_names)
    return df

# Preprocess the dataset (Exclude Pregnancies feature)
def preprocess_data(df):
    X = df.iloc[:, 1:-1].values  # Exclude the first column (Pregnancies)
    y = df.iloc[:, -1].values    # Target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Standardize features
    return X_scaled, y, scaler

# Train the Decision Tree model
def train_decision_tree(X_train, y_train, max_depth=None):
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    return dt

# Streamlit app
def main():
    st.title("Diabetes Prediction using Decision Tree")
    st.write("Enter the following details to predict diabetes:")

    # User inputs (excluding Pregnancies)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

    # Load and preprocess data
    df = load_data()
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Decision Tree model
    dt = train_decision_tree(X_train, y_train, max_depth=5)

    # Make predictions
    if st.button("Predict"):
        input_data = np.array([[glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = dt.predict(input_scaled)[0]
        prediction_prob = dt.predict_proba(input_scaled)[0]

        if prediction == 1:
            st.error(f"The patient is likely to have diabetes with a probability of {prediction_prob[1]:.2f}.")
        else:
            st.success(f"The patient is unlikely to have diabetes with a probability of {prediction_prob[0]:.2f}.")

    # Display model accuracy
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy on Test Data: {accuracy:.2f}")

if __name__ == "__main__":
    main()
