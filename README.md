# 🩺 Diabetes Prediction using Machine Learning

This project predicts whether a person is likely to have diabetes using a machine learning model trained on the Pima Indians Diabetes dataset. It was developed as part of the AIML V Semester Open-Ended Project.

## 📌 Features

- Predicts diabetes based on user inputs.
- Uses a trained machine learning model (`diabetes_model.pkl`).
- Simple and interactive web app interface (likely via Streamlit).
- Clean and user-friendly UI.
- Based on real-world health data.

## Live preview
Follow the link to get the live preview: https://diabetes-prediction-main.streamlit.app/

## 📂 Project Structure

      ├── diabetes_app.py       # Main application script
      ├── diabetes_model.pkl    # Trained ML model 
      ├── diabetes.csv          # Dataset (Pima Indians Diabetes dataset) 


## 🚀 How to Run the Project

### 🔧 Requirements

Make sure you have the following installed:

- Python 3.7+
- pandas
- scikit-learn
- streamlit
- joblib or pickle

### ▶️ Run the App

#### Clone the repository:
      git clone https://github.com/ahhmeddd-11/Diabetes-Prediction.git
      cd diabetes-prediction-aiml-v-sem-project

#### Install dependencies:
      pip install -r requirements.txt
   
#### Run the app:
      streamlit run diabetes_app.py
   
   
#### The project uses the Pima Indians Diabetes Dataset, which includes features such as:
        Glucose
        BloodPressure
        SkinThickness
        Insulin
        BMI
        DiabetesPedigreeFunction
        Age
        
## 🤖 Model
The model is trained using scikit-learn and saved as a .pkl file (diabetes_model.pkl). It uses logistic regression or a similar classification algorithm to predict diabetes.

## 📌 Author
Syed Ahmed Ali – syedahmed4957@gmail.com

## 📝 License
This project is open-source and available under the MIT License.
