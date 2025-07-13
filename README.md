# loan-approval-prediction
Machine learning project to predict loan approvals using applicant financial and demographic data. Includes data preprocessing, model training, evaluation, and a Streamlit app.
 Loan Approval Prediction

This project predicts whether a loan application will be approved based on user inputs such as income, credit history, employment, and more. It uses machine learning models trained on a public dataset and includes a web-based interface built using Streamlit.

## 🚀 Features
- Data preprocessing and feature engineering
- Multiple ML models (Random Forest, Logistic Regression, etc.)
- Model evaluation and selection
- Streamlit UI for live loan approval prediction
- Project structure with modular code

## 📁 Folder Structure
loan-approval-prediction/
├── data/ # Raw dataset
├── notebook/ # Jupyter notebook (EDA, modeling)
├── src/ # Python script for training model
├── app/ # Streamlit app for prediction
├── saved_model/ # Trained model (.pkl)
├── requirements.txt # Python dependencies
├── README.md # Project overview
└── .gitignore # Files/folders to ignore in Git

## 📊 Tech Stack
- Python
- Pandas, NumPy, Scikit-learn
- Streamlit
- Joblib
2. How to Run:
   1. Clone this repo
2. Install dependencies:  
pip install -r requirements.txt
3. Train model:  
python src/model.py

4. Run the app:  
streamlit run app/streamlit_app.py

Dataset Source:
[GeeksforGeeks: Loan Approval Dataset](https://www.geeksforgeeks.org/)
