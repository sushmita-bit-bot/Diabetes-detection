# Diabetes Prediction Application

This application predicts whether a person has diabetes based on health indicators like BMI, glucose level, age, etc. using the Pima Indians Diabetes Dataset.

## Project Structure

```
diabetes-prediction/
├── backend/
│   ├── app.py                # Flask API server
│   ├── model.py              # ML model training and prediction
│   ├── requirements.txt      # Python dependencies
│   └── diabetes.csv          # Dataset
└── frontend/
    ├── index.html            # Main HTML file
    ├── styles.css            # CSS styles
    └── script.js             # JavaScript for frontend logic
```

## Technologies Used

- **Backend**: Python, Flask, Scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **ML Models**: Logistic Regression, Random Forest, SVM

## How to Run

1. Install backend dependencies:
   ```
   cd backend
   pip install -r requirements.txt
   ```

2. Start the backend server:
   ```
   python app.py
   ```

3. Open the frontend/index.html file in a web browser.