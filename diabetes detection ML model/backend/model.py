import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Path to save trained models
MODEL_PATH = 'models'

def load_and_preprocess_data():
    """Load and preprocess the Pima Indians Diabetes Dataset."""
    # Check if dataset exists, if not, download it
    dataset_path = 'diabetes.csv'
    if not os.path.exists(dataset_path):
        # URL for Pima Indians Diabetes Dataset
        url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
        column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(url, names=column_names)
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)
    
    # Handle missing values (replace 0 with nan for certain columns)
    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_not_accepted:
        df[column] = df[column].replace(0, np.nan)
        mean = df[column].mean(skipna=True)
        df[column] = df[column].fillna(mean)
    
    # Split features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save the scaler for future predictions
    with open(os.path.join(MODEL_PATH, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train, X_test, y_train, y_test

def train_models():
    """Train and save multiple ML models."""
    # Create models directory if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Train Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Train SVM
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    
    # Save models
    with open(os.path.join(MODEL_PATH, 'logistic_regression.pkl'), 'wb') as f:
        pickle.dump(lr, f)
    
    with open(os.path.join(MODEL_PATH, 'random_forest.pkl'), 'wb') as f:
        pickle.dump(rf, f)
    
    with open(os.path.join(MODEL_PATH, 'svm.pkl'), 'wb') as f:
        pickle.dump(svm, f)
    
    return {
        'logistic_regression': lr_accuracy,
        'random_forest': rf_accuracy,
        'svm': svm_accuracy
    }

def predict_diabetes(input_data, model_name='ensemble'):
    """Make predictions using trained models."""
    # Load the scaler
    with open(os.path.join(MODEL_PATH, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    if model_name == 'logistic_regression':
        # Load Logistic Regression model
        with open(os.path.join(MODEL_PATH, 'logistic_regression.pkl'), 'rb') as f:
            model = pickle.load(f)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
    elif model_name == 'random_forest':
        # Load Random Forest model
        with open(os.path.join(MODEL_PATH, 'random_forest.pkl'), 'rb') as f:
            model = pickle.load(f)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
    elif model_name == 'svm':
        # Load SVM model
        with open(os.path.join(MODEL_PATH, 'svm.pkl'), 'rb') as f:
            model = pickle.load(f)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
    else:  # ensemble - use voting from all models
        # Load all models
        with open(os.path.join(MODEL_PATH, 'logistic_regression.pkl'), 'rb') as f:
            lr = pickle.load(f)
        with open(os.path.join(MODEL_PATH, 'random_forest.pkl'), 'rb') as f:
            rf = pickle.load(f)
        with open(os.path.join(MODEL_PATH, 'svm.pkl'), 'rb') as f:
            svm = pickle.load(f)
        
        # Get predictions from all models
        lr_pred = lr.predict(scaled_data)[0]
        rf_pred = rf.predict(scaled_data)[0]
        svm_pred = svm.predict(scaled_data)[0]
        
        # Ensemble prediction (majority voting)
        predictions = [lr_pred, rf_pred, svm_pred]
        prediction = 1 if sum(predictions) >= 2 else 0
        
        # Average probability
        lr_prob = lr.predict_proba(scaled_data)[0][1]
        rf_prob = rf.predict_proba(scaled_data)[0][1]
        svm_prob = svm.predict_proba(scaled_data)[0][1]
        probability = (lr_prob + rf_prob + svm_prob) / 3
        model_name = 'ensemble'
    
    return {
        'prediction': prediction,
        'probability': probability,
        'model': model_name
    }