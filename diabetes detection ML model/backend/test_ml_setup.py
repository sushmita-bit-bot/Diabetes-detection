import os
import sys

def test_ml_setup():
    """Test the ML setup by loading data and training models."""
    print("Testing ML setup...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    try:
        # Try importing required packages
        print("Checking required packages...")
        import pandas as pd
        print(f"pandas version: {pd.__version__}")
        
        import numpy as np
        print(f"numpy version: {np.__version__}")
        
        import sklearn
        print(f"scikit-learn version: {sklearn.__version__}")
        
        import flask
        print(f"flask version: {flask.__version__}")
        
        print("All required packages are installed!")
    except ImportError as e:
        print(f"Missing package: {str(e)}")
        print("Please install the required packages using: pip install -r requirements.txt")
        return False
    
    # Check if dataset exists
    dataset_path = 'diabetes.csv'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. It will be downloaded when running the application.")
    else:
        print(f"Dataset found at {dataset_path}.")
    
    print("ML setup test completed!")
    return True

if __name__ == "__main__":
    test_ml_setup()