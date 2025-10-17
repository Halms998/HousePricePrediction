import numpy as np
import yaml
import argparse
import joblib
import json
import os  # ← Add this missing import
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()
    
    # Load data and model
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
    model = joblib.load(args.model)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Save metrics
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Evaluation complete. Metrics saved to", args.out)

if __name__ == "__main__":
    main()