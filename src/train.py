import numpy as np
import yaml
import argparse
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_out', type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
    
    # Train model
    if config['train']['model_type'] == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=config['train']['n_estimators'],
            max_depth=config['train']['max_depth'],
            random_state=config['train']['random_state']
        )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Save model
    joblib.dump(model, args.model_out)
    
    # Print metrics
    print("Model training complete. Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()