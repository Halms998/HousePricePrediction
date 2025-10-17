import pandas as pd
import numpy as np
import yaml
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_csv', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()
    
    # Load config
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    df = pd.read_csv(args.in_csv)
    target_col = config['data']['target']
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Define preprocessing - only use features specified in config
    numeric_features = [f for f in config['features']['numeric_features'] if f in X.columns]
    categorical_features = [f for f in config['features']['categorical_features'] if f in X.columns]
    
    print(f"Using numeric features: {numeric_features}")
    print(f"Using categorical features: {categorical_features}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    # Preprocess features
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save processed features and target
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, 'X_train.npy'), X_train_processed)
    np.save(os.path.join(args.out_dir, 'X_test.npy'), X_test_processed)
    np.save(os.path.join(args.out_dir, 'y_train.npy'), y_train.values)
    np.save(os.path.join(args.out_dir, 'y_test.npy'), y_test.values)
    
    # Save preprocessor for later use in Flask app
    joblib.dump(preprocessor, os.path.join(args.out_dir, 'preprocessor.pkl'))
    
    # Save feature names for reference
    feature_names = numeric_features + categorical_features
    with open(os.path.join(args.out_dir, 'feature_names.txt'), 'w') as f:
        for name in feature_names:
            f.write(name + '\n')
    
    print("Feature engineering complete")
    print(f"Processed training data shape: {X_train_processed.shape}")
    print(f"Processed test data shape: {X_test_processed.shape}")

if __name__ == "__main__":
    main()