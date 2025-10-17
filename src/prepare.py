import pandas as pd
import yaml
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    df = pd.read_csv(config['data']['dataset_path'])
    
    # Basic preprocessing for house price data
    target_col = config['data']['target']
    
    # Remove rows with missing target and extremely high prices (outliers)
    df = df.dropna(subset=[target_col])
    df = df[df[target_col] > 0]  # Remove zero or negative prices
    df = df[df[target_col] < df[target_col].quantile(0.99)]  # Remove top 1% outliers
    
    # Handle missing values in features
    numeric_features = config['features']['numeric_features']
    categorical_features = config['features']['categorical_features']
    
    for feature in numeric_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(df[feature].median())
    
    for feature in categorical_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna('Unknown')
    
    # Save processed data
    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, 'house_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Data prepared and saved to {output_path}")
    print(f"Final dataset shape: {df.shape}")

if __name__ == "__main__":
    main()