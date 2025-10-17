import pandas as pd

# Load your dataset
df = pd.read_csv('data/pakistan_house_price.csv')

# Print column names
print("Column names:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nData types:")
print(df.dtypes)