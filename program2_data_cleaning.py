import pandas as pd
import numpy as np
import os

data_path = 'data/mpg.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError("Dataset not found! Place 'mpg.csv' inside the 'data/' folder.")

data = pd.read_csv(data_path)

# Replace '?' with NaN in horsepower
data['horsepower'] = data['horsepower'].replace('?', np.nan)
data['horsepower'] = data['horsepower'].astype(float)
data['horsepower'].fillna(data['horsepower'].mean(), inplace=True)

# Drop unnecessary columns
if 'car name' in data.columns:
    data.drop(['car name'], axis=1, inplace=True)

data.to_csv('data/mpg_clean.csv', index=False)
print("✅ Cleaned data saved as 'data/mpg_clean.csv'")
