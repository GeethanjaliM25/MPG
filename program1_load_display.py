import pandas as pd

# Load dataset
data = pd.read_csv('data/mpg.csv')

print("🔹 First 5 rows of the dataset:")
print(data.head())

print("\n🔹 Dataset Info:")
print(data.info())

print("\n🔹 Missing Values:")
print(data.isnull().sum())

print("\n✅ Data loaded successfully!")
