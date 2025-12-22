import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/mpg_clean.csv')

# Histogram of MPG
plt.figure(figsize=(10,5))
sns.histplot(data['mpg'], kde=True, color='blue')
plt.title("Distribution of MPG")
plt.xlabel("MPG")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Scatterplot Weight vs MPG
plt.figure(figsize=(10,5))
sns.scatterplot(x='weight', y='mpg', data=data, color='green')
plt.title("Weight vs MPG")
plt.tight_layout()
plt.show()

print("✅ Visualization complete! Two graphs displayed.")
