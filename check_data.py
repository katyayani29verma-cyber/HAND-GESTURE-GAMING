import pandas as pd

data = pd.read_csv("gesture_data.csv", header=None)
print("Shape (rows, columns):", data.shape)
print("First 5 rows:")
print(data.head())