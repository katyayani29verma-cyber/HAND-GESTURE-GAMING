import pandas as pd

# Load dataset
data = pd.read_csv("gesture_data.csv", header=None)

# Convert label column to string
data.iloc[:, -1] = data.iloc[:, -1].astype(str)

# Remove rows where label is 'nan'
cleaned = data[data.iloc[:, -1] != 'nan']

print("Before cleaning:", data.shape)
print("After cleaning:", cleaned.shape)

# Save cleaned dataset
cleaned.to_csv("gesture_data_clean.csv", index=False, header=False)

print("Cleaned dataset saved as gesture_data_clean.csv")
