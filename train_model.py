import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1. Load dataset
# -----------------------------
# No header in CSV, so header=None
data = pd.read_csv("gesture_data_clean.csv", header=None)

# Features: first 63 columns
X = data.iloc[:, :-1]

# Labels: last column (FORCE string)
y = data.iloc[:, -1].astype(str)


# -----------------------------
# 2. Encode labels
# -----------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("Gesture classes:", list(encoder.classes_))


# -----------------------------
# 3. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# -----------------------------
# 4. Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# -----------------------------
# 5. Evaluate model
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------
# 6. Save model and encoder
# -----------------------------
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\nModel saved as gesture_model.pkl")
print("Label encoder saved as label_encoder.pkl")
