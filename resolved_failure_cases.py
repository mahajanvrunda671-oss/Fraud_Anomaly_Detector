
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

print("RESOLVED FAILURE CASE ANALYSIS STARTED")

# 1. Load dataset
DATA_PATH = os.path.join("creditcard.csv (3)", "creditcard.csv")
df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Recreate the original train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Fit scaler on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Load original and adapted models
original_model = joblib.load("outputs/xgb_fraud_model.pkl")
adapted_model = joblib.load("outputs/xgb_fraud_model_adapted.pkl")

# 5. Predict on exactly the same test transactions
original_prob = original_model.predict_proba(X_test_scaled)[:, 1]
adapted_prob = adapted_model.predict_proba(X_test_scaled)[:, 1]

original_pred = (original_prob >= 0.5).astype(int)
adapted_pred = (adapted_prob >= 0.5).astype(int)

actual = np.asarray(y_test)

# 6. Identify errors corrected by adaptation
original_wrong = original_pred != actual
adapted_correct = adapted_pred == actual

resolved_mask = original_wrong & adapted_correct

results = pd.DataFrame({
    "Actual_Class": actual,
    "Original_Prediction": original_pred,
    "Adapted_Prediction": adapted_pred,
    "Original_Probability": original_prob,
    "Adapted_Probability": adapted_prob,
    "Original_Status": np.where(original_wrong, "Incorrect", "Correct"),
    "Adapted_Status": np.where(adapted_correct, "Correct", "Incorrect"),
    "Failure_Type": np.where(
        (actual == 0) & (original_pred == 1), "False Positive",
        np.where(
            (actual == 1) & (original_pred == 0),
            "False Negative", "Not an original failure"
        )
    ),
    "Resolved_After_Adaptation": resolved_mask
})

# 7. Keep only genuinely resolved original failures
resolved_cases = results[resolved_mask].copy()

os.makedirs("outputs", exist_ok=True)

results.to_csv(
    "outputs/all_original_vs_adapted_cases.csv",
    index=False
)

resolved_cases.to_csv(
    "outputs/resolved_failure_cases.csv",
    index=False
)

# 8. Display results
tn1, fp1, fn1, tp1 = confusion_matrix(
    actual, original_pred, labels=[0, 1]
).ravel()

tn2, fp2, fn2, tp2 = confusion_matrix(
    actual, adapted_pred, labels=[0, 1]
).ravel()

print("\n===== ORIGINAL MODEL =====")
print("False Positives:", fp1)
print("False Negatives:", fn1)

print("\n===== ADAPTED MODEL =====")
print("False Positives:", fp2)
print("False Negatives:", fn2)

print("\n===== FAILURE RESOLUTION =====")
print("Total resolved cases:", len(resolved_cases))

print(resolved_cases.head(10).to_string(index=False))

print("\nSaved:")
print("outputs/all_original_vs_adapted_cases.csv")
print("outputs/resolved_failure_cases.csv")

print("\nANALYSIS COMPLETED")
print("MY SCRIPT IS RUNNING")