
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

print("FAILURE ANALYSIS STARTED", flush=True)

# 1. Load dataset
file_path = os.path.join("creditcard.csv (3)", "creditcard.csv")

df = pd.read_csv(file_path)
print("Dataset loaded:", df.shape, flush=True)

# 2. Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# 3. Same split as Milestone 2
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Recreate scaler
scaler = StandardScaler()
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Load saved model
model_path = os.path.join("outputs", "xgb_fraud_model.pkl")
model = joblib.load(model_path)

print("XGBoost model loaded successfully!", flush=True)

# 6. Predict
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 7. Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\n===== FAILURE ANALYSIS =====")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# 8. Performance metrics
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred, zero_division=0))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC:", average_precision_score(y_test, y_prob))

# 9. Save incorrect predictions
results = X_test.copy()
results["Actual_Class"] = y_test.to_numpy()
results["Predicted_Class"] = y_pred
results["Fraud_Probability"] = y_prob

false_positives = results[
    (results["Actual_Class"] == 0) &
    (results["Predicted_Class"] == 1)
]

false_negatives = results[
    (results["Actual_Class"] == 1) &
    (results["Predicted_Class"] == 0)
]

os.makedirs("outputs", exist_ok=True)

false_positives.to_csv("outputs/false_positives.csv", index=False)
false_negatives.to_csv("outputs/false_negatives.csv", index=False)

print("\nFalse Positive cases saved:", len(false_positives))
print("False Negative cases saved:", len(false_negatives))

print("\nFAILURE ANALYSIS COMPLETED!")
