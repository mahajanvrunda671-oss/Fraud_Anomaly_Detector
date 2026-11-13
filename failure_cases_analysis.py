
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("FAILURE CASES ANALYSIS STARTED")

# 1. Load dataset
data_path = os.path.join("creditcard.csv (3)", "creditcard.csv")
df = pd.read_csv(data_path)

X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Reproduce original train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Load original model
model = joblib.load("outputs/xgb_fraud_model.pkl")

# 4. Scale data
scaler = StandardScaler()
scaler.fit(X_train)

X_test_scaled = scaler.transform(X_test)

# 5. Predict
probabilities = model.predict_proba(X_test_scaled)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# 6. Identify false positives and false negatives
actual = y_test.to_numpy()

false_positive_indices = np.where(
    (actual == 0) & (predictions == 1)
)[0]

false_negative_indices = np.where(
    (actual == 1) & (predictions == 0)
)[0]

print("False Positives found:", len(false_positive_indices))
print("False Negatives found:", len(false_negative_indices))

# 7. Calculate feature signals
feature_names = X.columns.tolist()
feature_importance = model.feature_importances_

def get_top_features(row):
    weighted_scores = (
        np.abs(row) * feature_importance
    )

    top_indices = np.argsort(weighted_scores)[-3:][::-1]

    return ", ".join(
        feature_names[i] for i in top_indices
    )

# 8. Select up to 5 cases from each category
selected_fp = sorted(
    false_positive_indices,
    key=lambda i: probabilities[i],
    reverse=True
)[:5]

selected_fn = sorted(
    false_negative_indices,
    key=lambda i: probabilities[i]
)[:5]

records = []

# 9. Prepare false positive cases
for i in selected_fp:
    records.append({
        "Case_Type": "False Positive",
        "Actual_Class": int(actual[i]),
        "Predicted_Class": int(predictions[i]),
        "Fraud_Probability": round(float(probabilities[i]), 5),
        "Prediction_Confidence": round(float(probabilities[i]), 5),
        "Top_Feature_Signals": get_top_features(X_test_scaled[i]),
        "Possible_Failure_Reason":
            "Legitimate transaction received a fraud probability above the decision threshold.",
        "Suggested_Mitigation":
            "Review flagged transaction, examine false-positive patterns and tune the decision threshold."
    })

# 10. Prepare false negative cases
for i in selected_fn:
    records.append({
        "Case_Type": "False Negative",
        "Actual_Class": int(actual[i]),
        "Predicted_Class": int(predictions[i]),
        "Fraud_Probability": round(float(probabilities[i]), 5),
        "Prediction_Confidence": round(float(1 - probabilities[i]), 5),
        "Top_Feature_Signals": get_top_features(X_test_scaled[i]),
        "Possible_Failure_Reason":
            "Fraudulent transaction received a fraud probability below the decision threshold.",
        "Suggested_Mitigation":
            "Review missed fraud patterns, improve fraud examples in training and evaluate a lower threshold."
    })

# 11. Save failure case report
failure_df = pd.DataFrame(records)

os.makedirs("outputs", exist_ok=True)

failure_df.to_csv(
    "outputs/failure_cases_detailed.csv",
    index=False
)

print("\n===== SELECTED FAILURE CASES =====")
print(failure_df.to_string(index=False))

print("\nTotal cases saved:", len(failure_df))
print("False Positive examples:", len(selected_fp))
print("False Negative examples:", len(selected_fn))

print("\nSaved: outputs/failure_cases_detailed.csv")
print("\nFAILURE CASES ANALYSIS COMPLETED!")
