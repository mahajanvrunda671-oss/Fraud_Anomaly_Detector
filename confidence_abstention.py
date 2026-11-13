
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

print("CONFIDENCE-BASED ABSTENTION STARTED")

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

# 3. Load model and prepare data
model = joblib.load("outputs/xgb_fraud_model.pkl")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Get fraud probabilities
probabilities = model.predict_proba(X_test_scaled)[:, 1]

# 5. Normal predictions
normal_predictions = (probabilities >= 0.5).astype(int)

# 6. Confidence-based abstention
# If probability is between 0.20 and 0.80,
# send transaction for manual review.

lower_threshold = 0.20
upper_threshold = 0.80

decisions = []

for probability in probabilities:

    if probability >= upper_threshold:
        decisions.append("FRAUD ALERT")

    elif probability <= lower_threshold:
        decisions.append("NORMAL")

    else:
        decisions.append("MANUAL REVIEW")

decisions = np.array(decisions)

# 7. Calculate review statistics
review_mask = decisions == "MANUAL REVIEW"
fraud_mask = decisions == "FRAUD ALERT"
normal_mask = decisions == "NORMAL"

review_count = review_mask.sum()
fraud_count = fraud_mask.sum()
normal_count = normal_mask.sum()

print("\n===== ABSTENTION SUMMARY =====")
print("Total transactions:", len(decisions))
print("Normal decisions:", normal_count)
print("Fraud alerts:", fraud_count)
print("Manual reviews:", review_count)

print(
    "Manual review percentage:",
    round(review_count / len(decisions) * 100, 2),
    "%"
)

# 8. Evaluate automatic decisions only
automatic_mask = ~review_mask

if automatic_mask.sum() > 0:

    actual_auto = y_test.to_numpy()[automatic_mask]
    predicted_auto = (
        decisions[automatic_mask] == "FRAUD ALERT"
    ).astype(int)

    print("\n===== AUTOMATIC DECISION PERFORMANCE =====")
    print(
        "Automatic decision coverage:",
        round(automatic_mask.mean() * 100, 2),
        "%"
    )

    print(
        "Accuracy:",
        round(
            accuracy_score(actual_auto, predicted_auto),
            5
        )
    )

    print(
        "Precision:",
        round(
            precision_score(
                actual_auto,
                predicted_auto,
                zero_division=0
            ),
            5
        )
    )

    print(
        "Recall:",
        round(
            recall_score(
                actual_auto,
                predicted_auto,
                zero_division=0
            ),
            5
        )
    )

    print(
        "F1 Score:",
        round(
            f1_score(
                actual_auto,
                predicted_auto,
                zero_division=0
            ),
            5
        )
    )

# 9. Overall metrics at standard threshold
print("\n===== STANDARD MODEL PERFORMANCE =====")
print(
    "Accuracy:",
    round(accuracy_score(y_test, normal_predictions), 5)
)
print(
    "Recall:",
    round(recall_score(y_test, normal_predictions), 5)
)
print(
    "PR-AUC:",
    round(average_precision_score(y_test, probabilities), 5)
)

# 10. Save transaction-level decisions
output = pd.DataFrame({
    "Actual_Class": y_test.to_numpy(),
    "Fraud_Probability": probabilities,
    "Standard_Prediction": normal_predictions,
    "Abstention_Decision": decisions
})

os.makedirs("outputs", exist_ok=True)

output.to_csv(
    "outputs/confidence_abstention_results.csv",
    index=False
)

# Save summary
summary = pd.DataFrame({
    "Decision": [
        "NORMAL",
        "FRAUD ALERT",
        "MANUAL REVIEW"
    ],
    "Count": [
        normal_count,
        fraud_count,
        review_count
    ]
})

summary.to_csv(
    "outputs/abstention_summary.csv",
    index=False
)

print("\nSaved:")
print("- outputs/confidence_abstention_results.csv")
print("- outputs/abstention_summary.csv")

print("\nCONFIDENCE-BASED ABSTENTION COMPLETED!")
