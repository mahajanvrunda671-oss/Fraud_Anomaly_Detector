
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

print("OOD TESTING STARTED")

# 1. Load dataset
possible_paths = [
    os.path.join("creditcard.csv (3)", "creditcard.csv"),
    os.path.join("creditcard (3)", "creditcard.csv")
]

data_path = next(
    (p for p in possible_paths if os.path.exists(p)),
    None
)

if data_path is None:
    raise FileNotFoundError("creditcard.csv not found")

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

# 3. Load model and scaler
model = joblib.load("outputs/xgb_fraud_model.pkl")

scaler = StandardScaler()
scaler.fit(X_train)

X_test_scaled = scaler.transform(X_test)

# 4. Generate OOD datasets
rng = np.random.default_rng(42)

X_ood_moderate = X_test.copy()
X_ood_moderate["Amount"] = (
    X_ood_moderate["Amount"] * 2 + 20
)
X_ood_moderate["V14"] = (
    X_ood_moderate["V14"] + 1.0
)

X_ood_severe = X_test.copy()
X_ood_severe["Amount"] = (
    X_ood_severe["Amount"] * 5 + 100
)
X_ood_severe["V14"] = (
    X_ood_severe["V14"] + 3.0
)
X_ood_severe["V10"] = (
    X_ood_severe["V10"] - 3.0
)

# 5. Scale OOD datasets using original training scaler
X_ood_moderate_scaled = scaler.transform(X_ood_moderate)
X_ood_severe_scaled = scaler.transform(X_ood_severe)

# 6. Define evaluation function
def evaluate_dataset(name, X_scaled, actual):
    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "Dataset": name,
        "Precision": precision_score(
            actual, predictions, zero_division=0
        ),
        "Recall": recall_score(
            actual, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            actual, predictions, zero_division=0
        ),
        "PR_AUC": average_precision_score(
            actual, probabilities
        ),
        "ROC_AUC": roc_auc_score(
            actual, probabilities
        ),
        "Mean_Fraud_Probability": np.mean(probabilities)
    }

# 7. Evaluate clean and shifted datasets
results = []

results.append(
    evaluate_dataset(
        "Clean Test Data",
        X_test_scaled,
        y_test
    )
)

results.append(
    evaluate_dataset(
        "Moderate OOD",
        X_ood_moderate_scaled,
        y_test
    )
)

results.append(
    evaluate_dataset(
        "Severe OOD",
        X_ood_severe_scaled,
        y_test
    )
)

results_df = pd.DataFrame(results)

print("\n===== OOD PERFORMANCE RESULTS =====")
print(results_df.to_string(index=False))

# 8. Calculate distribution shift
def calculate_shift(reference, shifted):
    ref_mean = reference.mean(axis=0)
    shifted_mean = shifted.mean(axis=0)

    ref_std = reference.std(axis=0)
    shifted_std = shifted.std(axis=0)

    shift_score = np.mean(
        np.abs(shifted_mean - ref_mean) /
        (ref_std + 1e-8)
    )

    return shift_score

shift_results = pd.DataFrame({
    "Dataset": ["Moderate OOD", "Severe OOD"],
    "Mean_Standardized_Shift": [
        calculate_shift(X_test_scaled, X_ood_moderate_scaled),
        calculate_shift(X_test_scaled, X_ood_severe_scaled)
    ]
})

print("\n===== OOD DISTRIBUTION SHIFT =====")
print(shift_results.to_string(index=False))

# 9. Save results
os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/ood_performance_results.csv",
    index=False
)

shift_results.to_csv(
    "outputs/ood_distribution_shift.csv",
    index=False
)

# 10. Generate performance comparison graph
metrics = ["Precision", "Recall", "F1_Score", "PR_AUC"]

results_df.set_index("Dataset")[metrics].plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Fraud Detection Performance Under OOD Conditions")
plt.ylabel("Metric Score")
plt.xlabel("Dataset Condition")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower left")
plt.tight_layout()

plt.savefig(
    "outputs/ood_performance_comparison.png",
    dpi=300
)

plt.close()

print("\nSaved:")
print("outputs/ood_performance_results.csv")
print("outputs/ood_distribution_shift.csv")
print("outputs/ood_performance_comparison.png")

print("\nOOD TESTING COMPLETED!")
