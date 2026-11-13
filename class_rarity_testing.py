
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
    roc_auc_score,
    confusion_matrix
)

print("CORRECTED CLASS RARITY TESTING STARTED")

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

# 2. Reproduce original split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Load original model and scaler
model = joblib.load("outputs/xgb_fraud_model.pkl")

scaler = StandardScaler()
scaler.fit(X_train)

X_test_scaled = scaler.transform(X_test)

# 4. Generate original model probabilities
probabilities = model.predict_proba(
    X_test_scaled
)[:, 1]

test_data = X_test.copy()
test_data["Actual_Class"] = y_test.values
test_data["Fraud_Probability"] = probabilities

fraud_data = test_data[
    test_data["Actual_Class"] == 1
]

legitimate_data = test_data[
    test_data["Actual_Class"] == 0
]

print("\nAvailable fraud transactions:", len(fraud_data))
print("Available legitimate transactions:", len(legitimate_data))

# 5. Construct target prevalence scenarios
scenarios = {
    "Original Prevalence": len(fraud_data),
    "Rare Fraud 0.10%": int(
        0.0010 * len(legitimate_data) / (1 - 0.0010)
    ),
    "Extremely Rare Fraud 0.05%": int(
        0.0005 * len(legitimate_data) / (1 - 0.0005)
    )
}

results = []

for scenario_name, fraud_count in scenarios.items():

    if fraud_count > len(fraud_data):
        raise ValueError("Not enough fraud samples")

    if fraud_count == len(fraud_data):
        selected_fraud = fraud_data.copy()
    else:
        selected_fraud = fraud_data.sample(
            n=fraud_count,
            random_state=42
        )

    scenario_data = pd.concat(
        [legitimate_data, selected_fraud]
    )

    actual = scenario_data[
        "Actual_Class"
    ].to_numpy()

    scenario_probabilities = scenario_data[
        "Fraud_Probability"
    ].to_numpy()

    predictions = (
        scenario_probabilities >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predictions,
        labels=[0, 1]
    ).ravel()

    results.append({
        "Scenario": scenario_name,
        "Transactions": len(actual),
        "Fraud_Count": int(np.sum(actual)),
        "Fraud_Prevalence_Percent": (
            np.mean(actual) * 100
        ),
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
            actual, scenario_probabilities
        ),
        "ROC_AUC": roc_auc_score(
            actual, scenario_probabilities
        ),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "Fraud_Alerts": int(tp + fp)
    })

# 6. Display results
results_df = pd.DataFrame(results)

print("\n===== CORRECTED CLASS RARITY RESULTS =====")
print(results_df.to_string(index=False))

# 7. Save results separately
os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/class_rarity_corrected_results.csv",
    index=False
)

# 8. Generate comparison graph
metrics = [
    "Precision",
    "Recall",
    "F1_Score",
    "PR_AUC"
]

results_df.set_index("Scenario")[metrics].plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Fraud Detection Under Increasing Class Rarity")
plt.ylabel("Metric Score")
plt.xlabel("Fraud Prevalence Scenario")
plt.ylim(0, 1.05)
plt.xticks(rotation=10)
plt.tight_layout()

plt.savefig(
    "outputs/class_rarity_corrected_comparison.png",
    dpi=300
)

plt.close()

print("\nSaved:")
print("outputs/class_rarity_corrected_results.csv")
print("outputs/class_rarity_corrected_comparison.png")

print("\nCORRECTED CLASS RARITY TESTING COMPLETED!")
