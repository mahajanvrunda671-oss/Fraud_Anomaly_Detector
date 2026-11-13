
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

print("BLACK-BOX ADVERSARIAL EVALUATION STARTED")

# --------------------------------------------------
# 1. LOAD DATASET AND MODEL
# --------------------------------------------------

possible_paths = [
    os.path.join("creditcard.csv (3)", "creditcard.csv"),
    os.path.join("creditcard (3)", "creditcard.csv")
]

data_path = next(
    (p for p in possible_paths if os.path.exists(p)),
    None
)

if data_path is None:
    raise FileNotFoundError("Dataset not found")

df = pd.read_csv(data_path)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load("outputs/xgb_fraud_model.pkl")

scaler = StandardScaler()
scaler.fit(X_train)

X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 2. IDENTIFY IMPORTANT FEATURES
# --------------------------------------------------

importance = model.feature_importances_

top_indices = np.argsort(importance)[-5:][::-1]

top_features = X.columns[top_indices].tolist()

print("\nTop 5 Important Features:")
for feature in top_features:
    print(feature)

# --------------------------------------------------
# 3. BASELINE PREDICTIONS
# --------------------------------------------------

baseline_probabilities = model.predict_proba(
    X_test_scaled
)[:, 1]

# --------------------------------------------------
# 4. BLACK-BOX ATTACK SIMULATION
# --------------------------------------------------

# The attacker can query prediction probabilities.
# It does not access model parameters or gradients.
# It tests both directions and selects the change
# that reduces the fraud probability most.

attack_strengths = [0.05, 0.10, 0.20]

results = []

# Baseline evaluation
scenarios = [
    ("Clean Data", X_test_scaled.copy())
]

for strength in attack_strengths:

    attacked_data = X_test_scaled.copy()

    for feature in top_features:

        feature_index = X.columns.get_loc(feature)

        # Try positive and negative perturbations
        positive_candidate = attacked_data.copy()
        negative_candidate = attacked_data.copy()

        positive_candidate[:, feature_index] += strength
        negative_candidate[:, feature_index] -= strength

        positive_probabilities = model.predict_proba(
            positive_candidate
        )[:, 1]

        negative_probabilities = model.predict_proba(
            negative_candidate
        )[:, 1]

        # Choose the candidate that lowers fraud probability
        choose_positive = (
            positive_probabilities < negative_probabilities
        )

        attacked_data[:, feature_index] = np.where(
            choose_positive,
            positive_candidate[:, feature_index],
            negative_candidate[:, feature_index]
        )

    scenarios.append(
        (f"Black-box Attack {strength}", attacked_data)
    )

# --------------------------------------------------
# 5. EVALUATE ALL SCENARIOS
# --------------------------------------------------

for scenario_name, scenario_data in scenarios:

    probabilities = model.predict_proba(
        scenario_data
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    pr_auc = average_precision_score(
        y_test, probabilities
    )

    roc_auc = roc_auc_score(
        y_test, probabilities
    )

    results.append({
        "Scenario": scenario_name,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "Fraud_Alerts": int(predictions.sum()),
        "Mean_Fraud_Probability": float(
            probabilities.mean()
        )
    })

results_df = pd.DataFrame(results)

# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/adversarial_evaluation_results.csv",
    index=False
)

print("\n===== ADVERSARIAL EVALUATION RESULTS =====")
print(results_df.to_string(index=False))

# --------------------------------------------------
# 7. GENERATE ROBUSTNESS GRAPH
# --------------------------------------------------

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

plt.title("Clean vs Black-box Adversarial Performance")
plt.ylabel("Metric Score")
plt.xlabel("Evaluation Scenario")
plt.ylim(0, 1.05)
plt.xticks(rotation=10)
plt.tight_layout()

plt.savefig(
    "outputs/adversarial_robustness_comparison.png",
    dpi=300
)

plt.close()

print("\nSaved:")
print("outputs/adversarial_evaluation_results.csv")
print("outputs/adversarial_robustness_comparison.png")

print("\nBLACK-BOX ADVERSARIAL EVALUATION COMPLETED")
