
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

print("ROBUSTNESS TESTING STARTED", flush=True)

# ------------------------------------------------
# 1. Load dataset and recreate original test split
# ------------------------------------------------

file_path = os.path.join("creditcard.csv (3)", "creditcard.csv")
df = pd.read_csv(file_path)

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
scaler.fit(X_train)

X_test_scaled = scaler.transform(X_test)
y_test = y_test.to_numpy()

# ------------------------------------------------
# 2. Load original trained XGBoost model
# ------------------------------------------------

model = joblib.load("outputs/xgb_fraud_model.pkl")

os.makedirs("outputs", exist_ok=True)

results = []

def evaluate_model(X_input, test_name, severity):
    probabilities = model.predict_proba(X_input)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "Test": test_name,
        "Severity": severity,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "PR_AUC": average_precision_score(y_test, probabilities)
    }

    results.append(metrics)

    print(
        f"{test_name} | Severity: {severity} | "
        f"Recall: {metrics['Recall']:.4f} | "
        f"F1: {metrics['F1']:.4f} | "
        f"PR-AUC: {metrics['PR_AUC']:.4f}",
        flush=True
    )

    return metrics


# ------------------------------------------------
# 3. Clean baseline
# ------------------------------------------------

print("\nTesting clean data...")
evaluate_model(X_test_scaled, "Clean", 0)


# ------------------------------------------------
# 4. Gaussian noise testing
# ------------------------------------------------

print("\nTesting Gaussian noise...")

rng = np.random.default_rng(42)

for sigma in [0.05, 0.10, 0.20]:

    noisy_data = (
        X_test_scaled +
        rng.normal(
            loc=0,
            scale=sigma,
            size=X_test_scaled.shape
        )
    )

    evaluate_model(noisy_data, "Gaussian Noise", sigma)


# ------------------------------------------------
# 5. Random feature dropout
# ------------------------------------------------

print("\nTesting feature dropout...")

for dropout_rate in [0.05, 0.10, 0.20]:

    dropped_data = X_test_scaled.copy()

    mask = rng.random(dropped_data.shape) < dropout_rate

    # Zero represents the training mean after standardization
    dropped_data[mask] = 0

    evaluate_model(
        dropped_data,
        "Feature Dropout",
        dropout_rate
    )


# ------------------------------------------------
# 6. Adversarial-style feature perturbation
# ------------------------------------------------

print("\nTesting adversarial-style perturbations...")

# Select five most important features
importance = model.feature_importances_
top_features = np.argsort(importance)[::-1][:5]

print("Selected feature indices:", top_features)

for epsilon in [0.05, 0.10, 0.20]:

    adversarial_data = X_test_scaled.copy()

    # Greedy black-box evasion:
    # For each selected feature, test +epsilon and -epsilon.
    # Keep the direction that reduces predicted fraud probability.

    for feature_idx in top_features:

        plus_data = adversarial_data.copy()
        minus_data = adversarial_data.copy()

        plus_data[:, feature_idx] += epsilon
        minus_data[:, feature_idx] -= epsilon

        plus_scores = model.predict_proba(plus_data)[:, 1]
        minus_scores = model.predict_proba(minus_data)[:, 1]

        choose_plus = plus_scores < minus_scores

        adversarial_data[choose_plus, feature_idx] += epsilon
        adversarial_data[~choose_plus, feature_idx] -= epsilon

    evaluate_model(
        adversarial_data,
        "Adversarial Perturbation",
        epsilon
    )


# ------------------------------------------------
# 7. Save all results
# ------------------------------------------------

results_df = pd.DataFrame(results)

results_path = "outputs/robustness_results.csv"
results_df.to_csv(results_path, index=False)

print("\n===== ROBUSTNESS RESULTS =====")
print(results_df.to_string(index=False))


# ------------------------------------------------
# 8. Plot robustness curves
# ------------------------------------------------

for metric in ["Recall", "F1", "PR_AUC"]:

    plt.figure(figsize=(9, 5))

    for test_name in [
        "Gaussian Noise",
        "Feature Dropout",
        "Adversarial Perturbation"
    ]:

        subset = results_df[
            results_df["Test"] == test_name
        ]

        plt.plot(
            subset["Severity"],
            subset[metric],
            marker="o",
            label=test_name
        )

    plt.axhline(
        y=results_df.loc[
            results_df["Test"] == "Clean", metric
        ].iloc[0],
        linestyle="--",
        label="Clean Baseline"
    )

    plt.xlabel("Perturbation Severity")
    plt.ylabel(metric)
    plt.title(f"Robustness Curve - {metric}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(
        f"outputs/robustness_{metric.lower()}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


print("\nAll robustness experiments completed!")
print("Results saved in outputs folder.")
