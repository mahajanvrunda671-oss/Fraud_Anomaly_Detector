
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, roc_auc_score,
    precision_recall_curve, confusion_matrix
)
from xgboost import XGBClassifier

print("IMPROVED MODEL ADAPTATION STARTED")

# 1. Load dataset
data_path = os.path.join("creditcard (3)", "creditcard.csv")

if not os.path.exists(data_path):
    data_path = os.path.join("creditcard.csv (3)", "creditcard.csv")

df = pd.read_csv(data_path)
X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Create training and untouched test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Split training data into fitting and validation sets
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=21,
    stratify=y_train
)

# 4. Simulate drift consistently
def introduce_drift(data):
    drifted = data.copy()
    drifted["Amount"] = drifted["Amount"] * 1.5 + 10
    drifted["V14"] = drifted["V14"] + 0.5
    return drifted

X_fit_drifted = introduce_drift(X_fit)
X_val_drifted = introduce_drift(X_val)
X_test_drifted = introduce_drift(X_test)

print("Simulated drift in Amount and V14.")

# 5. Train a reference model on clean training data
reference_scaler = StandardScaler()
X_fit_clean_scaled = reference_scaler.fit_transform(X_fit)

scale_weight = (y_fit == 0).sum() / (y_fit == 1).sum()

def create_model():
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=200,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1
    )

reference_model = create_model()
reference_model.fit(X_fit_clean_scaled, y_fit)

# 6. Train adapted model on drifted training data
adapted_scaler = StandardScaler()
X_fit_drifted_scaled = adapted_scaler.fit_transform(X_fit_drifted)

X_val_drifted_scaled = adapted_scaler.transform(X_val_drifted)
X_test_drifted_scaled = adapted_scaler.transform(X_test_drifted)

adapted_model = create_model()

print("Retraining adapted model...")
start_time = time.perf_counter()

adapted_model.fit(X_fit_drifted_scaled, y_fit)

training_time = time.perf_counter() - start_time

# 7. Generate validation probabilities
val_probabilities = adapted_model.predict_proba(
    X_val_drifted_scaled
)[:, 1]

# Select threshold using validation F1 score
precision_values, recall_values, thresholds = precision_recall_curve(
    y_val, val_probabilities
)

f1_values = (
    2 * precision_values[:-1] * recall_values[:-1]
    / (precision_values[:-1] + recall_values[:-1] + 1e-12)
)

best_index = np.argmax(f1_values)
best_threshold = thresholds[best_index]

print(f"\nValidation-selected threshold: {best_threshold:.4f}")
print(f"Validation F1 at selected threshold: {f1_values[best_index]:.4f}")

# 8. Evaluate both models on the same untouched drifted test data
reference_test_scaled = reference_scaler.transform(X_test_drifted)

reference_prob = reference_model.predict_proba(
    reference_test_scaled
)[:, 1]

adapted_prob = adapted_model.predict_proba(
    X_test_drifted_scaled
)[:, 1]

reference_pred = (reference_prob >= 0.5).astype(int)
adapted_default_pred = (adapted_prob >= 0.5).astype(int)
adapted_tuned_pred = (adapted_prob >= best_threshold).astype(int)

actual = np.asarray(y_test)

# 9. Calculate metrics
def calculate_metrics(name, actual, predictions, probabilities):
    return {
        "Model": name,
        "Accuracy": accuracy_score(actual, predictions),
        "Precision": precision_score(actual, predictions, zero_division=0),
        "Recall": recall_score(actual, predictions, zero_division=0),
        "F1_Score": f1_score(actual, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(actual, probabilities),
        "PR_AUC": average_precision_score(actual, probabilities)
    }

results = [
    calculate_metrics(
        "Reference Model (Clean Training)",
        actual, reference_pred, reference_prob
    ),
    calculate_metrics(
        "Adapted Model (Default Threshold)",
        actual, adapted_default_pred, adapted_prob
    ),
    calculate_metrics(
        "Adapted Model (Validation-Tuned Threshold)",
        actual, adapted_tuned_pred, adapted_prob
    )
]

results_df = pd.DataFrame(results)

print("\n===== FINAL TEST RESULTS =====")
print(results_df.to_string(index=False))

# 10. Identify genuinely corrected errors
original_wrong = reference_pred != actual
adapted_correct = adapted_tuned_pred == actual

resolved_mask = original_wrong & adapted_correct

comparison = pd.DataFrame({
    "Actual_Class": actual,
    "Reference_Prediction": reference_pred,
    "Adapted_Prediction": adapted_tuned_pred,
    "Reference_Probability": reference_prob,
    "Adapted_Probability": adapted_prob,
    "Failure_Type": np.where(
        (actual == 0) & (reference_pred == 1), "False Positive",
        np.where(
            (actual == 1) & (reference_pred == 0),
            "False Negative", "No Original Failure"
        )
    ),
    "Resolved_After_Adaptation": resolved_mask
})

resolved_cases = comparison[resolved_mask].copy()

print("\n===== FAILURE RESOLUTION =====")
print("Reference model errors:", int(original_wrong.sum()))
print("Errors corrected by adapted model:", len(resolved_cases))

if len(resolved_cases) > 0:
    print(resolved_cases.head(10).to_string(index=False))
else:
    print("No original errors were corrected in this experiment.")

# 11. Save outputs
os.makedirs("outputs", exist_ok=True)

joblib.dump(adapted_model, "outputs/xgb_fraud_model_adapted.pkl")
joblib.dump(adapted_scaler, "outputs/adapted_scaler.pkl")

results_df.to_csv(
    "outputs/improved_adaptation_results.csv", index=False
)

comparison.to_csv(
    "outputs/all_original_vs_adapted_cases.csv", index=False
)

resolved_cases.to_csv(
    "outputs/resolved_failure_cases.csv", index=False
)

# 12. Save comparison graph
results_df.set_index("Model")[
    ["Precision", "Recall", "F1_Score", "PR_AUC"]
].plot(kind="bar", figsize=(11, 6))

plt.title("Reference vs Adapted Model Performance")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/improved_adaptation_comparison.png")
plt.close()

# 13. Save adaptation details
model_size_mb = (
    os.path.getsize("outputs/xgb_fraud_model_adapted.pkl")
    / (1024 * 1024)
)

print("\n===== ADAPTATION DETAILS =====")
print(f"Retraining time: {training_time:.2f} seconds")
print(f"Adapted model size: {model_size_mb:.2f} MB")
print(f"Selected threshold: {best_threshold:.4f}")

print("\nSaved results in outputs folder.")
print("IMPROVED MODEL ADAPTATION COMPLETED!")
