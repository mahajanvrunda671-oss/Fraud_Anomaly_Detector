
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

# 1. Load dataset
DATA_PATH = "creditcard.csv (3)/creditcard.csv"

if not os.path.exists(DATA_PATH):
    DATA_PATH = "creditcard (3)/creditcard.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Reproduce original train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Recreate original scaler
original_scaler = StandardScaler()
original_scaler.fit(X_train)

X_test_original = original_scaler.transform(X_test)

# 4. Load models and tuned scaler
original_model = joblib.load("outputs/xgb_fraud_model.pkl")
tuned_model = joblib.load("outputs/xgb_fraud_model_tuned.pkl")
tuned_scaler = joblib.load("outputs/xgb_fraud_scaler_tuned.pkl")

X_test_tuned = tuned_scaler.transform(X_test)

# 5. Predictions
original_prob = original_model.predict_proba(X_test_original)[:, 1]
tuned_prob = tuned_model.predict_proba(X_test_tuned)[:, 1]

original_pred = (original_prob >= 0.5).astype(int)
tuned_pred = (tuned_prob >= 0.5).astype(int)

# 6. Evaluation
def evaluate_model(name, y_true, predictions, probabilities):

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Model": name,
        "Precision": precision_score(y_true, predictions),
        "Recall": recall_score(y_true, predictions),
        "F1": f1_score(y_true, predictions),
        "PR_AUC": average_precision_score(y_true, probabilities),
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

results = [
    evaluate_model("Original", y_test, original_pred, original_prob),
    evaluate_model("Tuned", y_test, tuned_pred, tuned_prob)
]

comparison = pd.DataFrame(results)

print("\n===== ORIGINAL VS TUNED MODEL =====")
print(comparison.to_string(index=False))

os.makedirs("outputs", exist_ok=True)
comparison.to_csv(
    "outputs/original_vs_tuned_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")
