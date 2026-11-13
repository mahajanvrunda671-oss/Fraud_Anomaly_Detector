
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score
)

# 1. Load dataset
DATA_PATH = "creditcard.csv (3)/creditcard.csv"

if not os.path.exists(DATA_PATH):
    DATA_PATH = "creditcard (3)/creditcard.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Same train-test split
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

# 5. Predict
original_prob = original_model.predict_proba(X_test_original)[:, 1]
tuned_prob = tuned_model.predict_proba(X_test_tuned)[:, 1]

original_pred = (original_prob >= 0.5).astype(int)
tuned_pred = (tuned_prob >= 0.5).astype(int)

os.makedirs("outputs/figures", exist_ok=True)

# GRAPH 1: Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, name, predictions in [
    (axes[0], "Original XGBoost", original_pred),
    (axes[1], "Tuned XGBoost", tuned_pred)
]:
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"]
    )
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(name)

plt.tight_layout()
plt.savefig(
    "outputs/figures/confusion_matrix_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# GRAPH 2: Precision-Recall curve
plt.figure(figsize=(8, 6))

for name, probabilities in [
    ("Original XGBoost", original_prob),
    ("Tuned XGBoost", tuned_prob)
]:
    precision, recall, _ = precision_recall_curve(
        y_test, probabilities
    )
    ap = average_precision_score(y_test, probabilities)

    plt.plot(
        recall,
        precision,
        label=f"{name} (AP = {ap:.4f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve: Original vs Tuned")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/figures/precision_recall_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("\n===== GRAPH GENERATION COMPLETE =====")
print("1. confusion_matrix_comparison.png")
print("2. precision_recall_comparison.png")
print("\nFiles saved in outputs/figures/")
