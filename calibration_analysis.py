
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

print("CALIBRATION ANALYSIS STARTED")

# 1. Load dataset
data_path = os.path.join("creditcard (3)", "creditcard.csv")

if not os.path.exists(data_path):
    data_path = os.path.join("creditcard.csv (3)", "creditcard.csv")

df = pd.read_csv(data_path)

X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Recreate original train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Fit scaler on training data
scaler = StandardScaler()
scaler.fit(X_train)

X_test_scaled = scaler.transform(X_test)

# 4. Load original model
model = joblib.load("outputs/xgb_fraud_model.pkl")

# 5. Calculate fraud probabilities
probabilities = model.predict_proba(X_test_scaled)[:, 1]

actual = np.asarray(y_test)

# 6. Calculate Brier score
brier = brier_score_loss(actual, probabilities)

# 7. Calculate calibration curve
fraction_positive, mean_predicted = calibration_curve(
    actual,
    probabilities,
    n_bins=10,
    strategy="quantile"
)

# 8. Calculate Expected Calibration Error
bin_edges = np.quantile(probabilities, np.linspace(0, 1, 11))
bin_edges = np.unique(bin_edges)

ece = 0.0

for i in range(len(bin_edges) - 1):
    if i == len(bin_edges) - 2:
        mask = (
            (probabilities >= bin_edges[i]) &
            (probabilities <= bin_edges[i + 1])
        )
    else:
        mask = (
            (probabilities >= bin_edges[i]) &
            (probabilities < bin_edges[i + 1])
        )

    if np.sum(mask) > 0:
        average_confidence = np.mean(probabilities[mask])
        average_accuracy = np.mean(actual[mask])
        ece += (
            np.sum(mask) / len(actual)
        ) * abs(average_confidence - average_accuracy)

# 9. Create output folder
os.makedirs("outputs", exist_ok=True)

# 10. Confidence histogram
plt.figure(figsize=(9, 6))

plt.hist(
    probabilities,
    bins=20,
    edgecolor="black"
)

plt.title("Fraud Probability Confidence Histogram")
plt.xlabel("Predicted Fraud Probability")
plt.ylabel("Number of Transactions")
plt.tight_layout()

plt.savefig("outputs/confidence_histogram.png")
plt.close()

# 11. Reliability diagram
plt.figure(figsize=(8, 6))

plt.plot(
    [0, 1], [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    label="XGBoost Model"
)

plt.title("Model Calibration / Reliability Diagram")
plt.xlabel("Mean Predicted Fraud Probability")
plt.ylabel("Actual Fraud Frequency")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/calibration_reliability_diagram.png")
plt.close()

# 12. Save numerical results
calibration_results = pd.DataFrame({
    "Mean_Predicted_Probability": mean_predicted,
    "Actual_Fraud_Frequency": fraction_positive
})

calibration_results.to_csv(
    "outputs/calibration_results.csv",
    index=False
)

# 13. Display results
print("\n===== CALIBRATION RESULTS =====")
print(f"Brier Score: {brier:.6f}")
print(f"Expected Calibration Error: {ece:.6f}")

print("\n===== CALIBRATION BINS =====")
print(calibration_results.to_string(index=False))

print("\nSaved:")
print("outputs/confidence_histogram.png")
print("outputs/calibration_reliability_diagram.png")
print("outputs/calibration_results.csv")

print("\nCALIBRATION ANALYSIS COMPLETED!")
