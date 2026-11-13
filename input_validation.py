
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

print("INPUT VALIDATION STARTED")

# 1. Load dataset
data_path = os.path.join("creditcard.csv (3)", "creditcard.csv")
df = pd.read_csv(data_path)

X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Load original model
model = joblib.load("outputs/xgb_fraud_model.pkl")

# 4. Fit scaler on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Baseline evaluation
baseline_probabilities = model.predict_proba(X_test_scaled)[:, 1]
baseline_predictions = (baseline_probabilities >= 0.5).astype(int)


def show_metrics(name, actual, predictions, probabilities):
    print(f"\n===== {name} =====")
    print("Accuracy:", round(accuracy_score(actual, predictions), 5))
    print("Precision:", round(precision_score(actual, predictions, zero_division=0), 5))
    print("Recall:", round(recall_score(actual, predictions, zero_division=0), 5))
    print("F1 Score:", round(f1_score(actual, predictions, zero_division=0), 5))
    print("PR-AUC:", round(average_precision_score(actual, probabilities), 5))


show_metrics(
    "BASELINE - CLEAN DATA",
    y_test,
    baseline_predictions,
    baseline_probabilities
)

# 6. Create test data with simulated invalid values
X_corrupted = X_test.copy()

# Introduce missing values
X_corrupted.iloc[0:100, 0] = np.nan

# Introduce infinite values
X_corrupted.iloc[100:200, 1] = np.inf

# Introduce extreme values
X_corrupted.iloc[200:300, 2] = 1e10

# 7. Detect invalid values
print("\n===== INPUT VALIDATION REPORT =====")

missing_count = X_corrupted.isna().sum().sum()
infinite_count = np.isinf(X_corrupted.to_numpy()).sum()

print("Missing values:", missing_count)
print("Infinite values:", infinite_count)

# 8. Sanitize input using training-set statistics
X_sanitized = X_corrupted.replace(
    [np.inf, -np.inf],
    np.nan
)

# Replace invalid values using training medians
# Replace invalid values using training medians
training_medians = X_train.median()
X_sanitized = X_sanitized.fillna(training_medians)

# Detect and clip extreme finite values using training percentiles
lower_limits = X_train.quantile(0.01)
upper_limits = X_train.quantile(0.99)

extreme_before = (
    (X_sanitized < lower_limits) |
    (X_sanitized > upper_limits)
).sum().sum()

X_sanitized = X_sanitized.clip(
    lower=lower_limits,
    upper=upper_limits,
    axis=1
)

extreme_after = (
    (X_sanitized < lower_limits) |
    (X_sanitized > upper_limits)
).sum().sum()

print("\n===== EXTREME VALUE HANDLING =====")
print("Extreme values before clipping:", extreme_before)
print("Extreme values after clipping:", extreme_after)

# Check remaining invalid values
remaining_missing = X_sanitized.isna().sum().sum()
remaining_infinite = np.isinf(X_sanitized.to_numpy()).sum()

print("\n===== AFTER SANITIZATION =====")
print("Remaining missing values:", remaining_missing)
print("Remaining infinite values:", remaining_infinite)

# 9. Predict after sanitization
X_sanitized_scaled = scaler.transform(X_sanitized)

sanitized_probabilities = model.predict_proba(
    X_sanitized_scaled
)[:, 1]

sanitized_predictions = (
    sanitized_probabilities >= 0.5
).astype(int)

show_metrics(
    "AFTER INPUT SANITIZATION",
    y_test,
    sanitized_predictions,
    sanitized_probabilities
)

# 10. Save validation report
report = pd.DataFrame({
    "Check": [
        "Missing values before sanitization",
        "Infinite values before sanitization",
        "Missing values after sanitization",
        "Infinite values after sanitization"
    ],
    "Count": [
        missing_count,
        infinite_count,
        remaining_missing,
        remaining_infinite
    ]
})

os.makedirs("outputs", exist_ok=True)

report.to_csv(
    "outputs/input_validation_report.csv",
    index=False
)

print("\nSaved: outputs/input_validation_report.csv")
print("\nINPUT VALIDATION COMPLETED!")
