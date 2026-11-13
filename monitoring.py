
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

print("MONITORING SYSTEM STARTED", flush=True)

# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

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

print("Training data:", X_train.shape)
print("Monitoring data:", X_test.shape)

# -----------------------------------------
# 2. Create monitoring batch
# -----------------------------------------

reference_data = X_train.copy()

current_data = X_test.sample(
    n=min(10000, len(X_test)),
    random_state=42
).copy()

# Simulate distribution drift in selected features
# This is a controlled experiment, not real production drift.

current_data["Amount"] = (
    current_data["Amount"] * 1.5 + 10
)

current_data["V14"] = (
    current_data["V14"] + 0.5
)

print("\nSimulated drift introduced in Amount and V14.")

# -----------------------------------------
# 3. PSI calculation function
# -----------------------------------------

def calculate_psi(reference, current, bins=10):

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)

    breakpoints = np.unique(
        np.quantile(reference, quantiles)
    )

    if len(breakpoints) < 2:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(
        reference, bins=breakpoints
    )

    cur_counts, _ = np.histogram(
        current, bins=breakpoints
    )

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    epsilon = 0.0001

    ref_pct = np.maximum(ref_pct, epsilon)
    cur_pct = np.maximum(cur_pct, epsilon)

    psi = np.sum(
        (cur_pct - ref_pct) *
        np.log(cur_pct / ref_pct)
    )

    return float(psi)


# -----------------------------------------
# 4. Monitor all feature distributions
# -----------------------------------------

monitoring_results = []

for column in X_train.columns:

    psi_value = calculate_psi(
        reference_data[column],
        current_data[column]
    )

    missing_rate = current_data[column].isnull().mean()

    if pd.isna(psi_value):
        status = "CHECK DATA"
    elif psi_value < 0.10:
        status = "STABLE"
    elif psi_value < 0.20:
        status = "MODERATE DRIFT"
    else:
        status = "HIGH DRIFT"

    monitoring_results.append({
        "Feature": column,
        "PSI": psi_value,
        "Current_Missing_Rate": missing_rate,
        "Status": status
    })

results_df = pd.DataFrame(monitoring_results)

results_df = results_df.sort_values(
    by="PSI", ascending=False
)

# -----------------------------------------
# 5. Print monitoring report
# -----------------------------------------

print("\n========== DATA DRIFT REPORT ==========")
print(results_df.to_string(index=False))

print("\n========== DRIFT SUMMARY ==========")

print(
    "Stable features:",
    (results_df["Status"] == "STABLE").sum()
)

print(
    "Moderate drift:",
    (results_df["Status"] == "MODERATE DRIFT").sum()
)

print(
    "High drift:",
    (results_df["Status"] == "HIGH DRIFT").sum()
)

# -----------------------------------------
# 6. Save monitoring results
# -----------------------------------------

os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/monitoring_drift_results.csv",
    index=False
)

# -----------------------------------------
# 7. Create drift dashboard
# -----------------------------------------

top_features = results_df.head(10).sort_values(
    by="PSI", ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["PSI"]
)

plt.axvline(
    0.10,
    linestyle="--",
    label="Moderate drift threshold"
)

plt.axvline(
    0.20,
    linestyle="--",
    label="High drift threshold"
)

plt.xlabel("Population Stability Index (PSI)")
plt.ylabel("Features")
plt.title("Data Drift Monitoring Dashboard")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/drift_dashboard.png",
    dpi=300
)

plt.close()

# -----------------------------------------
# 8. Simulated delayed-label model monitoring
# -----------------------------------------

model = joblib.load("outputs/xgb_fraud_model.pkl")

scaler = StandardScaler()
scaler.fit(X_train)

current_scaled = scaler.transform(current_data)

current_labels = y_test.loc[current_data.index]

probabilities = model.predict_proba(current_scaled)[:, 1]

print("\n========== MODEL MONITORING ==========")

print(
    "Current batch ROC-AUC:",
    round(
        roc_auc_score(current_labels, probabilities),
        4
    )
)

print(
    "Current batch PR-AUC:",
    round(
        average_precision_score(current_labels, probabilities),
        4
    )
)

print("\nMonitoring results saved successfully.")
print("Dashboard saved: outputs/drift_dashboard.png")
print("MONITORING COMPLETED!")
