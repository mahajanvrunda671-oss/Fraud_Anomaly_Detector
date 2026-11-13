
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("ENHANCED SYSTEM SAFETY CHECKS STARTED")

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MAX_BATCH_SIZE = 1000
MAX_INVALID_RATE = 0.20
FRAUD_THRESHOLD = 0.5

# --------------------------------------------------
# 1. PRE-FLIGHT CHECKS
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

model_path = "outputs/xgb_fraud_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found")

df = pd.read_csv(data_path)

if "Class" not in df.columns:
    raise ValueError("Target column Class is missing")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load(model_path)

scaler = StandardScaler()
scaler.fit(X_train)

training_medians = X_train.median()
lower_limits = X_train.quantile(0.01)
upper_limits = X_train.quantile(0.99)

expected_features = list(X_train.columns)

if list(X.columns) != expected_features:
    raise ValueError("Dataset schema mismatch")

if model.n_features_in_ != len(expected_features):
    raise ValueError("Model feature compatibility failed")

if not np.isfinite(training_medians.to_numpy()).all():
    raise ValueError("Invalid training medians")

print("\n===== PRE-FLIGHT CHECKS =====")
print("Dataset loaded: PASS")
print("Model file found: PASS")
print("Feature count:", len(expected_features))
print("Model compatibility: PASS")
print("Scaler initialized: PASS")
print("Training statistics validated: PASS")

# --------------------------------------------------
# 2. ENHANCED RUNTIME VALIDATION
# --------------------------------------------------

def validate_and_predict(input_data):

    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if input_data.empty:
        raise ValueError("Empty input batch")

    if len(input_data) > MAX_BATCH_SIZE:
        raise ValueError(
            f"Batch too large. Maximum allowed: {MAX_BATCH_SIZE}"
        )

    if input_data.columns.duplicated().any():
        raise ValueError("Duplicate feature columns detected")

    if list(input_data.columns) != expected_features:
        raise ValueError(
            "Missing, extra, or incorrectly ordered features"
        )

    clean_data = input_data.copy()

    # Reject non-numeric values
    for column in expected_features:
        clean_data[column] = pd.to_numeric(
            clean_data[column],
            errors="coerce"
        )

    # Identify missing and infinite values
    clean_data = clean_data.replace(
        [np.inf, -np.inf], np.nan
    )

    invalid_rate = (
        clean_data.isna().sum().sum()
        / clean_data.size
    )

    if invalid_rate > MAX_INVALID_RATE:
        raise ValueError(
            f"Invalid data rate {invalid_rate:.2%} exceeds "
            f"allowed threshold {MAX_INVALID_RATE:.0%}"
        )

    # Sanitize acceptable missing values
    clean_data = clean_data.fillna(training_medians)

    # Clip extreme values
    clean_data = clean_data.clip(
        lower=lower_limits,
        upper=upper_limits,
        axis=1
    )

    if clean_data.isnull().values.any():
        raise ValueError("Missing values remain")

    if not np.isfinite(clean_data.to_numpy()).all():
        raise ValueError("Non-finite values remain")

    # Scale and predict
    scaled_data = scaler.transform(clean_data)

    probabilities = model.predict_proba(
        scaled_data
    )[:, 1]

    # --------------------------------------------------
    # 3. POST-PREDICTION VALIDATION
    # --------------------------------------------------

    if len(probabilities) != len(input_data):
        raise ValueError("Prediction count mismatch")

    if not np.isfinite(probabilities).all():
        raise ValueError("Invalid prediction probabilities")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probability outside valid range")

    predictions = (
        probabilities >= FRAUD_THRESHOLD
    ).astype(int)

    if not np.isin(predictions, [0, 1]).all():
        raise ValueError("Invalid prediction labels")

    return probabilities, predictions


# --------------------------------------------------
# 4. SAFETY STRESS TESTS
# --------------------------------------------------

test_batch = X_test.iloc[:100].copy()

test_cases = {}

test_cases["Normal Input"] = test_batch.copy()

missing_input = test_batch.copy()
missing_input.iloc[0, 0] = np.nan
test_cases["Missing Value"] = missing_input

infinite_input = test_batch.copy()
infinite_input.iloc[0, 1] = np.inf
test_cases["Infinite Value"] = infinite_input

extreme_input = test_batch.copy()
extreme_input.iloc[0, 0] = 1e12
test_cases["Extreme Value"] = extreme_input

# Excessive missing values
excessive_missing = test_batch.copy()
excessive_missing.iloc[:, :10] = np.nan
test_cases["Excessive Missing Values"] = excessive_missing

# Non-numeric values
# Non-numeric values
nonnumeric_input = test_batch.copy().astype(object)
nonnumeric_input.iloc[0, 0] = "invalid"
test_cases["Non-numeric Input"] = nonnumeric_input

# Oversized batch
oversized_input = pd.concat(
    [test_batch] * 11,
    ignore_index=True
)
test_cases["Oversized Batch"] = oversized_input

# Missing feature
missing_feature_input = test_batch.drop(
    columns=[expected_features[0]]
)
test_cases["Missing Feature"] = missing_feature_input

# Duplicate feature
# Actual duplicate feature test
duplicate_input = test_batch.copy()

duplicate_columns = expected_features.copy()
duplicate_columns[-1] = duplicate_columns[-2]

duplicate_input.columns = duplicate_columns

test_cases["Duplicate Feature"] = duplicate_input

results = []

for name, sample in test_cases.items():

    try:
        probabilities, predictions = validate_and_predict(sample)

        status = "PASS"

        details = "Input accepted and prediction validated"

        alerts = int(predictions.sum())

        avg_probability = float(probabilities.mean())

    except Exception as error:

        status = "BLOCKED"

        details = str(error)

        alerts = 0

        avg_probability = np.nan

    results.append({
        "Test_Case": name,
        "Status": status,
        "Transactions": len(sample),
        "Fraud_Alerts": alerts,
        "Average_Fraud_Probability": avg_probability,
        "Details": details
    })


# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/enhanced_system_safety_results.csv",
    index=False
)

print("\n===== ENHANCED SYSTEM SAFETY RESULTS =====")
print(results_df.to_string(index=False))

print("\nSaved: outputs/enhanced_system_safety_results.csv")

print("\nENHANCED SYSTEM SAFETY CHECKS COMPLETED")
