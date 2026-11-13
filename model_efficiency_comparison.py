
import os
import time
import joblib
import numpy as np
import pandas as pd
import psutil

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

print("MODEL EFFICIENCY COMPARISON STARTED")

# --------------------------------------------------
# 1. LOAD DATASET
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
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 2. LOAD ORIGINAL AND ADAPTED MODELS
# --------------------------------------------------

original_model = joblib.load(
    "outputs/xgb_fraud_model.pkl"
)

adapted_model = joblib.load(
    "outputs/xgb_fraud_model_adapted.pkl"
)

# Recreate original scaler
original_scaler = StandardScaler()
original_scaler.fit(X_train)

# Load adapted scaler
adapted_scaler = joblib.load(
    "outputs/adapted_scaler.pkl"
)

# --------------------------------------------------
# 3. PREPARE CLEAN AND DRIFTED DATA
# --------------------------------------------------

X_clean = X_test.copy()

X_drifted = X_test.copy()
X_drifted["Amount"] = (
    X_drifted["Amount"] * 1.5 + 10
)
X_drifted["V14"] = X_drifted["V14"] + 0.5

# --------------------------------------------------
# 4. BENCHMARK FUNCTION
# --------------------------------------------------

def benchmark_model(
    model,
    scaler,
    model_name,
    data_name,
    input_data,
    labels
):

    scaled_data = scaler.transform(input_data)

    # Warm-up
    model.predict_proba(scaled_data[:100])

    # Latency benchmark
    latencies = []
    process = psutil.Process(os.getpid())

    memory_before = process.memory_info().rss / (1024 ** 2)

    batch_size = 500
    number_of_batches = 20

    for _ in range(number_of_batches):

        batch = scaled_data[:batch_size]

        start = time.perf_counter()

        model.predict_proba(batch)

        end = time.perf_counter()

        latency_ms = (
            (end - start) * 1000 / batch_size
        )

        latencies.append(latency_ms)

    memory_after = process.memory_info().rss / (1024 ** 2)

    # Performance evaluation on complete dataset
    probabilities = model.predict_proba(
        scaled_data
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    precision = precision_score(
        labels, predictions, zero_division=0
    )

    recall = recall_score(
        labels, predictions, zero_division=0
    )

    f1 = f1_score(
        labels, predictions, zero_division=0
    )

    pr_auc = average_precision_score(
        labels, probabilities
    )

    roc_auc = roc_auc_score(
        labels, probabilities
    )

    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)

    avg_latency = np.mean(latencies)

    throughput = 1000 / avg_latency

    model_file = (
        "outputs/xgb_fraud_model.pkl"
        if model_name == "Original Model"
        else "outputs/xgb_fraud_model_adapted.pkl"
    )

    model_size = (
        os.path.getsize(model_file) / (1024 ** 2)
    )

    return {
        "Model": model_name,
        "Dataset": data_name,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "P50_Latency_ms": p50,
        "P90_Latency_ms": p90,
        "Throughput_tx_per_sec": throughput,
        "Model_Size_MB": model_size,
        "Process_Memory_Before_MB": memory_before,
        "Process_Memory_After_MB": memory_after
    }


# --------------------------------------------------
# 5. RUN FOUR COMPARISONS
# --------------------------------------------------

results = []

experiments = [
    (
        original_model,
        original_scaler,
        "Original Model",
        "Clean Data",
        X_clean
    ),
    (
        original_model,
        original_scaler,
        "Original Model",
        "Drifted Data",
        X_drifted
    ),
    (
        adapted_model,
        adapted_scaler,
        "Adapted Model",
        "Clean Data",
        X_clean
    ),
    (
        adapted_model,
        adapted_scaler,
        "Adapted Model",
        "Drifted Data",
        X_drifted
    )
]

for model, scaler, model_name, data_name, data in experiments:

    print(
        "\nTesting:",
        model_name,
        "|",
        data_name
    )

    result = benchmark_model(
        model,
        scaler,
        model_name,
        data_name,
        data,
        y_test
    )

    results.append(result)


# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/model_efficiency_comparison.csv",
    index=False
)

print("\n===== MODEL EFFICIENCY COMPARISON =====")
print(results_df.to_string(index=False))

print(
    "\nSaved: outputs/model_efficiency_comparison.csv"
)

print("\nMODEL EFFICIENCY COMPARISON COMPLETED")
