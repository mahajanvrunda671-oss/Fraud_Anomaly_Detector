
import os
import time
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("\nREVISED PERFORMANCE BENCHMARK STARTED")

# --------------------------------------------------
# 1. Find dataset
# --------------------------------------------------

possible_paths = [
    os.path.join("creditcard.csv (3)", "creditcard.csv"),
    os.path.join("creditcard (3)", "creditcard.csv"),
    "creditcard.csv"
]

data_path = next(
    (p for p in possible_paths if os.path.isfile(p)),
    None
)

if data_path is None:
    raise FileNotFoundError(
        "Dataset not found. Check your creditcard.csv location."
    )

df = pd.read_csv(data_path)

X = df.drop(columns=["Class"])
y = df["Class"]

# Reproduce original split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 2. Load model and fit training scaler
# --------------------------------------------------

model_path = "outputs/xgb_fraud_model.pkl"

if not os.path.isfile(model_path):
    raise FileNotFoundError("Trained model not found.")

model = joblib.load(model_path)

scaler = StandardScaler()
scaler.fit(X_train)

training_medians = X_train.median()
lower_limits = X_train.quantile(0.01)
upper_limits = X_train.quantile(0.99)

# --------------------------------------------------
# 3. Define complete validation pipeline
# --------------------------------------------------

def prepare_input(batch):

    clean = batch.copy()

    clean = clean.replace(
        [np.inf, -np.inf],
        np.nan
    )

    clean = clean.fillna(training_medians)

    clean = clean.clip(
        lower=lower_limits,
        upper=upper_limits,
        axis=1
    )

    scaled = scaler.transform(clean)

    return scaled


def full_pipeline(batch):

    scaled = prepare_input(batch)

    return model.predict_proba(scaled)


# --------------------------------------------------
# 4. Benchmark function
# --------------------------------------------------

def benchmark(function, batch, repetitions, warmups=10):

    # Warm-up calls
    for _ in range(warmups):
        function(batch)

    latencies_ms = []

    for _ in range(repetitions):

        start = time.perf_counter()

        function(batch)

        elapsed = time.perf_counter() - start

        latencies_ms.append(elapsed * 1000)

    latencies_ms = np.array(latencies_ms)

    total_seconds = np.sum(latencies_ms) / 1000

    batch_size = len(batch)

    total_transactions = batch_size * repetitions

    throughput = total_transactions / total_seconds

    return {
        "P50_call_ms": np.percentile(latencies_ms, 50),
        "P90_call_ms": np.percentile(latencies_ms, 90),
        "P95_call_ms": np.percentile(latencies_ms, 95),
        "Mean_call_ms": np.mean(latencies_ms),
        "P50_amortized_ms_per_transaction":
            np.percentile(latencies_ms, 50) / batch_size,
        "Throughput_transactions_per_second": throughput,
        "Batch_size": batch_size,
        "Repetitions": repetitions
    }


# --------------------------------------------------
# 5. Benchmark different batch sizes
# --------------------------------------------------

batch_sizes = [1, 32, 128, 500]

results = []

for batch_size in batch_sizes:

    batch = X_test.iloc[:batch_size].copy()

    scaled_batch = scaler.transform(batch)

    # More repetitions for small batches
    if batch_size == 1:
        repetitions = 1000
    elif batch_size == 32:
        repetitions = 300
    elif batch_size == 128:
        repetitions = 150
    else:
        repetitions = 100

    print(f"\nBenchmarking batch size: {batch_size}")

    # Model-only: scaling excluded
    model_result = benchmark(
        lambda data: model.predict_proba(data),
        scaled_batch,
        repetitions
    )

    model_result["Pipeline"] = "Model-only"
    results.append(model_result)

    # End-to-end: cleaning + scaling + prediction
    pipeline_result = benchmark(
        full_pipeline,
        batch,
        repetitions
    )

    pipeline_result["Pipeline"] = "Full validation pipeline"
    results.append(pipeline_result)


# --------------------------------------------------
# 6. Save benchmark results
# --------------------------------------------------

results_df = pd.DataFrame(results)

os.makedirs("outputs", exist_ok=True)

output_path = (
    "outputs/revised_performance_benchmark.csv"
)

results_df.to_csv(output_path, index=False)

# --------------------------------------------------
# 7. System information
# --------------------------------------------------

print("\n===== SYSTEM INFORMATION =====")
print("Python:", platform.python_version())
print("Platform:", platform.platform())
print("Scikit-learn:", sklearn.__version__)
print("XGBoost:", xgboost.__version__)
print("CPU:", platform.processor())

print("\n===== BENCHMARK RESULTS =====")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nSaved:", output_path)

print("\nREVISED PERFORMANCE BENCHMARK COMPLETED")
