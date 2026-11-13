
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("PERFORMANCE MEASUREMENT STARTED")

# 1. Load dataset
possible_paths = [
    os.path.join("creditcard.csv (3)", "creditcard.csv"),
    os.path.join("creditcard (3)", "creditcard.csv")
]

data_path = next(
    (p for p in possible_paths if os.path.exists(p)),
    None
)

if data_path is None:
    raise FileNotFoundError("creditcard.csv not found")

df = pd.read_csv(data_path)

X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Reproduce original split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Load model and fit scaler
model_path = "outputs/xgb_fraud_model.pkl"
model = joblib.load(model_path)

scaler = StandardScaler()
scaler.fit(X_train)

# Training-derived sanitization limits
training_medians = X_train.median()
lower_limits = X_train.quantile(0.01)
upper_limits = X_train.quantile(0.99)

# 4. Prepare benchmark data
batch_size = 500
number_of_batches = 20

X_benchmark = X_test.iloc[
    :batch_size * number_of_batches
].copy()

batches = [
    X_benchmark.iloc[i:i + batch_size].copy()
    for i in range(0, len(X_benchmark), batch_size)
]

def sanitize_and_predict(batch):
    # Replace infinite values with missing values
    clean = batch.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Replace missing values using training medians
    clean = clean.fillna(training_medians)

    # Clip extreme values using training percentiles
    clean = clean.clip(
        lower=lower_limits,
        upper=upper_limits,
        axis=1
    )

    # Scale and predict
    scaled = scaler.transform(clean)

    return model.predict_proba(scaled)

# 5. Warm-up
model.predict_proba(scaler.transform(batches[0]))
sanitize_and_predict(batches[0])

# 6. Benchmark model-only inference
model_latencies = []

for batch in batches:
    scaled = scaler.transform(batch)

    start = time.perf_counter()
    model.predict_proba(scaled)
    elapsed = time.perf_counter() - start

    model_latencies.append(elapsed)

# 7. Benchmark complete validation pipeline
pipeline_latencies = []

for batch in batches:
    start = time.perf_counter()
    sanitize_and_predict(batch)
    elapsed = time.perf_counter() - start

    pipeline_latencies.append(elapsed)

# 8. Calculate statistics
def calculate_metrics(latencies):
    latencies = np.array(latencies)

    per_transaction_ms = (
        latencies * 1000 / batch_size
    )

    return {
        "P50 latency (ms/transaction)": np.percentile(
            per_transaction_ms, 50
        ),
        "P90 latency (ms/transaction)": np.percentile(
            per_transaction_ms, 90
        ),
        "Average latency (ms/transaction)": np.mean(
            per_transaction_ms
        ),
        "Throughput (transactions/second)": np.mean(
            batch_size / latencies
        )
    }

model_metrics = calculate_metrics(model_latencies)
pipeline_metrics = calculate_metrics(pipeline_latencies)

model_size_mb = (
    os.path.getsize(model_path) / (1024 * 1024)
)

# 9. Display results
print("\n===== MODEL-ONLY PERFORMANCE =====")

for name, value in model_metrics.items():
    print(f"{name}: {value:.5f}")

print("\n===== VALIDATION + MODEL PIPELINE =====")

for name, value in pipeline_metrics.items():
    print(f"{name}: {value:.5f}")

print(f"\nModel size: {model_size_mb:.2f} MB")

# 10. Save comparison
results = []

for metric in model_metrics:
    results.append({
        "Metric": metric,
        "Model_Only": model_metrics[metric],
        "Validation_Pipeline": pipeline_metrics[metric]
    })

results.append({
    "Metric": "Model size (MB)",
    "Model_Only": model_size_mb,
    "Validation_Pipeline": model_size_mb
})

results_df = pd.DataFrame(results)

os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/performance_pipeline_comparison.csv",
    index=False
)

print("\nSaved: outputs/performance_pipeline_comparison.csv")
print("\nPERFORMANCE MEASUREMENT COMPLETED!")
