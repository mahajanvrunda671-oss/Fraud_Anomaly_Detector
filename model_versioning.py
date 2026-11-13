
import os
import json
import hashlib
import joblib
import pandas as pd
import sklearn
import xgboost
import datetime
import platform

from pathlib import Path

print("MODEL VERSIONING STARTED")

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_PATHS = [
    Path("creditcard.csv (3)/creditcard.csv"),
    Path("creditcard (3)/creditcard.csv")
]

MODEL_FILES = {
    "v1.0.0": {
        "name": "Original XGBoost Model",
        "path": OUTPUT_DIR / "xgb_fraud_model.pkl",
        "description": "Original model trained on clean training data"
    },
    "v1.1.0": {
        "name": "Adapted XGBoost Model",
        "path": OUTPUT_DIR / "xgb_fraud_model_adapted.pkl",
        "description": "Adapted model trained using simulated drifted data"
    }
}

# --------------------------------------------------
# 2. FILE HASH FUNCTION
# --------------------------------------------------

def calculate_sha256(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# --------------------------------------------------
# 3. DATASET VERSION INFORMATION
# --------------------------------------------------

dataset_path = next(
    (path for path in DATASET_PATHS if path.exists()),
    None
)

if dataset_path is None:
    raise FileNotFoundError("Credit card dataset not found")

dataset = pd.read_csv(dataset_path)

dataset_metadata = {
    "dataset_name": "Credit Card Fraud Detection Dataset",
    "dataset_version": "dataset_v1.0",
    "file_name": dataset_path.name,
    "rows": int(dataset.shape[0]),
    "columns": int(dataset.shape[1]),
    "fraud_transactions": int(dataset["Class"].sum()),
    "legitimate_transactions": int(
        (dataset["Class"] == 0).sum()
    ),
    "fraud_prevalence_percent": float(
        dataset["Class"].mean() * 100
    ),
    "sha256": calculate_sha256(dataset_path)
}

print("\n===== DATASET VERSION =====")
print(json.dumps(dataset_metadata, indent=4))


# --------------------------------------------------
# 4. MODEL VERSION REGISTRY
# --------------------------------------------------

registry = []

for version, details in MODEL_FILES.items():

    model_path = details["path"]

    if not model_path.exists():

        print(f"\nWARNING: Model not found: {model_path}")
        continue

    model = joblib.load(model_path)

    model_metadata = {
        "version": version,
        "model_name": details["name"],
        "description": details["description"],
        "file_name": model_path.name,
        "file_size_mb": round(
            model_path.stat().st_size / (1024 * 1024), 6
        ),
        "sha256": calculate_sha256(model_path),
        "algorithm": type(model).__name__,
        "number_of_features": int(
            getattr(model, "n_features_in_", 0)
        ),
        "model_parameters": {
            key: model.get_params().get(key)
            for key in [
                "max_depth",
                "learning_rate",
                "n_estimators",
                "random_state",
                "scale_pos_weight"
            ]
            if hasattr(model, "get_params")
        }
    }

    registry.append(model_metadata)

    print(f"\n===== MODEL {version} =====")
    print(json.dumps(model_metadata, indent=4))


# --------------------------------------------------
# 5. EXPERIMENT AND ENVIRONMENT INFORMATION
# --------------------------------------------------

experiment_metadata = {
    "project": "Explainable AI-Based Credit Card Fraud Detection",
    "milestone": "Milestone 3",
    "created_at_utc": datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(),
    "python_version": platform.python_version(),
    "operating_system": platform.platform(),
    "pandas_version": pd.__version__,
    "scikit_learn_version": sklearn.__version__,
    "xgboost_version": xgboost.__version__,
    "dataset_version": dataset_metadata["dataset_version"],
    "model_versions": [
        item["version"] for item in registry
    ]
}

# --------------------------------------------------
# 6. SAVE VERSION MANIFEST
# --------------------------------------------------

manifest = {
    "dataset": dataset_metadata,
    "models": registry,
    "experiment": experiment_metadata
}

manifest_path = OUTPUT_DIR / "model_version_manifest.json"

with open(manifest_path, "w") as file:

    json.dump(
        manifest,
        file,
        indent=4,
        default=str
    )

# --------------------------------------------------
# 7. SAVE MODEL REGISTRY CSV
# --------------------------------------------------

registry_df = pd.DataFrame(registry)

registry_path = OUTPUT_DIR / "model_version_registry.csv"

registry_df.to_csv(
    registry_path,
    index=False
)

# --------------------------------------------------
# 8. FINAL SUMMARY
# --------------------------------------------------

print("\n===== VERSIONING SUMMARY =====")

print("Dataset version:", dataset_metadata["dataset_version"])

print("Registered models:", len(registry))

for item in registry:
    print(
        item["version"],
        "-",
        item["model_name"],
        "-",
        item["file_size_mb"],
        "MB"
    )

print("\nSaved files:")
print(manifest_path)
print(registry_path)

print("\nMODEL VERSIONING COMPLETED")
