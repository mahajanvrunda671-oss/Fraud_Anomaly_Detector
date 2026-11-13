
import os
import time
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

print("\nHYPERPARAMETER TUNING STARTED")

# ------------------------------------------------
# 1. Load dataset
# ------------------------------------------------

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
    raise FileNotFoundError("creditcard.csv not found")

df = pd.read_csv(data_path)

X = df.drop(columns=["Class"])
y = df["Class"]

# Reproduce original split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create validation data only from original training partition
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=21,
    stratify=y_train_full
)

# ------------------------------------------------
# 2. Scale using training partition only
# ------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

# ------------------------------------------------
# 3. Define 8 configurations
# ------------------------------------------------

parameter_grid = {
    "max_depth": [3, 6],
    "learning_rate": [0.03, 0.05],
    "n_estimators": [150, 250],
}

configurations = list(ParameterGrid(parameter_grid))

# ------------------------------------------------
# 4. Train and validate each configuration
# ------------------------------------------------

results = []
trained_models = []

for index, params in enumerate(configurations, start=1):

    print(f"\nConfiguration {index}/{len(configurations)}")
    print(params)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        **params
    )

    start = time.perf_counter()

    model.fit(X_train_scaled, y_train)

    training_time = time.perf_counter() - start

    val_probabilities = model.predict_proba(
        X_val_scaled
    )[:, 1]

    val_predictions = (val_probabilities >= 0.5).astype(int)

    precision = precision_score(
        y_val, val_predictions, zero_division=0
    )

    recall = recall_score(
        y_val, val_predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, val_predictions, zero_division=0
    )

    pr_auc = average_precision_score(
        y_val, val_probabilities
    )

    roc_auc = roc_auc_score(
        y_val, val_probabilities
    )

    results.append({
        "Configuration": index,
        **params,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "Training_Time_Seconds": training_time
    })

    trained_models.append(model)

    print(
        f"Validation PR-AUC: {pr_auc:.6f} | "
        f"F1: {f1:.6f}"
    )

# ------------------------------------------------
# 5. Select best configuration using validation PR-AUC
# ------------------------------------------------

results_df = pd.DataFrame(results)

best_index = results_df["PR_AUC"].idxmax()

best_model = trained_models[best_index]

print("\n===== HYPERPARAMETER RESULTS =====")
print(results_df.to_string(index=False))

print("\n===== BEST CONFIGURATION =====")
print(results_df.iloc[best_index].to_string())

# ------------------------------------------------
# 6. Evaluate selected model on original test set
# ------------------------------------------------

# Refit scaler on the complete original training partition
final_scaler = StandardScaler()

X_train_full_scaled = final_scaler.fit_transform(
    X_train_full
)

X_test_scaled = final_scaler.transform(X_test)

final_negative = (y_train_full == 0).sum()
final_positive = (y_train_full == 1).sum()

final_weight = final_negative / final_positive

best_params = {
    key: results_df.iloc[best_index][key]
    for key in ["max_depth", "learning_rate", "n_estimators"]
}

final_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=final_weight,
    random_state=42,
    n_jobs=-1,
    max_depth=int(best_params["max_depth"]),
    learning_rate=float(best_params["learning_rate"]),
    n_estimators=int(best_params["n_estimators"])
)

final_model.fit(X_train_full_scaled, y_train_full)

test_probabilities = final_model.predict_proba(
    X_test_scaled
)[:, 1]

test_predictions = (test_probabilities >= 0.5).astype(int)

test_results = {
    "Precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "F1": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "PR_AUC": average_precision_score(
        y_test, test_probabilities
    ),
    "ROC_AUC": roc_auc_score(
        y_test, test_probabilities
    )
}

print("\n===== FINAL TEST RESULTS =====")

for metric, value in test_results.items():
    print(f"{metric}: {value:.6f}")

# ------------------------------------------------
# 7. Save outputs separately
# ------------------------------------------------

os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/hyperparameter_tuning_results.csv",
    index=False
)

pd.DataFrame([test_results]).to_csv(
    "outputs/tuned_model_test_results.csv",
    index=False
)

joblib.dump(
    final_model,
    "outputs/xgb_fraud_model_tuned.pkl"
)

joblib.dump(
    final_scaler,
    "outputs/xgb_fraud_scaler_tuned.pkl"
)

print("\nSaved tuning results and tuned model.")
print("HYPERPARAMETER TUNING COMPLETED")
