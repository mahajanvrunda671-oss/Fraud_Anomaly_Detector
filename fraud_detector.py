import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import xgboost as xgb
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

print("Current folder:",os.getcwd())
print("Files:",os.listdir())
print("CSV files:",[f for f in os.listdir() if f.lower().endswith("CSV")] )
file_path=r"creditcard.csv (3)\creditcard.csv"
if os.path.exists(file_path):
    print("Dataset found successfully!")
    df=pd.read_csv(file_path)
    
    print(df.head())
    print("Rows and columns:", df.shape)
    print('\n--- Dataset Info & Schema ---')
    print(df.info())
    
    print("\n---Missing Values Check---")
    print(df.isnull().sum())


    print("\n---Duplicates Rows Check---")
    print(f"Total duplicate rows: {df.duplicated().sum()}")

    print("\n---Summary Statistic---")
    print(df.describe())

    print("\n---Class Distribution---")
    print(df['Class'].value_counts())
    print(df['Class'].value_counts(normalize=True) * 100)

    output_dir="outputs"
    os.makedirs(output_dir, exist_ok=True)

    summary_path=os.path.join(output_dir, 'summary_statistics.csv')
    df.describe().to_csv(summary_path)
    print(f"\n[Saved Artifact] Summary ststistics saved to: {summary_path}")

    plt.figure(figsize=(6,4))
    sns.countplot(x='Class',data=df)
    plt.yscale('log')
    
    plt.title('Distribution of Fraud vs Normal Transactions')
    plt.xlabel('Class (0: Normal, 1:Fraud)')
    plt.ylabel('Count')
    plot_path=os.path.join(output_dir, 'class_distribution_plot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"[Saved Artifact] Plot saved to: {plot_path}")

    print("\n--- Generating Correlation Heatmap ---")
    plt.figure(figsize=(10,8))

    subset_cols=list(df.columns[:10]) + ['Amount','Class']
    corr_matrix=df[subset_cols].corr()

    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap (Subset)')

    heatmap_path=os.path.join(output_dir,'correlation_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight')

else:
    print("Dataset not found!")

# 3. Features and Target Selection
X=df.drop(columns=['Class'])
y=df['Class']

# 4. Stratified Train_Test Split
X_train, X_test, y_train, y_test=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Feature Standardization
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# 6. Class Imbalance Handling 
neg_count=(y_train==0).sum()
pos_count=(y_train==1).sum()
scale_pos_weight_val=neg_count / pos_count if pos_count>0 else 1.0

#----Baseline Model Comparison-----
print("Training Baseline Model (Logistic Regression)...")
from sklearn.linear_model import LogisticRegression
baseline_clf=LogisticRegression(random_state=42, max_iter=1000)
baseline_clf.fit(X_train_scaled, y_train)
baseline_pred_proba=baseline_clf.predict_proba(X_test_scaled)[:,1]
baseline_auc=roc_auc_score(y_test, baseline_pred_proba)
print(f"Baseline ROC-AUC Score: {baseline_auc:.4f}")



# 7. XGBoost Model Intialization and Training
xgb_clf=xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric=['auc', 'logloss', 'aucpr'],
    max_depth=6,
    learning_rate=0.05,
    n_estimators=200,
    scale_pos_weight=scale_pos_weight_val,
    random_state=42,
    n_jobs=-1
)
print("Training XGBoost Classifier with Learning Curve tracking...")
eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)]
xgb_clf.fit(X_train_scaled,y_train, eval_set=eval_set, verbose=False)

# Plot Learning Curve
results=xgb_clf.evals_result()
plt.figure(figsize=(8,5))
plt.plot(results['validation_0']['auc'], label='Train AUC')
plt.plot(results['validation_1']['auc'], label='Test AUC')
plt.title('XGBoost Learning Curve (AUC vs. Boosting Iterations)')
plt.xlabel('Boosting Iterations/ Trees')
plt.ylabel('AUC Score')
plt.legend()
lc_path=os.path.join(output_dir, 'learning_curve.png')
plt.savefig(lc_path, bbox_inches='tight')
plt.close()
print(f"[Saved Artifact] Learning curve plot saved to: {lc_path}")


# Second graph: Validation Log-Loss and PR-AUC

results = xgb_clf.evals_result()
iterations = range(1, len(results['validation_1']['logloss']) + 1)

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(iterations,results['validation_1']['logloss'],color='red',label='Val Log-Loss')
ax1.set_xlabel('Boosting Iterations (Trees)')
ax1.set_ylabel('Validation Log-Loss', color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.plot(iterations,results['validation_1']['aucpr'],color='blue',label='Val PR-AUC')
ax2.set_ylabel('Validation PR-AUC', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.title('Training Convergence & Loss/Metric Optimization')
fig.tight_layout()

convergence_path = os.path.join(output_dir, 'training_convergence_loss_pr_auc.png')
plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[Saved Artifact] Convergence graph saved to: {convergence_path}")


# 8. Evaluation Metrics
y_pred=xgb_clf.predict(X_test_scaled)
y_pred_proba=xgb_clf.predict_proba(X_test_scaled)[:,1]

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"PR-AUC Score: {average_precision_score(y_test, y_pred_proba):.4f}")
print("Classification_Report:")
print(classification_report(y_test, y_pred))

# 9. Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
cm_path=os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(cm_path, bbox_inches='tight')
plt.close()

# 10. Feature Importance Plot
plt.figure(figsize=(10,6))
xgb.plot_importance(xgb_clf, max_num_features=10, height=0.5, color='teal')
plt.title('Top 10 Feature Importances')
fi_path=os.path.join(output_dir, 'feature_importance.png')
plt.savefig(fi_path, bbox_inches='tight')
plt.close()

# 11. Explainable AI using SHAP
print("\nGenerating SHAP Explanations...")
explainer=shap.TreeExplainer(xgb_clf)
shap_values=explainer.shap_values(X_test_scaled)

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
shap_path=os.path.join(output_dir, 'shap_summary_plot.png')
plt.savefig(shap_path, bbox_inches='tight')
plt.close()

# 12. Save the Trained Model
model_path=os.path.join(output_dir, 'xgb_fraud_model.pkl')
joblib.dump(xgb_clf, model_path)

print(f"\n[Saved Artifact] Confusion matrix plot saved to: {cm_path}")
print(f"\n[Saved Artifact] Feature Importance plot saved to: {fi_path}")
print(f"\n[Saved Artifact] SHAP summary plot saved to: {shap_path}")
print(f"\n[Saved Artifact] Trained model plot saved to: {model_path}")

# 13. Efficiency Metrics Measurement
start_time=time.time()
train_time=time.time() - start_time

# Model size on disk
model_path=os.path.join(output_dir, 'xgb_fraud_model.pkl')
model_size_kb=os.path.getsize(model_path)/1024
print(f"\n[Efficiency Metric] Model Size on Disk: {model_size_kb:.2f} KB")

# Inference Latency & Throughput
start_inf=time.time()
y_pred_test=xgb_clf.predict(X_test_scaled)
inf_time=time.time() - start_inf

total_samples=len(X_test_scaled)
latency_per_sample=(inf_time/total_samples) * 1000
throughput=total_samples/inf_time

print(f"[Efficiency Metric] Inference Latency per sample: {latency_per_sample:.4f} ms")
print(f"[Efficiency Metric] Throughput: {throughput:.2f} samples/sec")

# 13. Print top factors for flagged fraud transactions
flagged_indices=np.where(y_pred==1)[0]

if len(flagged_indices)>0:
    sample_idx=flagged_indices[0]
    print(f"\n---Top Contributing Factors for Flagged Transactions #{sample_idx}---- ")

    features_names=X.columns
    sample_shap=shap_values[sample_idx]
    sample_features=X_test.iloc[sample_idx]

    top_factor_idx=np.argsort(np.abs(sample_shap))[::-1][:5]

    for rank, idx in enumerate(top_factor_idx, 1):
        feat=features_names[idx]
        val=sample_features[feat]
        shap_val=sample_shap[idx]
        impact="Increased Fraud Risk" if shap_val > 0 else "Decreased Fraud Risk"
        print(f"{rank}.Feature: {feat:<12} | Value: {val:<8.4f} | SHAP Impact: {shap_val:+.4f} ({impact})")



































































