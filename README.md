# INSE 6450 - AI in Systems Engineering: Milestone 2
**Team 10: Explainable AI-Based Anomaly Detection**

## 📌 Project Overview
This project implements an optimized and explainable XGBoost classifier to detect fraudulent credit card transactions from highly imbalanced data (~0.17% fraud cases). It includes comprehensive model evaluation, efficiency metrics benchmarking, real-time deployment architecture design, and SHAP-based local/global interpretability.

---

## 📂 Repository Structure
- `fraud_detector.py`: Main Python script for training, evaluation, efficiency checks, and SHAP explanations.
- `outputs/`: Folder containing generated plots (Learning curves, confusion matrix, SHAP summary plots) and trained model artifacts (`.pkl`).
- `requirements.txt`: Python package dependencies.

---

## 📊 Dataset Access
The project uses the standard Credit Card Fraud Detection dataset available on Kaggle:
- [View Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## ⚙️ How to Run the Code
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd Fraud_Anomaly_Detector

2. Install Required dependencies
pip install -r requirements.txt

3. Run the fraud detector script
python fraud_detector.py# Fraud_Anomaly_Detector
