# ==============================
# PRACTICAL NO. 4
# AIM: Multi-class Classification using OvR & OvO Strategies
# Dataset: Wine (sklearn)
# ==============================

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load Dataset
data = load_wine()
X = data.data
y = data.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Feature Scaling
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Base Models
base_model_lr  = LogisticRegression(max_iter=500)
base_model_svm = SVC(kernel='rbf', probability=True)

# OvR Strategy
ovr_lr  = OneVsRestClassifier(base_model_lr)
ovr_svm = OneVsRestClassifier(base_model_svm)
ovr_lr.fit(X_train, y_train)
ovr_svm.fit(X_train, y_train)

# OvO Strategy
ovo_lr  = OneVsOneClassifier(base_model_lr)
ovo_svm = OneVsOneClassifier(base_model_svm)
ovo_lr.fit(X_train, y_train)
ovo_svm.fit(X_train, y_train)

# Evaluation Function
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted')
    print(f"\n{name}")
    print("-" * 40)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

# Results
evaluate_model("OvR - Logistic Regression", y_test, ovr_lr.predict(X_test))
evaluate_model("OvR - SVM",                 y_test, ovr_svm.predict(X_test))
evaluate_model("OvO - Logistic Regression", y_test, ovo_lr.predict(X_test))
evaluate_model("OvO - SVM",                 y_test, ovo_svm.predict(X_test))
