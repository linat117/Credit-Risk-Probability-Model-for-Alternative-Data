from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Credit Risk Model Training")

# Load prepared dataset
df = pd.read_csv('data/processed/model_training_data.csv')

# Separate features and target
X = df.drop(columns=[ 'is_high_risk'])
y = df['is_high_risk']

X = X.select_dtypes(include=["int64", "float64"])

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Initialize models
log_reg = LogisticRegression(
    max_iter=1000,
    random_state=42)
rf = RandomForestClassifier(random_state=42)

# Train
log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV

# Example: Random Forest tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Start MLflow run
with mlflow.start_run(run_name="RandomForest_Experiment"):

    # Train model
    best_rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = best_rf.predict(X_test)
    y_prob = best_rf.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Log model
    mlflow.sklearn.log_model(best_rf, "model")
