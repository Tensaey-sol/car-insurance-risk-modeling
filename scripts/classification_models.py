import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

def train_classification_models(df: pd.DataFrame, target_col: str = 'has_claim', test_size: float = 0.2, random_state: int = 42):
    # Prepare features
    X = df.drop(columns=[target_col, 'TotalClaims', 'TotalPremium'], errors='ignore')
    y = df[target_col]

    # Encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    results['Logistic Regression'] = evaluate_classifier(y_test, y_pred_lr, y_proba_lr)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    results['Random Forest'] = evaluate_classifier(y_test, y_pred_rf, y_proba_rf)

    # XGBoost
    xgb = XGBClassifier(eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    results['XGBoost'] = evaluate_classifier(y_test, y_pred_xgb, y_proba_xgb)

    return results, {'lr': lr, 'rf': rf, 'xgb': xgb}, X_test, y_test

def evaluate_classifier(y_true, y_pred, y_proba):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'ConfusionMatrix': confusion_matrix(y_true, y_pred),
        'ClassificationReport': classification_report(y_true, y_pred, output_dict=True)
    }