import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


def split_claim_severity_data(df: pd.DataFrame, target_col='TotalClaims', test_size=0.2, random_state=42):
    """
    Filters to rows with claims > 0, separates features and target, and splits into train/test.
    """
    df_claims = df[df[target_col] > 0].copy()
    X = df_claims.drop(columns=[target_col, 'has_claim', 'TotalPremium', 'margin'])  # Drop leakage columns
    y = df_claims[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {'RMSE': rmse, 'R2': r2, 'predictions': preds}
