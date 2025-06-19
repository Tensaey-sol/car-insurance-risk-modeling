import shap
import matplotlib.pyplot as plt

def compute_shap_values(model, X_test, model_name: str):
    """
    Compute SHAP values for a given tree-based model.
    
    Parameters:
    - model: Trained model (Random Forest or XGBoost)
    - X_test: Test features (pandas DataFrame or numpy array)
    - model_name: Name of the model for error reporting (str)
    
    Returns:
    - shap_values: SHAP values for the test set
    - explainer: SHAP TreeExplainer object
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        return shap_values, explainer
    except Exception as e:
        print(f"Error computing SHAP values for {model_name}: {e}")
        return None, None

def plot_shap_summary(shap_values, X_test, model_name: str, max_display: int = 10):
    """
    Generate and display a SHAP summary plot (bar) for feature importance.
    
    Parameters:
    - shap_values: SHAP values from compute_shap_values
    - X_test: Test features (pandas DataFrame or numpy array)
    - model_name: Name of the model for plot title (str)
    - max_display: Number of top features to display (int, default=10)
    """
    if shap_values is None:
        print(f"No SHAP values available for {model_name}. Skipping plot.")
        return
    
    plt.figure(figsize=(15, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=max_display, show=False)
    plt.title(f"SHAP Feature Importance for {model_name}")
    plt.tight_layout()
    plt.show()

def run_shap_analysis(rf_model, xgb_model, X_test):
    """
    Run SHAP analysis for Random Forest and XGBoost models.
    
    Parameters:
    - rf_model: Trained Random Forest model
    - xgb_model: Trained XGBoost model
    - X_test: Test features (pandas DataFrame or numpy array)
    """
    # Compute SHAP values for Random Forest
    rf_shap_values, _ = compute_shap_values(rf_model, X_test, "Random Forest")
    plot_shap_summary(rf_shap_values, X_test, "Random Forest")

    # Compute SHAP values for XGBoost
    xgb_shap_values, _ = compute_shap_values(xgb_model, X_test, "XGBoost")
    plot_shap_summary(xgb_shap_values, X_test, "XGBoost")