import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from src.config import LOG_REG_PARAMS, GRAD_BOOST_PARAMS, DATA_DIR

def train_models(X_train, y_train):
    """
    Trains Logistic Regression and Gradient Boosting models on the provided training set.
    """
    print("[Models] Training Logistic Regression...")
    lr_model = LogisticRegression(**LOG_REG_PARAMS)
    lr_model.fit(X_train, y_train)
    print("  Logistic Regression train accuracy: {:.4f}".format(lr_model.score(X_train, y_train)))

    print("[Models] Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(**GRAD_BOOST_PARAMS)
    gb_model.fit(X_train, y_train)
    print("  Gradient Boosting train accuracy:  {:.4f}".format(gb_model.score(X_train, y_train)))

    # Save models locally
    models_dir = DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    with open(models_dir / "logistic_regression.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    with open(models_dir / "gradient_boosting.pkl", "wb") as f:
        pickle.dump(gb_model, f)
    print(f"[Models] Models serialized to {models_dir}")

    return lr_model, gb_model

def get_feature_importances(lr_model, gb_model, feature_names):
    """
    Extracts the feature importances / coefficients for both models,
    paired with feature names, sorted by importance.
    """
    # 1. Logistic Regression Coefficients
    lr_coefs = lr_model.coef_[0]
    lr_importance = []
    for name, coef in zip(feature_names, lr_coefs):
        lr_importance.append({
            "feature": name,
            "coefficient": float(coef),
            "abs_coef": float(abs(coef))
        })
    # Sort by absolute coefficient size
    lr_importance = sorted(lr_importance, key=lambda x: x["abs_coef"], reverse=True)

    # 2. Gradient Boosting Gini Importance
    gb_importances = gb_model.feature_importances_
    gb_importance = []
    for name, imp in zip(feature_names, gb_importances):
        gb_importance.append({
            "feature": name,
            "importance": float(imp)
        })
    # Sort by importance
    gb_importance = sorted(gb_importance, key=lambda x: x["importance"], reverse=True)

    return lr_importance, gb_importance
