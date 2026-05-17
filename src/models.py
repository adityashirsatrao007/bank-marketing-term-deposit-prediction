import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.config import (
    LOG_REG_PARAMS, KNN_PARAMS, NAIVE_BAYES_PARAMS, 
    RANDOM_FOREST_PARAMS, GRAD_BOOST_PARAMS, DATA_DIR
)

def train_models(X_train, y_train):
    """
    Trains all five classification models (Logistic Regression, KNN, Naive Bayes, 
    Random Forest, and Gradient Boosting Classifier) on the balanced training set.
    """
    # 1. Logistic Regression
    print("[Models] Training Logistic Regression...")
    lr_model = LogisticRegression(**LOG_REG_PARAMS)
    lr_model.fit(X_train, y_train)
    print("  Logistic Regression train accuracy: {:.4f}".format(lr_model.score(X_train, y_train)))

    # 2. K-Nearest Neighbors
    print("[Models] Training K-Nearest Neighbors...")
    knn_model = KNeighborsClassifier(**KNN_PARAMS)
    knn_model.fit(X_train, y_train)
    print("  K-Nearest Neighbors train accuracy:  {:.4f}".format(knn_model.score(X_train, y_train)))

    # 3. Naive Bayes
    print("[Models] Training Naive Bayes (Gaussian NB)...")
    nb_model = GaussianNB(**NAIVE_BAYES_PARAMS)
    nb_model.fit(X_train, y_train)
    print("  Naive Bayes train accuracy:          {:.4f}".format(nb_model.score(X_train, y_train)))

    # 4. Random Forest
    print("[Models] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    rf_model.fit(X_train, y_train)
    print("  Random Forest train accuracy:         {:.4f}".format(rf_model.score(X_train, y_train)))

    # 5. Gradient Boosting Classifier (Champion)
    print("[Models] Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(**GRAD_BOOST_PARAMS)
    gb_model.fit(X_train, y_train)
    print("  Gradient Boosting train accuracy:    {:.4f}".format(gb_model.score(X_train, y_train)))

    # Save models locally
    models_dir = DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    with open(models_dir / "logistic_regression.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    with open(models_dir / "knn.pkl", "wb") as f:
        pickle.dump(knn_model, f)
    with open(models_dir / "naive_bayes.pkl", "wb") as f:
        pickle.dump(nb_model, f)
    with open(models_dir / "random_forest.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open(models_dir / "gradient_boosting.pkl", "wb") as f:
        pickle.dump(gb_model, f)
    print(f"[Models] All 5 models serialized to {models_dir}")

    return lr_model, knn_model, nb_model, rf_model, gb_model

def get_feature_importances(lr_model, rf_model, gb_model, feature_names):
    """
    Extracts the feature importances / coefficients for LR, RF, and GB models,
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
    lr_importance = sorted(lr_importance, key=lambda x: x["abs_coef"], reverse=True)

    # 2. Random Forest Gini Importance
    rf_importances = rf_model.feature_importances_
    rf_importance = []
    for name, imp in zip(feature_names, rf_importances):
        rf_importance.append({
            "feature": name,
            "importance": float(imp)
        })
    rf_importance = sorted(rf_importance, key=lambda x: x["importance"], reverse=True)

    # 3. Gradient Boosting Gini Importance
    gb_importances = gb_model.feature_importances_
    gb_importance = []
    for name, imp in zip(feature_names, gb_importances):
        gb_importance.append({
            "feature": name,
            "importance": float(imp)
        })
    gb_importance = sorted(gb_importance, key=lambda x: x["importance"], reverse=True)

    return lr_importance, rf_importance, gb_importance
