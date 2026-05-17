import os
from pathlib import Path

# Paths
BASE_DIR = Path("/run/media/aditya/Z/AML")
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
WEB_DIR = BASE_DIR / "web"
ASSETS_DIR = WEB_DIR / "assets"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# File paths for B.Tech alignment (bank-additional-full)
DATASET_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
ZIP_PATH = DATA_DIR / "bank-marketing.zip"
EXTRACTED_DIR = DATA_DIR / "extracted"
BANK_ADDITIONAL_ZIP_INNER = EXTRACTED_DIR / "bank-additional.zip"
BANK_ADDITIONAL_CSV = DATA_DIR / "bank-additional-full.csv"
MODEL_BUNDLE_PATH = ASSETS_DIR / "model_bundle.json"

# ML Pipeline Configuration
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
CV_SPLITS = 5

# Logistic Regression Hyperparameters
LOG_REG_PARAMS = {
    "C": 0.1,
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "solver": "lbfgs"
}

# K-Nearest Neighbors Hyperparameters
KNN_PARAMS = {
    "n_neighbors": 5,
    "weights": "uniform",
    "metric": "minkowski",
    "p": 2,
    "n_jobs": -1
}

# Naive Bayes Hyperparameters (GaussianNB uses standard defaults)
NAIVE_BAYES_PARAMS = {}

# Random Forest Hyperparameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# Gradient Boosting Hyperparameters (Tuned Champion)
GRAD_BOOST_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": RANDOM_STATE
}

# Feature definitions for UCI Bank Marketing Dataset (bank-additional-full)
CATEGORICAL_FEATURES = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["divorced", "married", "single", "unknown"],
    "education": ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"],
    "default": ["no", "yes", "unknown"],
    "housing": ["no", "yes", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], # Actual 10 months in dataset
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "poutcome": ["failure", "nonexistent", "success"]
}

# Macroeconomic indicators and demography numeric features
NUMERICAL_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]

TARGET_COLUMN = "y"  # Target variable "yes" or "no"
