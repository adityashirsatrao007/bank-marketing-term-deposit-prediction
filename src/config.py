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

# File paths
DATASET_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
ZIP_PATH = DATA_DIR / "bank-marketing.zip"
EXTRACTED_DIR = DATA_DIR / "extracted"
BANK_ZIP_INNER = EXTRACTED_DIR / "bank.zip"
BANK_FULL_CSV = DATA_DIR / "bank-full.csv"
MODEL_BUNDLE_PATH = ASSETS_DIR / "model_bundle.json"

# ML Pipeline Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

# Logistic Regression Hyperparameters
LOG_REG_PARAMS = {
    "C": 0.1,
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "solver": "lbfgs"
}

# Gradient Boosting Hyperparameters
GRAD_BOOST_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 4,
    "random_state": RANDOM_STATE
}

# Feature definitions for UCI Bank Marketing Dataset
# Categorical columns and their expected categories (in alphabetical order for consistency)
CATEGORICAL_FEATURES = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["failure", "other", "success", "unknown"]
}

NUMERICAL_FEATURES = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

TARGET_COLUMN = "y"  # "yes" or "no"
