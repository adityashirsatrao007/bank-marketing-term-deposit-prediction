# Construction Plan: Bank Marketing Campaign Prediction

This construction plan serves as a local developer blueprint detailing the implementation tasks for the Bank Campaign Subscription Prediction model and its accompanying interactive web dashboard.

---

## Phase 1: Setup & Data Pipeline

### Step 1: Base Configuration & Ingestion
- **Context Brief**: Establish workspace structure. Automatically fetch, verify, and extract the UCI Bank Marketing Campaign dataset.
- **Tasks**:
  - Implement `/src/config.py` with default hyperparameters (seed 42, splits, directories).
  - Implement `/src/data_ingestion.py` using `urllib` to download the zip and extract `bank-full.csv`.
  - Verify that the downloaded file contains the correct number of rows (~41,188) and 17 columns.
- **Verification Commands**:
  - `python -c "from src.data_ingestion import download_data; download_data()"`
- **Exit Criteria**: Data folder contains `bank-full.csv` with a correct schema.

### Step 2: Custom Preprocessor & Splitter
- **Context Brief**: Build feature engineering rules. Handle highly imbalanced target values (`yes` = 11.7%, `no` = 88.3%). Scale numeric variables and encode categories consistently for both training and interactive Web JS predictions.
- **Tasks**:
  - Implement `/src/preprocessor.py` with custom category mappings and a fitted StandardScaler.
  - Implement stratified splitting to preserve class balances across train and test sets.
  - Save scaling metrics (mean, scale/variance) to standard output/file so client-side JavaScript can replicate scaling perfectly.
- **Verification Commands**:
  - `python -c "from src.preprocessor import preprocess_data; train_x, test_x = preprocess_data()"`
- **Exit Criteria**: Features are normalized, and category encoding arrays are locked down.

---

## Phase 2: Model Training & Evaluation

### Step 3: Estimators & Training Pipeline
- **Context Brief**: Train the two primary classifiers: Logistic Regression (L2 penalty) and Scikit-Learn Gradient Boosting. Compute feature importances.
- **Tasks**:
  - Implement `/src/models.py` to instantiate and fit models.
  - Extract Logistic Regression coefficients and Gradient Boosting tree feature importances.
  - Save fitted model instances using `pickle` to the filesystem.
- **Exit Criteria**: Models are trained and serialized; feature importances are extracted.

### Step 4: Metric Calculation & Visualizations
- **Context Brief**: Calculate offline metrics and plot publication-quality evaluation assets.
- **Tasks**:
  - Compute Confusion Matrix, Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
  - Generate Confusion Matrix heatmap, ROC curves with area scores, and horizontal feature importance charts as high-res PNGs in `web/assets/`.
- **Exit Criteria**: High-resolution evaluation charts exist in `/web/assets/`.

---

## Phase 3: Premium Web Dashboard

### Step 5: Web UI Layout & Aesthetics
- **Context Brief**: Construct a stunning glassmorphism dark-themed dashboard.
- **Tasks**:
  - Create `/web/index.html` with a premium header, model metric scoreboard, visual plot explorer, and live simulator.
  - Create `/web/style.css` using modern gradients, Outfield Google Fonts, glass effects, and CSS responsive structures.
- **Exit Criteria**: HTML and CSS are complete, responsive, and visually stunning.

### Step 6: Serverless Inference JS Engine
- **Context Brief**: Build a pure-JavaScript engine inside `web/app.js` that duplicates preprocessing and runs in-browser predictions without an active python server.
- **Tasks**:
  - Export fitted model weights and category maps from Python to `web/assets/model_bundle.json`.
  - Implement standard sigmoid logistic inference and decision tree predictions in JavaScript.
  - Handle form inputs, run inference, and animate prediction probability cards on screen.
- **Exit Criteria**: Complete client-side prediction matches Python outputs.

---

## Phase 4: Master CLI, Actions & GitHub Push

### Step 7: Execution Orchestrator (`run.py`)
- **Context Brief**: Standardize running instructions with a CLI interface.
- **Tasks**:
  - Write `/run.py` to coordinate the entire pipeline and start a local web server automatically.
- **Exit Criteria**: Running `python run.py` completes execution and hosts the server.

### Step 8: CI/CD Pages Deployment & Git Push
- **Context Brief**: Configure automation, create GitHub repo, and publish live dashboard to GitHub Pages.
- **Tasks**:
  - Setup `.github/workflows/deploy.yml`.
  - Create a public repo `bank-marketing-campaign-prediction` on `github.com/adityashirsatrao007` and push!
- **Exit Criteria**: Codebase is pushed and deployed live!
