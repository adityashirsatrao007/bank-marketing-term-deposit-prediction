#!/usr/bin/env python3
"""
Orchid B.Tech ML Pipeline Orchestrator & Dashboard Host
======================================================
This script runs the entire refactored 5-model campaign classification pipeline:
1. Ingestion of the macroeconomic 'bank-additional-full.csv' dataset.
2. Leakage prevention (strictly dropping 'duration').
3. Partitioning (stratified 70/15/15 Train/Val/Test split).
4. Balancing (stratified undersampling to 50/50 ratio on the training partition).
5. Model Training (Logistic Regression, KNN, Naive Bayes, Random Forest, and Gradient Boosting).
6. Benchmarking & Multi-Model ROC plotting.
7. Serializing the model weights & preprocessing parameters for serverless client-side execution.
8. Launching a local lightweight server hosting the interactive dashboard.

Usage:
  python run.py                 # Run entire pipeline and host dashboard
  python run.py --pipeline-only  # Run ML pipeline only without starting web server
"""

import os
import sys
import argparse
import http.server
import socketserver
import webbrowser
import threading

# Append root directory to sys.path to enable src imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_ingestion import download_data
from src.preprocessor import BankPreprocessor
from src.models import train_models
from src.evaluation import evaluate_and_plot
from src.export_model import export_to_json
from src.config import WEB_DIR

PORT = 8080

def run_pipeline():
    """Executes the full machine learning classification pipeline."""
    print("\n" + "="*75)
    print("      STARTING B.TECH ALIGNED BANK TERM DEPOSIT PREDICTION PIPELINE")
    print("="*75 + "\n")

    # 1. Download & Extract Macroeconomic Dataset
    df = download_data()

    # 2. Preprocess & Stratified 70/15/15 Splitting + 50/50 Undersampling
    preprocessor = BankPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_and_process(df)

    # 3. Train all 5 classifiers (Logistic Regression, KNN, Naive Bayes, Random Forest, Gradient Boosting)
    lr_model, knn_model, nb_model, rf_model, gb_model = train_models(X_train, y_train)

    # 4. Evaluate Test Performance & Generate High-Res combined plots
    metrics = evaluate_and_plot(
        lr_model, knn_model, nb_model, rf_model, gb_model, 
        X_test, y_test, preprocessor.feature_names
    )

    # 5. Export Preprocessing Scales & Model Weights to JSON for local dashboard inference
    export_to_json(preprocessor, lr_model, gb_model, metrics)

    print("\n" + "="*75)
    print("                B.TECH ML PIPELINE EXECUTED SUCCESSFULLY")
    print("="*75 + "\n")
    return True

def host_dashboard():
    """Hosts the interactive web dashboard on a local server."""
    cwd = os.getcwd()
    try:
        # Move to web folder to serve assets cleanly
        os.chdir(WEB_DIR)
        
        Handler = http.server.SimpleHTTPRequestHandler
        socketserver.TCPServer.allow_reuse_address = True
        
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"[Server] Hosting interactive dashboard at http://localhost:{PORT}")
            print("[Server] Press Ctrl+C to stop hosting.")
            
            # Automatically open user browser in a non-blocking thread
            threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n[Server] Shutting down HTTP server.")
    except Exception as e:
        print(f"[Server] ERROR starting server: {e}")
    finally:
        os.chdir(cwd)

def main():
    parser = argparse.ArgumentParser(description="Orchid B.Tech ML Campaign Orchestrator")
    parser.add_argument(
        "--pipeline-only", 
        action="store_true", 
        help="Execute training and plotting without hosting the dashboard"
    )
    args = parser.parse_args()

    # Run ML Pipeline
    success = run_pipeline()

    # Launch Local Dashboard Server unless requested otherwise
    if success and not args.pipeline_only:
        host_dashboard()

if __name__ == "__main__":
    main()
