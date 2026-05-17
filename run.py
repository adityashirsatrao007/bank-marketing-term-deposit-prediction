#!/usr/bin/env python3
"""
Aegis Pipeline Orchestrator & Dashboard Host
=============================================
This script runs the entire ML campaign classification pipeline (Ingestion,
Preprocessing, Model Training, Evaluation, Weights Serializations) and launches
a local lightweight HTTP server hosting the glassmorphic analytics dashboard.

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
    print("\n" + "="*70)
    print("        STARTING AEGIS BANK MARKETING CAMPAIGN PREDICTION PIPELINE")
    print("="*70 + "\n")

    # 1. Download & Extract Dataset
    df = download_data()

    # 2. Preprocess & Stratified Splitting
    preprocessor = BankPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.split_and_process(df)

    # 3. Train Classifiers (Logistic Regression + Gradient Boosting)
    lr_model, gb_model = train_models(X_train, y_train)

    # 4. Evaluate Test Performance & Generate High-Res PNG Visuals
    metrics = evaluate_and_plot(lr_model, gb_model, X_test, y_test, preprocessor.feature_names)

    # 5. Export Preprocessing Scales & Model Weights to JSON
    export_to_json(preprocessor, lr_model, gb_model, metrics)

    print("\n" + "="*70)
    print("                  ML CAMPAIGN PIPELINE EXECUTED SUCCESSFULLY")
    print("="*70 + "\n")
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
    parser = argparse.ArgumentParser(description="Aegis ML Campaign Orchestrator")
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
