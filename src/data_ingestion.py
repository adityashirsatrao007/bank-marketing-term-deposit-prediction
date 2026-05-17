import os
import urllib.request
import zipfile
import pandas as pd
from src.config import DATASET_URL, ZIP_PATH, EXTRACTED_DIR, BANK_ZIP_INNER, BANK_FULL_CSV, DATA_DIR

def download_data():
    """
    Downloads the UCI Bank Marketing dataset, extracts the zip,
    and moves 'bank-full.csv' to the target data folder.
    """
    if os.path.exists(BANK_FULL_CSV):
        print(f"[Ingestion] 'bank-full.csv' already exists at {BANK_FULL_CSV}. Skipping download.")
        return load_and_verify_data()

    print(f"[Ingestion] Downloading dataset from {DATASET_URL}...")
    try:
        # Download the top-level zip
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print(f"[Ingestion] Top-level zip downloaded to {ZIP_PATH}")

        # Extract top-level zip
        os.makedirs(EXTRACTED_DIR, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACTED_DIR)
        print(f"[Ingestion] Top-level zip extracted to {EXTRACTED_DIR}")

        # Inside the top-level zip, there is "bank.zip" which contains "bank-full.csv"
        if os.path.exists(BANK_ZIP_INNER):
            print(f"[Ingestion] Extracting inner zip: {BANK_ZIP_INNER}...")
            with zipfile.ZipFile(BANK_ZIP_INNER, 'r') as inner_zip:
                inner_zip.extract("bank-full.csv", path=DATA_DIR)
            print(f"[Ingestion] 'bank-full.csv' extracted directly to {DATA_DIR}")
        else:
            raise FileNotFoundError(f"Could not find inner zip 'bank.zip' at {BANK_ZIP_INNER}")

    except Exception as e:
        print(f"[Ingestion] ERROR during download and extraction: {e}")
        raise e
    
    # Cleanup temporary downloads if desired
    # We can keep them or delete. For now, keep them or clean them up.
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    import shutil
    if os.path.exists(EXTRACTED_DIR):
        shutil.rmtree(EXTRACTED_DIR)
    print("[Ingestion] Cleanup of temporary zip folders completed.")

    return load_and_verify_data()

def load_and_verify_data():
    """
    Loads bank-full.csv and validates its schema.
    """
    if not os.path.exists(BANK_FULL_CSV):
        raise FileNotFoundError(f"Data file not found at {BANK_FULL_CSV}")

    # The file bank-full.csv uses semicolon ';' as a delimiter
    df = pd.read_csv(BANK_FULL_CSV, sep=';')
    print(f"[Ingestion] Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Check shape
    expected_rows = 40000
    if df.shape[0] < expected_rows:
        print(f"[Ingestion] WARNING: Expected at least {expected_rows} rows, found {df.shape[0]}.")
    
    # Check columns
    expected_cols = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema verification failed! Missing expected columns: {missing_cols}")
    
    print("[Ingestion] Schema verification PASSED! Columns match expected features.")
    return df

if __name__ == "__main__":
    download_data()
