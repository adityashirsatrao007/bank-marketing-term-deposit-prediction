import os
import urllib.request
import zipfile
import pandas as pd
from src.config import DATASET_URL, ZIP_PATH, EXTRACTED_DIR, BANK_ADDITIONAL_ZIP_INNER, BANK_ADDITIONAL_CSV, DATA_DIR

def download_data():
    """
    Downloads the UCI Bank Marketing dataset, extracts 'bank-additional.zip'
    from the main zip, and extracts 'bank-additional-full.csv' to the data folder.
    """
    if os.path.exists(BANK_ADDITIONAL_CSV):
        print(f"[Ingestion] 'bank-additional-full.csv' already exists at {BANK_ADDITIONAL_CSV}. Skipping download.")
        return load_and_verify_data()

    print(f"[Ingestion] Downloading dataset from {DATASET_URL}...")
    try:
        # Download the top-level zip
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print(f"[Ingestion] Top-level zip downloaded to {ZIP_PATH}")

        # Extract top-level zip to access bank-additional.zip
        os.makedirs(EXTRACTED_DIR, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACTED_DIR)
        print(f"[Ingestion] Top-level zip extracted to {EXTRACTED_DIR}")

        # Inside the top-level zip, there is "bank-additional.zip" which contains "bank-additional-full.csv"
        if os.path.exists(BANK_ADDITIONAL_ZIP_INNER):
            print(f"[Ingestion] Extracting inner zip: {BANK_ADDITIONAL_ZIP_INNER}...")
            with zipfile.ZipFile(BANK_ADDITIONAL_ZIP_INNER, 'r') as inner_zip:
                # The file is inside a folder named 'bank-additional' inside bank-additional.zip
                # Let's extract it
                inner_zip.extract("bank-additional/bank-additional-full.csv", path=DATA_DIR)
            
            # Move it up to DATA_DIR and rename to BANK_ADDITIONAL_CSV
            extracted_csv_path = DATA_DIR / "bank-additional" / "bank-additional-full.csv"
            if os.path.exists(extracted_csv_path):
                os.rename(extracted_csv_path, BANK_ADDITIONAL_CSV)
                # Remove empty folder
                os.rmdir(DATA_DIR / "bank-additional")
            print(f"[Ingestion] 'bank-additional-full.csv' extracted directly to {BANK_ADDITIONAL_CSV}")
        else:
            raise FileNotFoundError(f"Could not find inner zip 'bank-additional.zip' at {BANK_ADDITIONAL_ZIP_INNER}")

    except Exception as e:
        print(f"[Ingestion] ERROR during download and extraction: {e}")
        raise e
    
    # Cleanup temporary downloads
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    import shutil
    if os.path.exists(EXTRACTED_DIR):
        shutil.rmtree(EXTRACTED_DIR)
    print("[Ingestion] Cleanup of temporary zip folders completed.")

    return load_and_verify_data()

def load_and_verify_data():
    """
    Loads bank-additional-full.csv and validates its schema.
    """
    if not os.path.exists(BANK_ADDITIONAL_CSV):
        raise FileNotFoundError(f"Data file not found at {BANK_ADDITIONAL_CSV}")

    # The file bank-additional-full.csv uses semicolon ';' as a delimiter
    df = pd.read_csv(BANK_ADDITIONAL_CSV, sep=';')
    print(f"[Ingestion] Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Check shape - bank-additional-full must have exactly 41,188 rows
    expected_rows = 41188
    if df.shape[0] != expected_rows:
        print(f"[Ingestion] WARNING: Expected exactly {expected_rows} rows, found {df.shape[0]}.")
    
    # Check columns
    expected_cols = [
        "age", "job", "marital", "education", "default", "housing", "loan", 
        "contact", "month", "day_of_week", "duration", "campaign", "pdays", 
        "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", 
        "euribor3m", "nr.employed", "y"
    ]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema verification failed! Missing expected columns: {missing_cols}")
    
    print("[Ingestion] Schema verification PASSED! Columns match expected macroeconomic features.")
    return df

if __name__ == "__main__":
    download_data()
