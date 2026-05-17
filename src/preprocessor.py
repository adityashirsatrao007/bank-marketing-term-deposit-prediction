import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, RANDOM_STATE

class BankPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.numerical_features = NUMERICAL_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        self.target_column = TARGET_COLUMN

    def preprocess_df(self, df, is_training=False):
        """
        Transforms a raw DataFrame into a fully encoded and scaled numerical matrix.
        Guarantees deterministic feature alignment by using the pre-defined categories in config.py.
        Strips 'duration' if present to prevent data leakage.
        """
        # Ensure 'duration' is explicitly and strictly dropped to prevent data leakage
        if "duration" in df.columns:
            print("[Preprocessor] Leakage Prevention: 'duration' column successfully dropped.")
            df_cleaned = df.drop(columns=["duration"])
        else:
            df_cleaned = df.copy()

        # 1. Target encoding
        y = None
        if self.target_column in df_cleaned.columns:
            y = df_cleaned[self.target_column].map({"yes": 1, "no": 0}).values

        # 2. Scale numerical features
        num_data = df_cleaned[self.numerical_features].copy()
        if is_training:
            scaled_num = self.scaler.fit_transform(num_data)
        else:
            scaled_num = self.scaler.transform(num_data)
        
        scaled_num_df = pd.DataFrame(scaled_num, columns=self.numerical_features, index=df_cleaned.index)

        # 3. Custom One-Hot Encoding using categories defined in config.py
        # This prevents missing columns if a category is rare or absent in a small split.
        cat_dfs = []
        for col, categories in self.categorical_features.items():
            for cat in categories:
                col_name = f"{col}_{cat}"
                cat_dfs.append(pd.Series((df_cleaned[col] == cat).astype(int), name=col_name, index=df_cleaned.index))

        encoded_cat_df = pd.concat(cat_dfs, axis=1)

        # 4. Combine all features
        X_df = pd.concat([scaled_num_df, encoded_cat_df], axis=1)
        
        if is_training:
            self.feature_names = list(X_df.columns)
            print(f"[Preprocessor] Preprocessing complete. Extracted {len(self.feature_names)} features.")

        return X_df.values, y

    def split_and_process(self, df):
        """
        Splits the raw DataFrame into 70% Training, 15% Validation, and 15% Testing stratified sets.
        Applies stratified undersampling exclusively to the training set for a 50/50 balance ratio.
        """
        # 1. First split: 70% Train, 30% Temp (Validation + Test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.30, 
            random_state=RANDOM_STATE, 
            stratify=df[self.target_column]
        )
        
        # 2. Second split: split Temp into 15% Validation and 15% Testing
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=temp_df[self.target_column]
        )

        print("[Preprocessor] Initial Stratified Splitting:")
        print(f"  Training Split:   {train_df.shape[0]} rows (Positive rate: {np.mean(train_df[self.target_column] == 'yes'):.2%})")
        print(f"  Validation Split: {val_df.shape[0]} rows (Positive rate: {np.mean(val_df[self.target_column] == 'yes'):.2%})")
        print(f"  Testing Split:    {test_df.shape[0]} rows (Positive rate: {np.mean(test_df[self.target_column] == 'yes'):.2%})")

        # 3. Apply Stratified Undersampling strictly on the Training set to balance class distribution to 50/50
        pos_train = train_df[train_df[self.target_column] == "yes"]
        neg_train = train_df[train_df[self.target_column] == "no"]

        n_pos = len(pos_train)
        neg_train_undersampled = neg_train.sample(n=n_pos, random_state=RANDOM_STATE)
        
        balanced_train_df = pd.concat([pos_train, neg_train_undersampled]).sample(frac=1, random_state=RANDOM_STATE)
        
        print("[Preprocessor] After Stratified Training Undersampling:")
        print(f"  Balanced Training Split: {balanced_train_df.shape[0]} rows (Positive rate: {np.mean(balanced_train_df[self.target_column] == 'yes'):.2%})")

        # 4. Preprocess all splits (fit scaler on balanced training data, transform validation & test)
        X_train, y_train = self.preprocess_df(balanced_train_df, is_training=True)
        X_val, y_val = self.preprocess_df(val_df, is_training=False)
        X_test, y_test = self.preprocess_df(test_df, is_training=False)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_preprocessing_bundle(self):
        """
        Returns a dictionary representation of preprocessor configurations
        (means, scales, category lists) to be consumed by Client JS.
        """
        bundle = {
            "numerical": {
                name: {
                    "mean": float(self.scaler.mean_[i]),
                    "scale": float(self.scaler.scale_[i])
                }
                for i, name in enumerate(self.numerical_features)
            },
            "categorical": self.categorical_features,
            "feature_names": self.feature_names
        }
        return bundle

if __name__ == "__main__":
    from src.data_ingestion import download_data
    df = download_data()
    preprocessor = BankPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_and_process(df)
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)
