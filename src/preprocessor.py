import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE

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
        """
        # 1. Target encoding
        y = None
        if self.target_column in df.columns:
            y = df[self.target_column].map({"yes": 1, "no": 0}).values

        # 2. Scale numerical features
        num_data = df[self.numerical_features].copy()
        if is_training:
            scaled_num = self.scaler.fit_transform(num_data)
        else:
            scaled_num = self.scaler.transform(num_data)
        
        scaled_num_df = pd.DataFrame(scaled_num, columns=self.numerical_features, index=df.index)

        # 3. Custom One-Hot Encoding using categories defined in config.py
        # This prevents missing columns if a category is rare or absent in a small split.
        cat_dfs = []
        for col, categories in self.categorical_features.items():
            for cat in categories:
                col_name = f"{col}_{cat}"
                cat_dfs.append(pd.Series((df[col] == cat).astype(int), name=col_name, index=df.index))

        encoded_cat_df = pd.concat(cat_dfs, axis=1)

        # 4. Combine all features
        X_df = pd.concat([scaled_num_df, encoded_cat_df], axis=1)
        
        if is_training:
            self.feature_names = list(X_df.columns)
            print(f"[Preprocessor] Preprocessing complete. Extracted {len(self.feature_names)} features.")

        return X_df.values, y

    def split_and_process(self, df):
        """
        Splits the raw DataFrame into stratified training and testing sets,
        then fits the preprocessing pipeline on train and applies it to test.
        """
        # Stratified train-test split based on the target class to handle imbalance
        train_df, test_df = train_test_split(
            df, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=df[self.target_column]
        )
        
        print(f"[Preprocessor] Stratified Split:")
        print(f"  Train set: {train_df.shape[0]} rows (Positive rate: {np.mean(train_df[self.target_column] == 'yes'):.2%})")
        print(f"  Test set:  {test_df.shape[0]} rows (Positive rate: {np.mean(test_df[self.target_column] == 'yes'):.2%})")

        X_train, y_train = self.preprocess_df(train_df, is_training=True)
        X_test, y_test = self.preprocess_df(test_df, is_training=False)

        return X_train, X_test, y_train, y_test

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
    X_train, X_test, y_train, y_test = preprocessor.split_and_process(df)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
