import unittest
import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor import BankPreprocessor
from src.models import train_models
from src.export_model import serialize_decision_tree
from sklearn.tree import DecisionTreeClassifier

class TestAegisMLPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create a mock Bank Marketing dataframe representing diverse scenarios
        data = {
            "age": [30, 45, 25, 59, 32, 41, 28, 35, 50, 40],
            "job": ["admin.", "blue-collar", "technician", "services", "management", 
                    "retired", "entrepreneur", "self-employed", "housemaid", "student"],
            "marital": ["single", "married", "single", "divorced", "married", 
                        "married", "single", "married", "divorced", "single"],
            "education": ["secondary", "primary", "tertiary", "secondary", "tertiary",
                          "primary", "secondary", "tertiary", "primary", "secondary"],
            "default": ["no", "no", "no", "no", "no", "no", "no", "no", "no", "no"],
            "balance": [1500, 300, 8000, -100, 450, 12000, 0, 750, 5000, 250],
            "housing": ["yes", "no", "yes", "no", "yes", "no", "yes", "yes", "no", "no"],
            "loan": ["no", "no", "no", "no", "yes", "no", "no", "no", "no", "no"],
            "contact": ["cellular", "unknown", "cellular", "telephone", "cellular",
                        "cellular", "unknown", "cellular", "telephone", "cellular"],
            "day": [15, 5, 20, 8, 12, 19, 3, 22, 17, 30],
            "month": ["may", "jul", "aug", "nov", "apr", "jun", "feb", "sep", "oct", "dec"],
            "duration": [120, 350, 80, 600, 150, 210, 95, 400, 180, 90],
            "campaign": [1, 2, 1, 3, 1, 4, 2, 1, 2, 1],
            "pdays": [-1, -1, 150, -1, -1, 90, -1, 360, -1, -1],
            "previous": [0, 0, 2, 0, 0, 1, 0, 4, 0, 0],
            "poutcome": ["unknown", "unknown", "success", "unknown", "unknown",
                         "failure", "unknown", "other", "unknown", "unknown"],
            "y": ["no", "no", "yes", "no", "no", "yes", "no", "yes", "no", "no"]
        }
        self.mock_df = pd.DataFrame(data)
        
    def test_preprocessor_transformation(self):
        """Validates that bank preprocessor handles scaling, dummy-encoding, and column alignments correctly."""
        preprocessor = BankPreprocessor()
        # Mock split_and_process inputs
        X_train, X_test, y_train, y_test = preprocessor.split_and_process(self.mock_df)
        
        # Verify split dimensions
        self.assertEqual(len(X_train), 8)
        self.assertEqual(len(X_test), 2)
        self.assertEqual(len(y_train), 8)
        self.assertEqual(len(y_test), 2)
        
        # Verify target mapping (y -> binary 0/1)
        self.assertTrue(all(val in [0, 1] for val in y_train))
        self.assertTrue(all(val in [0, 1] for val in y_test))
        
        # Check numerical scaling logic
        bundle = preprocessor.get_preprocessing_bundle()
        self.assertIn("numerical", bundle)
        self.assertIn("categorical", bundle)
        self.assertIn("balance", bundle["numerical"])
        
        # Balance index mapping in feature names
        balance_idx = preprocessor.feature_names.index("balance")
        # Ensure it has been scaled
        self.assertNotEqual(X_train[0, balance_idx], self.mock_df.iloc[0]["balance"])
        
    def test_model_training(self):
        """Verifies estimators can be fitted and produce correct prediction arrays and shapes."""
        preprocessor = BankPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.split_and_process(self.mock_df)
        
        lr_model, gb_model = train_models(X_train, y_train)
        
        # Verify classifiers are fitted and output probabilities
        lr_probs = lr_model.predict_proba(X_test)
        gb_probs = gb_model.predict_proba(X_test)
        
        self.assertEqual(lr_probs.shape, (2, 2))
        self.assertEqual(gb_probs.shape, (2, 2))
        
        # Check that probabilities sum to 1.0
        np.testing.assert_allclose(lr_probs.sum(axis=1), np.ones(2))
        np.testing.assert_allclose(gb_probs.sum(axis=1), np.ones(2))
        
    def test_tree_serialization(self):
        """Validates recursive tree serialization dictionary structure."""
        # Fit a simple decision tree
        clf = DecisionTreeClassifier(max_depth=2)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        clf.fit(X, y)
        
        serialized = serialize_decision_tree(clf)
        
        # Root node check
        self.assertIn("leaf", serialized)
        if not serialized["leaf"]:
            self.assertIn("feature_idx", serialized)
            self.assertIn("threshold", serialized)
            self.assertIn("left", serialized)
            self.assertIn("right", serialized)

if __name__ == "__main__":
    unittest.main()
