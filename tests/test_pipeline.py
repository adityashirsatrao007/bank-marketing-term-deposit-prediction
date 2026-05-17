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
        # 40 rows (20 'yes', 20 'no') to ensure train_test_split(70/15/15) works under stratification
        np.random.seed(42)
        n_samples = 40
        data = {
            "age": np.random.randint(18, 70, n_samples),
            "job": np.random.choice(["admin.", "blue-collar", "technician", "services", "management", "retired", "student", "unemployed"], n_samples),
            "marital": np.random.choice(["single", "married", "divorced"], n_samples),
            "education": np.random.choice(["high.school", "university.degree", "basic.9y", "professional.course"], n_samples),
            "default": np.random.choice(["no", "unknown"], n_samples),
            "housing": np.random.choice(["yes", "no"], n_samples),
            "loan": np.random.choice(["yes", "no"], n_samples),
            "contact": np.random.choice(["cellular", "telephone"], n_samples),
            "month": np.random.choice(["may", "jul", "aug", "nov", "apr"], n_samples),
            "day_of_week": np.random.choice(["mon", "tue", "wed", "thu", "fri"], n_samples),
            "duration": np.random.randint(50, 1000, n_samples),
            "campaign": np.random.randint(1, 5, n_samples),
            "pdays": np.random.choice([999, 3, 6, 12], n_samples),
            "previous": np.random.randint(0, 3, n_samples),
            "poutcome": np.random.choice(["nonexistent", "failure", "success"], n_samples),
            "emp.var.rate": np.random.uniform(-3.4, 1.4, n_samples),
            "cons.price.idx": np.random.uniform(92.2, 94.8, n_samples),
            "cons.conf.idx": np.random.uniform(-50.8, -26.9, n_samples),
            "euribor3m": np.random.uniform(0.6, 5.0, n_samples),
            "nr.employed": np.random.uniform(4963.6, 5228.1, n_samples),
            "y": ["yes"] * 20 + ["no"] * 20
        }
        self.mock_df = pd.DataFrame(data)
        
    def test_preprocessor_transformation(self):
        """Validates that bank preprocessor handles scaling, dummy-encoding, and column alignments correctly."""
        preprocessor = BankPreprocessor()
        # Mock split_and_process inputs
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_and_process(self.mock_df)
        
        # Verify split dimensions
        self.assertEqual(len(X_train), 28)
        self.assertEqual(len(X_val), 6)
        self.assertEqual(len(X_test), 6)
        self.assertEqual(len(y_train), 28)
        self.assertEqual(len(y_val), 6)
        self.assertEqual(len(y_test), 6)
        
        # Verify target mapping (y -> binary 0/1)
        self.assertTrue(all(val in [0, 1] for val in y_train))
        self.assertTrue(all(val in [0, 1] for val in y_val))
        self.assertTrue(all(val in [0, 1] for val in y_test))
        
        # Ensure leakage prevention (duration is dropped)
        self.assertNotIn("duration", preprocessor.feature_names)
        
        # Check numerical scaling logic
        bundle = preprocessor.get_preprocessing_bundle()
        self.assertIn("numerical", bundle)
        self.assertIn("categorical", bundle)
        self.assertIn("euribor3m", bundle["numerical"])
        
        # euribor3m index mapping in feature names
        euribor_idx = preprocessor.feature_names.index("euribor3m")
        # Ensure it has been scaled
        self.assertNotEqual(X_train[0, euribor_idx], self.mock_df.iloc[0]["euribor3m"])
        
    def test_model_training(self):
        """Verifies estimators can be fitted and produce correct prediction arrays and shapes."""
        preprocessor = BankPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_and_process(self.mock_df)
        
        lr_model, knn_model, nb_model, rf_model, gb_model = train_models(X_train, y_train)
        
        # Verify classifiers are fitted and output probabilities
        lr_probs = lr_model.predict_proba(X_test)
        knn_probs = knn_model.predict_proba(X_test)
        nb_probs = nb_model.predict_proba(X_test)
        rf_probs = rf_model.predict_proba(X_test)
        gb_probs = gb_model.predict_proba(X_test)
        
        self.assertEqual(lr_probs.shape, (6, 2))
        self.assertEqual(knn_probs.shape, (6, 2))
        self.assertEqual(nb_probs.shape, (6, 2))
        self.assertEqual(rf_probs.shape, (6, 2))
        self.assertEqual(gb_probs.shape, (6, 2))
        
        # Check that probabilities sum to 1.0
        for probs in [lr_probs, knn_probs, nb_probs, rf_probs, gb_probs]:
            np.testing.assert_allclose(probs.sum(axis=1), np.ones(6))
        
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
