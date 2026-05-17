import json
import numpy as np
from src.config import MODEL_BUNDLE_PATH

def export_to_json(preprocessor, lr_model, gb_model, metrics=None):
    """
    Translates trained estimators and preprocessing states into a single JSON bundle
    for client-side JavaScript execution.
    """
    print("[Export] Exporting pipeline parameters and model weights to JSON...")

    # 1. Preprocessing parameters
    bundle = preprocessor.get_preprocessing_bundle()
    
    # 1b. Evaluation metrics
    if metrics is not None:
        bundle["metrics"] = metrics

    # 2. Logistic Regression Parameters
    bundle["logistic_regression"] = {
        "coefficients": lr_model.coef_[0].tolist(),
        "intercept": float(lr_model.intercept_[0])
    }

    # 3. Gradient Boosting Parameters
    # Export recursive tree decision nodes
    trees_json = []
    for i in range(gb_model.n_estimators):
        # gb_model.estimators_[i, 0] is a DecisionTreeRegressor
        tree_estimator = gb_model.estimators_[i, 0]
        trees_json.append(serialize_decision_tree(tree_estimator))

    # Initial log-odds constant (prior)
    if hasattr(gb_model.init_, 'constant_'):
        init_val = float(gb_model.init_.constant_[0][0])
    elif hasattr(gb_model.init_, 'class_prior_'):
        prior = gb_model.init_.class_prior_[1]
        init_val = float(np.log(prior / (1.0 - prior)))
    elif hasattr(gb_model.init_, 'prior'):
        init_val = float(np.log(gb_model.init_.prior / (1.0 - gb_model.init_.prior)))
    else:
        init_val = 0.0

    bundle["gradient_boosting"] = {
        "n_estimators": int(gb_model.n_estimators),
        "learning_rate": float(gb_model.learning_rate),
        "init_value": init_val,
        "trees": trees_json
    }

    # Write out JSON
    with open(MODEL_BUNDLE_PATH, "w") as f:
        json.dump(bundle, f, indent=2)
        
    print(f"[Export] Success! Complete model bundle exported to {MODEL_BUNDLE_PATH}")

def serialize_decision_tree(tree_estimator):
    """
    Recursively serializes a scikit-learn DecisionTreeRegressor to a nested dict.
    """
    tree = tree_estimator.tree_

    def recurse(node_id):
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        if left_child == -1:  # Leaf node
            # For DecisionTreeRegressor, value has shape (node_count, 1, 1)
            return {
                "leaf": True,
                "value": float(tree.value[node_id][0][0])
            }
        else:
            return {
                "leaf": False,
                "feature_idx": int(tree.feature[node_id]),
                "threshold": float(tree.threshold[node_id]),
                "left": recurse(left_child),
                "right": recurse(right_child)
            }

    return recurse(0)
