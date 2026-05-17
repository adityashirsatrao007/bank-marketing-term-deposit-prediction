import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from src.config import ASSETS_DIR

# Set aesthetic styling for plots to match a premium UI
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'axes.edgecolor': '#cccccc',
    'grid.color': '#f0f0f0'
})

# Premium color scheme matching the 5-model visual palette
COLOR_LR = '#3b82f6'  # Elegant Blue
COLOR_KNN = '#8b5cf6' # Vibrant Purple
COLOR_NB = '#f59e0b'  # Bright Amber
COLOR_RF = '#ec4899'  # Modern Pink
COLOR_GB = '#14b8a6'  # Mint Teal
COLOR_CORAL = '#f43f5e' # Coral Pink
COLOR_DARK = '#1e293b' # Slate Dark

def evaluate_and_plot(lr_model, knn_model, nb_model, rf_model, gb_model, X_test, y_test, feature_names):
    """
    Computes classification scores for all 5 algorithms and generates high-res metrics plots.
    """
    print("[Evaluation] Predicting probabilities on the test set for all 5 models...")
    
    # 1. Logistic Regression
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

    # 2. KNN
    y_pred_knn = knn_model.predict(X_test)
    y_prob_knn = knn_model.predict_proba(X_test)[:, 1]

    # 3. Naive Bayes
    y_pred_nb = nb_model.predict(X_test)
    y_prob_nb = nb_model.predict_proba(X_test)[:, 1]

    # 4. Random Forest
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    # 5. Gradient Boosting
    y_pred_gb = gb_model.predict(X_test)
    y_prob_gb = gb_model.predict_proba(X_test)[:, 1]

    # Compute classification metrics
    metrics = {
        "Logistic Regression": calculate_metrics(y_test, y_pred_lr, y_prob_lr),
        "K-Nearest Neighbors": calculate_metrics(y_test, y_pred_knn, y_prob_knn),
        "Naive Bayes": calculate_metrics(y_test, y_pred_nb, y_prob_nb),
        "Random Forest": calculate_metrics(y_test, y_pred_rf, y_prob_rf),
        "Gradient Boosting": calculate_metrics(y_test, y_pred_gb, y_prob_gb)
    }

    # Print summary benchmarking table
    print("\n" + "="*95)
    print(f"{'Model Architecture':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-"*95)
    for model_name, score_dict in metrics.items():
        print(f"{model_name:<25} | {score_dict['Accuracy']:<10.4f} | {score_dict['Precision']:<10.4f} | {score_dict['Recall']:<10.4f} | {score_dict['F1-Score']:<10.4f} | {score_dict['ROC-AUC']:<10.4f}")
    print("="*95 + "\n")

    # Generate and save high-resolution publication-ready visual assets
    plot_roc_curves(y_test, y_prob_lr, y_prob_knn, y_prob_nb, y_prob_rf, y_prob_gb, metrics)
    plot_confusion_matrices(y_test, y_pred_lr, y_pred_gb)
    plot_feature_importances(lr_model, gb_model, feature_names)

    return metrics

def calculate_metrics(y_true, y_pred, y_prob):
    """Helper to compile classification scores."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

def plot_roc_curves(y_test, y_prob_lr, y_prob_knn, y_prob_nb, y_prob_rf, y_prob_gb, metrics):
    """Plots and saves the combined ROC Curves comparing all 5 models."""
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
    fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)

    plt.figure(figsize=(8, 6.5), dpi=150)
    
    # Random guess baseline
    plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', linewidth=1.5, label='Random Guess (AUC = 0.50)')
    
    # 5 Model Curves
    plt.plot(fpr_lr, tpr_lr, color=COLOR_LR, linewidth=2.0, 
             label=f"Logistic Regression (AUC = {metrics['Logistic Regression']['ROC-AUC']:.4f})")
    plt.plot(fpr_knn, tpr_knn, color=COLOR_KNN, linewidth=2.0, 
             label=f"K-Nearest Neighbors (AUC = {metrics['K-Nearest Neighbors']['ROC-AUC']:.4f})")
    plt.plot(fpr_nb, tpr_nb, color=COLOR_NB, linewidth=2.0, 
             label=f"Naive Bayes (AUC = {metrics['Naive Bayes']['ROC-AUC']:.4f})")
    plt.plot(fpr_rf, tpr_rf, color=COLOR_RF, linewidth=2.0, 
             label=f"Random Forest (AUC = {metrics['Random Forest']['ROC-AUC']:.4f})")
    plt.plot(fpr_gb, tpr_gb, color=COLOR_GB, linewidth=2.5, 
             label=f"Gradient Boosting (AUC = {metrics['Gradient Boosting']['ROC-AUC']:.4f}) [Champion]")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', labelpad=10)
    plt.title('Receiver Operating Characteristic (ROC) Curves - 5 Algorithms Benchmark', pad=15, fontweight='bold')
    plt.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', framealpha=0.9, fontsize=9.5)
    plt.tight_layout()
    
    out_path = ASSETS_DIR / 'roc_curve.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[Evaluation] Combined 5-Model ROC Curve plot saved to {out_path}")

def plot_confusion_matrices(y_test, y_pred_lr, y_pred_gb):
    """Plots side-by-side Confusion Matrices as elegant heatmaps."""
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_gb = confusion_matrix(y_test, y_pred_gb)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)
    
    cms = [("Logistic Regression (Baseline)", cm_lr, COLOR_LR), ("Gradient Boosting (Champion)", cm_gb, COLOR_GB)]
    
    for i, (name, cm, color) in enumerate(cms):
        ax = axes[i]
        annot = np.array([
            [f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}"],
            [f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"]
        ])
        
        im = ax.imshow(cm, cmap=plt.cm.Blues, alpha=0.85)
        ax.grid(False)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No (0)', 'Yes (1)'])
        ax.set_yticklabels(['No (0)', 'Yes (1)'])
        ax.set_xlabel('Predicted Subscription', labelpad=8)
        ax.set_ylabel('Actual Subscription', labelpad=8)
        ax.set_title(f"{name}\nConfusion Matrix", pad=12, fontweight='bold')

        thresh = cm.max() / 2.
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                text_color = "white" if val > thresh else "black"
                ax.text(c, r, annot[r, c], ha="center", va="center", 
                        color=text_color, fontsize=12, fontweight='bold')
                
    plt.tight_layout()
    out_path = ASSETS_DIR / 'confusion_matrix.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[Evaluation] Confusion Matrix plot saved to {out_path}")

def plot_feature_importances(lr_model, gb_model, feature_names):
    """Plots side-by-side Feature Importance horizontal bar charts (top 10)."""
    # 1. Logistic Regression Coefficients (top 10 absolute impact)
    lr_coefs = lr_model.coef_[0]
    lr_abs_indices = np.argsort(np.abs(lr_coefs))[::-1][:10]
    lr_top_names = [feature_names[idx] for idx in lr_abs_indices]
    lr_top_coefs = [lr_coefs[idx] for idx in lr_abs_indices]
    
    # 2. Gradient Boosting Importances (top 10)
    gb_imps = gb_model.feature_importances_
    gb_indices = np.argsort(gb_imps)[::-1][:10]
    gb_top_names = [feature_names[idx] for idx in gb_indices]
    gb_top_imps = [gb_imps[idx] for idx in gb_indices]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    # LR plot
    y_pos = np.arange(10)
    colors_lr = [COLOR_GB if c > 0 else COLOR_CORAL for c in lr_top_coefs]
    axes[0].barh(y_pos, lr_top_coefs, align='center', color=colors_lr, alpha=0.85, height=0.6)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(lr_top_names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Coefficient Value (Effect Size)')
    axes[0].set_title('Logistic Regression Top 10 Coefficients\n(Teal = Positive, Coral = Negative Impact)', pad=12, fontweight='bold')
    axes[0].axvline(0, color='#94a3b8', linestyle='-', linewidth=1)
    axes[0].grid(axis='x', color='#e2e8f0', linestyle='--', alpha=0.7)

    # GB plot
    axes[1].barh(y_pos, gb_top_imps, align='center', color=COLOR_LR, alpha=0.85, height=0.6)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(gb_top_names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Gini Feature Importance')
    axes[1].set_title('Gradient Boosting Top 10 Features\n(Relative Impurity Reduction)', pad=12, fontweight='bold')
    axes[1].grid(axis='x', color='#e2e8f0', linestyle='--', alpha=0.7)

    plt.tight_layout()
    out_path = ASSETS_DIR / 'feature_importance.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[Evaluation] Feature Importance plot saved to {out_path}")
