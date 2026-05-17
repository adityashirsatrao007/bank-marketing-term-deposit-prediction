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

# Sleek slate and coral color scheme
COLOR_LR = '#3b82f6'  # Teal-ish Blue
COLOR_GB = '#14b8a6'  # Deep Mint/Teal
COLOR_DARK = '#1e293b' # Dark Slate
COLOR_CORAL = '#f43f5e' # Coral/Red

def evaluate_and_plot(lr_model, gb_model, X_test, y_test, feature_names):
    """
    Computes metrics for both models and generates professional evaluation plots.
    """
    print("[Evaluation] Predicting on test set...")
    # Predictions
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

    y_pred_gb = gb_model.predict(X_test)
    y_prob_gb = gb_model.predict_proba(X_test)[:, 1]

    # Compute classification metrics
    metrics = {
        "Logistic Regression": calculate_metrics(y_test, y_pred_lr, y_prob_lr),
        "Gradient Boosting": calculate_metrics(y_test, y_pred_gb, y_prob_gb)
    }

    # Print summary table
    print("\n" + "="*60)
    print(f"{'Model Metric':<25} | {'Logistic Regression':<20} | {'Gradient Boosting':<20}")
    print("-"*60)
    for m in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        print(f"{m:<25} | {metrics['Logistic Regression'][m]:<20.4f} | {metrics['Gradient Boosting'][m]:<20.4f}")
    print("="*60 + "\n")

    # Generate and save visual assets
    plot_roc_curve(y_test, y_prob_lr, y_prob_gb, metrics)
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

def plot_roc_curve(y_test, y_prob_lr, y_prob_gb, metrics):
    """Plots and saves the ROC Curve comparing both models."""
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)

    plt.figure(figsize=(8, 6.5), dpi=150)
    
    # Random guess baseline
    plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', linewidth=1.5, label='Random Guess (AUC = 0.50)')
    
    # Model curves
    plt.plot(fpr_lr, tpr_lr, color=COLOR_LR, linewidth=2.5, 
             label=f"Logistic Regression (AUC = {metrics['Logistic Regression']['ROC-AUC']:.4f})")
    plt.plot(fpr_gb, tpr_gb, color=COLOR_GB, linewidth=2.5, 
             label=f"Gradient Boosting (AUC = {metrics['Gradient Boosting']['ROC-AUC']:.4f})")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', labelpad=10)
    plt.title('Receiver Operating Characteristic (ROC) Curves', pad=15, fontweight='bold')
    plt.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', framealpha=0.9)
    plt.tight_layout()
    
    out_path = ASSETS_DIR / 'roc_curve.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[Evaluation] ROC Curve plot saved to {out_path}")

def plot_confusion_matrices(y_test, y_pred_lr, y_pred_gb):
    """Plots side-by-side Confusion Matrices as gorgeous heatmaps."""
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_gb = confusion_matrix(y_test, y_pred_gb)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)
    
    cms = [("Logistic Regression", cm_lr, COLOR_LR), ("Gradient Boosting", cm_gb, COLOR_GB)]
    
    for i, (name, cm, color) in enumerate(cms):
        ax = axes[i]
        
        # Display the matrix using an elegant single-color gradient
        # Represent normalized percentages visually
        annot = np.array([
            [f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}"],
            [f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"]
        ])
        
        # Draw heatmap manually for maximum control and premium aesthetics
        im = ax.imshow(cm, cmap=plt.cm.Blues, alpha=0.85)
        
        # Hide standard gridlines
        ax.grid(False)
        
        # Labels and ticks
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No (0)', 'Yes (1)'])
        ax.set_yticklabels(['No (0)', 'Yes (1)'])
        ax.set_xlabel('Predicted Subscription', labelpad=8)
        ax.set_ylabel('Actual Subscription', labelpad=8)
        ax.set_title(f"{name}\nConfusion Matrix", pad=12, fontweight='bold')

        # Annotation text inside squares
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
    # Color based on sign (positive vs negative impact)
    colors_lr = [COLOR_GB if c > 0 else COLOR_CORAL for c in lr_top_coefs]
    axes[0].barh(y_pos, lr_top_coefs, align='center', color=colors_lr, alpha=0.85, height=0.6)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(lr_top_names)
    axes[0].invert_yaxis()  # top-down
    axes[0].set_xlabel('Coefficient Value (Effect Size)')
    axes[0].set_title('Logistic Regression Top 10 Coefficients\n(Teal = Positive, Coral = Negative)', pad=12, fontweight='bold')
    axes[0].axvline(0, color='#94a3b8', linestyle='-', linewidth=1)
    axes[0].grid(axis='x', color='#e2e8f0', linestyle='--', alpha=0.7)

    # GB plot
    axes[1].barh(y_pos, gb_top_imps, align='center', color=COLOR_LR, alpha=0.85, height=0.6)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(gb_top_names)
    axes[1].invert_yaxis()  # top-down
    axes[1].set_xlabel('Gini Feature Importance')
    axes[1].set_title('Gradient Boosting Top 10 Features\n(Relative Impurity Reduction)', pad=12, fontweight='bold')
    axes[1].grid(axis='x', color='#e2e8f0', linestyle='--', alpha=0.7)

    plt.tight_layout()
    out_path = ASSETS_DIR / 'feature_importance.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[Evaluation] Feature Importance plot saved to {out_path}")
