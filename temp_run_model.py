"""
Porto Seguro Safe Driver Prediction - Advanced Model Training & Analysis
=========================================================================

This notebook implements a comprehensive machine learning pipeline for binary classification
with LGBM and XGBoost models, including:
- Cross-validation with stratified folds
- Hyperparameter tuning
- Feature importance analysis
- Model calibration & evaluation
- Rich visualizations for model interpretation

Author: Lucas
"""

import os, sys, json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================

# Determine ROOT directory
if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    ROOT = Path.cwd() if Path.cwd().name not in ("notebooks","tools","tests") else Path.cwd().parent

REPORTS_IN  = Path(os.getenv("REPORTS_IN")  or (ROOT / "reports"))
REPORTS_OUT = Path(os.getenv("REPORTS_OUT") or (ROOT / "reports_Lucas"))
REPORTS_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.data_loader import load_and_save_data

# Speed/Profile Configuration
SPEED = os.getenv("SPEED", "MEDIUM").upper().strip()

def get_speed_config() -> Dict[str, Any]:
    """Get configuration based on speed profile."""
    base = dict(CV=5, N_EST=6000, EARLY_STOP=200, MODELS=["lgbm","xgb"], LR=0.03)
    
    profiles = {
        "FAST": dict(CV=3, N_EST=2000, EARLY_STOP=50, MODELS=["lgbm"], LR=0.05),
        "MEDIUM": dict(CV=5, N_EST=4000, EARLY_STOP=100),
        "FULL": dict(CV=5, N_EST=8000, EARLY_STOP=300)
    }
    
    if SPEED in profiles:
        base.update(profiles[SPEED])
    return base

CFG = get_speed_config()

# Global parameters
RND     = int(os.getenv("RND", "42"))
CV      = int(os.getenv("CV", str(CFG["CV"])))
N_EST   = int(os.getenv("N_EST", str(CFG["N_EST"])))
ESR     = int(os.getenv("EARLY_STOP", str(CFG["EARLY_STOP"])))
MODELS  = [m.strip() for m in os.getenv("MODELS", ",".join(CFG["MODELS"])).split(",") if m.strip()]
IMB     = os.getenv("IMB", "spw").lower()  # 'iso' or 'spw'
LR      = float(os.getenv("LR", str(CFG["LR"])))
MEMBER  = os.getenv("MEMBER", "Lucas")

print(f"""
{'='*80}
CONFIGURATION
{'='*80}
Speed Profile: {SPEED}
Cross-Validation Folds: {CV}
Max Estimators: {N_EST}
Early Stopping Rounds: {ESR}
Models: {', '.join(MODELS)}
Imbalance Strategy: {IMB}
Learning Rate: {LR}
Random Seed: {RND}
{'='*80}
""")



# ============================================================================
# UTILITY FUNCTIONS - Data Processing
# ============================================================================

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve

def split_feature_types(cols: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Split features into categorical, binary, and numeric."""
    cat_cols = [c for c in cols if str(c).endswith("_cat")]
    bin_cols = [c for c in cols if str(c).endswith("_bin")]
    num_cols = [c for c in cols if c not in cat_cols and c not in bin_cols and c != "target"]
    return cat_cols, bin_cols, num_cols

def load_selected_features() -> List[str]:
    """Load selected features from feature gate output."""
    feature_file = REPORTS_IN / "features_selected.csv"
    if not feature_file.exists():
        raise FileNotFoundError(f"Missing {feature_file}. Run feature selection first.")
    
    df = pd.read_csv(feature_file)
    if "raw_feature" not in df.columns:
        raise ValueError("features_selected.csv must have 'raw_feature' column.")
    
    return df["raw_feature"].astype(str).tolist()

def add_engineered_features(X: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """Add engineered features if they're in the selected list."""
    X = X.copy()
    
    if "missing_count" in selected_features:
        X["missing_count"] = X.isna().sum(axis=1)
    
    if "sum_all_bin" in selected_features:
        bin_cols = [c for c in X.columns if str(c).endswith("_bin")]
        if bin_cols:
            # Convert to numeric first to avoid categorical sum error
            bin_data = X[bin_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            X["sum_all_bin"] = bin_data.sum(axis=1)
        else:
            X["sum_all_bin"] = 0
    
    return X

def prepare_data_for_trees(
    X: pd.DataFrame, 
    selected_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare data for tree-based models."""
    # First, add engineered features BEFORE converting to categorical
    X = add_engineered_features(X, selected_cols)
    
    # Keep only selected features that exist
    available_cols = [c for c in selected_cols if c in X.columns]
    missing_cols = [c for c in selected_cols if c not in X.columns]
    
    if missing_cols:
        print(f"⚠️  Warning: {len(missing_cols)} selected features not found in data")
    
    X = X[available_cols].copy()
    
    # NOW convert categorical features (after engineering features)
    cat_cols, _, _ = split_feature_types(X.columns)
    for col in cat_cols:
        try:
            X[col] = X[col].astype("category")
        except:
            print(f"⚠️  Warning: Could not convert {col} to category dtype")
    
    return X, cat_cols

def calculate_pos_weight(y: pd.Series) -> float:
    """Calculate scale_pos_weight for imbalanced data."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return float(n_neg / max(n_pos, 1))

print("✓ Data processing utilities loaded")



# ============================================================================
# VISUALIZATION FUNCTIONS - Enhanced Plots
# ============================================================================

def create_figure(figsize=(10, 6), title=None):
    """Create a nicely formatted figure."""
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    return fig, ax

def save_roc_curve(y_true, y_pred, output_path, model_name="Model"):
    """Plot ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    
    fig, ax = create_figure(figsize=(8, 8), title='ROC Curve')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.3)
    ax.plot(fpr, tpr, linewidth=2.5, label=f'{model_name} (AUC = {auc_score:.4f})')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ROC curve: {output_path.name}")

def save_pr_curve(y_true, y_pred, output_path, model_name="Model"):
    """Plot Precision-Recall curve with AP score."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap_score = average_precision_score(y_true, y_pred)
    baseline = (y_true == 1).sum() / len(y_true)
    
    fig, ax = create_figure(figsize=(8, 8), title='Precision-Recall Curve')
    ax.plot([0, 1], [baseline, baseline], 'k--', label=f'Baseline (No Skill = {baseline:.4f})', alpha=0.3)
    ax.plot(recall, precision, linewidth=2.5, label=f'{model_name} (AP = {ap_score:.4f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved PR curve: {output_path.name}")

def save_calibration_plot(y_true, y_pred, output_path, n_bins=20):
    """Plot calibration curve to assess probability calibration."""
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='quantile')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)
    ax1.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, label='Model')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('True Probability', fontsize=12)
    ax1.set_title('Calibration Curve', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Histogram of predictions
    ax2.hist(y_pred, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Predictions', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved calibration plot: {output_path.name}")

def save_confusion_matrix(y_true, y_pred, output_path, threshold=0.5):
    """Plot confusion matrix."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    fig, ax = create_figure(figsize=(8, 6), title=f'Confusion Matrix (threshold={threshold})')
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Claim', 'Claim'],
                yticklabels=['No Claim', 'Claim'],
                ax=ax, annot_kws={'size': 14})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    
    # Add accuracy metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
    ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix: {output_path.name}")

def save_feature_importance(importance_df, output_path, top_n=30, title="Feature Importance"):
    """Plot top N features by importance."""
    if importance_df is None or importance_df.empty:
        print("⚠️  No feature importance data available")
        return
    
    # Get top features
    top_features = importance_df.head(top_n).iloc[::-1]  # Reverse for better display
    
    fig, ax = create_figure(figsize=(10, max(8, top_n * 0.3)), title=title)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax.barh(range(len(top_features)), top_features.values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_features.values)):
        ax.text(value, i, f' {value:.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved feature importance: {output_path.name}")

def save_threshold_analysis(y_true, y_pred, output_path):
    """Plot metrics across different probability thresholds."""
    thresholds = np.linspace(0, 1, 100)
    precisions, recalls, f1_scores = [], [], []
    
    for thresh in thresholds:
        y_pred_binary = (y_pred >= thresh).astype(int)
        tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
        fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    fig, ax = create_figure(figsize=(10, 6), title='Metrics vs Probability Threshold')
    
    ax.plot(thresholds, precisions, label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, label='Recall', linewidth=2)
    ax.plot(thresholds, f1_scores, label='F1 Score', linewidth=2, linestyle='--')
    
    # Mark optimal F1 threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_thresh = thresholds[optimal_idx]
    ax.axvline(optimal_thresh, color='red', linestyle=':', alpha=0.5, 
               label=f'Optimal F1 Threshold = {optimal_thresh:.3f}')
    
    ax.set_xlabel('Probability Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved threshold analysis: {output_path.name}")

print("✓ Visualization functions loaded")



# ============================================================================
# MODEL TRAINING FUNCTIONS - LightGBM & XGBoost
# ============================================================================

def train_lightgbm_cv(X, y, cat_cols, params=None):
    """Train LightGBM with cross-validation and return OOF predictions."""
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    
    imb_kwargs = {"is_unbalance": True} if IMB == "iso" else {"is_unbalance": False}
    
    default_params = {
        "n_estimators": N_EST,
        "random_state": RND,
        "n_jobs": -1,
        "learning_rate": LR,
        "num_leaves": 128,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "max_bin": 511,
        "feature_pre_filter": False,
        **imb_kwargs
    }
    
    if params:
        default_params.update(params)
    
    skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=RND)
    oof_preds = np.zeros(len(y))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        clf = LGBMClassifier(**default_params)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            categorical_feature=[c for c in cat_cols if c in X_train.columns],
            callbacks=[early_stopping(ESR), log_evaluation(0)]
        )
        
        oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
        
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_ap = average_precision_score(y_val, oof_preds[val_idx])
        fold_scores.append({'fold': fold, 'roc_auc': fold_auc, 'pr_auc': fold_ap})
        print(f"  Fold {fold}/{CV} - ROC-AUC: {fold_auc:.5f}, PR-AUC: {fold_ap:.5f}")
    
    metrics = {
        'pr_auc': average_precision_score(y, oof_preds),
        'roc_auc': roc_auc_score(y, oof_preds),
        'brier': brier_score_loss(y, oof_preds),
        'oof': oof_preds,
        'fold_scores': fold_scores
    }
    
    return metrics

def train_xgboost_cv(X, y, cat_cols, params=None):
    """Train XGBoost with cross-validation and return OOF predictions."""
    import xgboost as xgb
    
    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "eta": LR,
        "max_depth": 6,
        "min_child_weight": 2.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "alpha": 0.0,
        "gamma": 0.0,
        "seed": RND,
        "nthread": -1,
    }
    
    if params:
        for key, val in params.items():
            if key == "learning_rate":
                default_params["eta"] = val
            elif key not in ["scale_pos_weight"]:  # Will be set per-fold
                default_params[key] = val
    
    skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=RND)
    oof_preds = np.zeros(len(y))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if IMB == "spw":
            default_params["scale_pos_weight"] = calculate_pos_weight(y_train)
        
        try:
            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        except:
            # Fallback to OHE if categorical not supported
            X_train_ohe = pd.get_dummies(X_train, columns=cat_cols, dummy_na=True)
            X_val_ohe = pd.get_dummies(X_val, columns=cat_cols, dummy_na=True).reindex(
                columns=X_train_ohe.columns, fill_value=0
            )
            dtrain = xgb.DMatrix(X_train_ohe, label=y_train)
            dval = xgb.DMatrix(X_val_ohe, label=y_val)
        
        model = xgb.train(
            params=default_params,
            dtrain=dtrain,
            num_boost_round=N_EST,
            evals=[(dval, "valid")],
            early_stopping_rounds=ESR,
            verbose_eval=False
        )
        
        oof_preds[val_idx] = model.predict(dval)
        
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_ap = average_precision_score(y_val, oof_preds[val_idx])
        fold_scores.append({'fold': fold, 'roc_auc': fold_auc, 'pr_auc': fold_ap})
        print(f"  Fold {fold}/{CV} - ROC-AUC: {fold_auc:.5f}, PR-AUC: {fold_ap:.5f}")
    
    metrics = {
        'pr_auc': average_precision_score(y, oof_preds),
        'roc_auc': roc_auc_score(y, oof_preds),
        'brier': brier_score_loss(y, oof_preds),
        'oof': oof_preds,
        'fold_scores': fold_scores
    }
    
    return metrics

def train_final_model(model_name, X_train, y_train, X_test, y_test, cat_cols, params=None):
    """Train final model on full training set and evaluate on holdout test set."""
    # Use 10% for validation in final training
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=RND
    )
    
    print(f"\n{'='*80}")
    print(f"Training final {model_name.upper()} model...")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    if model_name == "lgbm":
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        
        imb_kwargs = {"is_unbalance": True} if IMB == "iso" else {"is_unbalance": False}
        
        default_params = {
            "n_estimators": N_EST, "random_state": RND, "n_jobs": -1,
            "learning_rate": LR, "num_leaves": 128, "max_depth": -1,
            "min_child_samples": 20, "subsample": 0.9, "colsample_bytree": 0.9,
            "reg_lambda": 1.0, "reg_alpha": 0.0, "max_bin": 511,
            "feature_pre_filter": False, **imb_kwargs
        }
        
        if params:
            default_params.update(params)
        
        model = LGBMClassifier(**default_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            categorical_feature=[c for c in cat_cols if c in X_tr.columns],
            callbacks=[early_stopping(ESR), log_evaluation(0)]
        )
        
        train_time = time.time() - start_time
        
        pred_start = time.time()
        y_pred = model.predict_proba(X_test)[:, 1]
        pred_time_ms_per_1k = 1000 * (time.time() - pred_start) / (len(X_test) / 1000)
        
        # Feature importance
        try:
            importance = model.booster_.feature_importance(importance_type="gain")
            feature_names = model.booster_.feature_name()
            importance_df = pd.Series(importance, index=feature_names, name="gain").sort_values(ascending=False)
        except:
            importance_df = pd.Series(
                model.feature_importances_, 
                index=X_train.columns, 
                name="split"
            ).sort_values(ascending=False)
        
        metadata = {
            "encoder": "native(LGBM)",
            "best_iteration": getattr(model, "best_iteration_", None),
            "n_trees": getattr(model, "n_estimators_", None)
        }
    
    else:  # xgboost
        import xgboost as xgb
        
        default_params = {
            "objective": "binary:logistic", "eval_metric": "aucpr",
            "tree_method": "hist", "eta": LR, "max_depth": 6,
            "min_child_weight": 2.0, "subsample": 0.9, "colsample_bytree": 0.9,
            "lambda": 1.0, "alpha": 0.0, "gamma": 0.0, "seed": RND, "nthread": -1,
        }
        
        if params:
            for key, val in params.items():
                if key == "learning_rate":
                    default_params["eta"] = val
                else:
                    default_params[key] = val
        
        if IMB == "spw":
            default_params["scale_pos_weight"] = calculate_pos_weight(y_tr)
        
        try:
            dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
            dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
            dtest = xgb.DMatrix(X_test, enable_categorical=True)
            encoder_type = "native(XGB)"
        except:
            X_tr_ohe = pd.get_dummies(X_tr, columns=cat_cols, dummy_na=True)
            X_val_ohe = pd.get_dummies(X_val, columns=cat_cols, dummy_na=True).reindex(
                columns=X_tr_ohe.columns, fill_value=0
            )
            X_test_ohe = pd.get_dummies(X_test, columns=cat_cols, dummy_na=True).reindex(
                columns=X_tr_ohe.columns, fill_value=0
            )
            dtrain = xgb.DMatrix(X_tr_ohe, label=y_tr)
            dval = xgb.DMatrix(X_val_ohe, label=y_val)
            dtest = xgb.DMatrix(X_test_ohe)
            encoder_type = "OHE(XGB-fallback)"
        
        model = xgb.train(
            params=default_params, dtrain=dtrain, num_boost_round=N_EST,
            evals=[(dval, "valid")], early_stopping_rounds=ESR, verbose_eval=False
        )
        
        train_time = time.time() - start_time
        
        pred_start = time.time()
        y_pred = model.predict(dtest)
        pred_time_ms_per_1k = 1000 * (time.time() - pred_start) / (len(X_test) / 1000)
        
        # Feature importance
        importance = model.get_score(importance_type="gain")
        importance_df = pd.Series(importance, name="gain").sort_values(ascending=False)
        
        metadata = {
            "encoder": encoder_type,
            "best_iteration": getattr(model, "best_iteration", None),
            "n_trees": getattr(model, "best_ntree_limit", None)
        }
    
    # Evaluate
    test_metrics = {
        'pr_auc': average_precision_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred),
        'brier': brier_score_loss(y_test, y_pred)
    }
    
    metadata.update({
        "fit_time_s": train_time,
        "predict_time_ms_per_1k": pred_time_ms_per_1k,
        "model_obj": model
    })
    
    print(f"\n✓ Training completed in {train_time:.2f}s")
    print(f"  Holdout ROC-AUC: {test_metrics['roc_auc']:.5f}")
    print(f"  Holdout PR-AUC: {test_metrics['pr_auc']:.5f}")
    print(f"  Brier Score: {test_metrics['brier']:.5f}")
    
    return y_pred, test_metrics, importance_df, metadata

print("✓ Model training functions loaded")



# ============================================================================
# MAIN EXECUTION - Load Data & Run Cross-Validation
# ============================================================================

# Load split indices and features
split_file = REPORTS_IN / "split_indices.json"
features_file = REPORTS_IN / "features_selected.csv"

if not split_file.exists() or not features_file.exists():
    raise FileNotFoundError("Missing required files. Run feature selection and data split first.")

# Load data
print(f"\n{'='*80}")
print("LOADING DATA")
print(f"{'='*80}")

with open(split_file, 'r') as f:
    split_indices = json.load(f)

selected_features = load_selected_features()
print(f"✓ Loaded {len(selected_features)} selected features")

# Load full dataset
df_full = load_and_save_data().replace(-1, np.nan)

# Split into train/test
X_train_raw = df_full.loc[split_indices["train"]].drop(columns=["target"])
y_train = df_full.loc[split_indices["train"], "target"].astype(int)
X_test_raw = df_full.loc[split_indices["test"]].drop(columns=["target"])
y_test = df_full.loc[split_indices["test"], "target"].astype(int)

print(f"✓ Train set: {len(X_train_raw):,} samples")
print(f"✓ Test set: {len(X_test_raw):,} samples")
print(f"✓ Positive class ratio (train): {y_train.mean():.4f}")
print(f"✓ Positive class ratio (test): {y_test.mean():.4f}")

# Prepare data for tree-based models
X_train, cat_cols = prepare_data_for_trees(X_train_raw, selected_features)
X_test, _ = prepare_data_for_trees(X_test_raw, selected_features)

print(f"✓ Final feature count: {X_train.shape[1]}")
print(f"✓ Categorical features: {len(cat_cols)}")

# ============================================================================
# CROSS-VALIDATION - Baseline Models
# ============================================================================

print(f"\n{'='*80}")
print("BASELINE CROSS-VALIDATION")
print(f"{'='*80}\n")

cv_results = {}

# Check which models are available
available_models = []
for model_name in MODELS:
    if model_name == "lgbm":
        try:
            import lightgbm
            available_models.append("lgbm")
            print("✓ LightGBM available")
        except ImportError:
            print("⚠️  LightGBM not available, skipping")
    elif model_name == "xgb":
        try:
            import xgboost
            available_models.append("xgb")
            print("✓ XGBoost available")
        except ImportError:
            print("⚠️  XGBoost not available, skipping")

if not available_models:
    raise RuntimeError("No models available. Install lightgbm or xgboost.")

# Train baseline models with CV
for model_name in available_models:
    print(f"\n{'-'*80}")
    print(f"Training {model_name.upper()} with {CV}-fold CV...")
    print(f"{'-'*80}")
    
    if model_name == "lgbm":
        cv_results[model_name] = train_lightgbm_cv(X_train, y_train, cat_cols)
    elif model_name == "xgb":
        cv_results[model_name] = train_xgboost_cv(X_train, y_train, cat_cols)
    
    print(f"\n✓ {model_name.upper()} CV Results:")
    print(f"  ROC-AUC: {cv_results[model_name]['roc_auc']:.5f}")
    print(f"  PR-AUC: {cv_results[model_name]['pr_auc']:.5f}")
    print(f"  Brier Score: {cv_results[model_name]['brier']:.5f}")
    
    # Save OOF predictions
    pd.DataFrame({
        'oof_predictions': cv_results[model_name]['oof'],
        'y_true': y_train.values
    }).to_csv(REPORTS_OUT / f"oof_{model_name}.csv", index=False)

# ============================================================================
# MODEL COMPARISON TABLE
# ============================================================================

comparison_df = pd.DataFrame([
    {
        'model': model_name.upper(),
        'roc_auc': results['roc_auc'],
        'pr_auc': results['pr_auc'],
        'brier': results['brier']
    }
    for model_name, results in cv_results.items()
])

comparison_df = comparison_df.sort_values('pr_auc', ascending=False).reset_index(drop=True)
comparison_df.to_csv(REPORTS_OUT / "model_comparison_cv.csv", index=False)

print(f"\n{'='*80}")
print("MODEL COMPARISON (Cross-Validation)")
print(f"{'='*80}\n")
print(comparison_df.to_string(index=False))

# Select best model
best_model_name = comparison_df.iloc[0]['model'].lower()
print(f"\n{'='*80}")
print(f"✓ Best Model: {best_model_name.upper()} (PR-AUC: {comparison_df.iloc[0]['pr_auc']:.5f})")
print(f"{'='*80}")



# ============================================================================
# FINAL MODEL TRAINING & EVALUATION
# ============================================================================

# Train final model on full training set
y_pred_test, holdout_metrics, feature_importance, metadata = train_final_model(
    best_model_name, X_train, y_train, X_test, y_test, cat_cols
)

# Save results
pd.DataFrame({
    'y_pred': y_pred_test,
    'y_true': y_test.values
}).to_csv(REPORTS_OUT / "holdout_predictions.csv", index=False)

pd.DataFrame([holdout_metrics]).to_csv(REPORTS_OUT / "holdout_metrics.csv", index=False)

if feature_importance is not None and not feature_importance.empty:
    feature_importance.reset_index().rename(
        columns={'index': 'feature', 0: 'importance'}
    ).to_csv(REPORTS_OUT / "feature_importance.csv", index=False)

print("\n✓ Results saved to", REPORTS_OUT)



# ============================================================================
# COMPREHENSIVE VISUALIZATIONS
# ============================================================================

print(f"\n{'='*80}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*80}\n")

# 1. ROC Curve
save_roc_curve(
    y_test.values, y_pred_test,
    REPORTS_OUT / "plot_roc_curve.png",
    model_name=best_model_name.upper()
)

# 2. Precision-Recall Curve
save_pr_curve(
    y_test.values, y_pred_test,
    REPORTS_OUT / "plot_pr_curve.png",
    model_name=best_model_name.upper()
)

# 3. Calibration Plot
save_calibration_plot(
    y_test.values, y_pred_test,
    REPORTS_OUT / "plot_calibration.png"
)

# 4. Confusion Matrix
save_confusion_matrix(
    y_test.values, y_pred_test,
    REPORTS_OUT / "plot_confusion_matrix.png",
    threshold=0.5
)

# 5. Feature Importance (Top 30)
if feature_importance is not None and not feature_importance.empty:
    save_feature_importance(
        feature_importance,
        REPORTS_OUT / "plot_feature_importance_top30.png",
        top_n=30,
        title=f"Top 30 Features - {best_model_name.upper()}"
    )

# 6. Threshold Analysis
save_threshold_analysis(
    y_test.values, y_pred_test,
    REPORTS_OUT / "plot_threshold_analysis.png"
)

print(f"\n{'='*80}")
print("✓ All visualizations saved!")
print(f"{'='*80}")



# ============================================================================
# MODEL COMPARISON VISUALIZATION
# ============================================================================

if len(cv_results) > 1:
    print("\nCreating model comparison visualizations...")
    
    # Create comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(cv_results.keys())
    metrics_to_plot = ['roc_auc', 'pr_auc', 'brier']
    titles = ['ROC-AUC', 'PR-AUC (Average Precision)', 'Brier Score (lower is better)']
    colors_map = {'lgbm': '#2ecc71', 'xgb': '#3498db'}
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        values = [cv_results[m][metric] for m in models]
        colors = [colors_map.get(m, '#95a5a6') for m in models]
        
        bars = axes[idx].bar(
            [m.upper() for m in models], 
            values, 
            color=colors,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )
        
        axes[idx].set_title(title, fontsize=13, fontweight='bold')
        axes[idx].set_ylabel('Score', fontsize=11)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
        
        # For Brier score, invert y-axis to show lower is better
        if metric == 'brier':
            axes[idx].invert_yaxis()
    
    plt.suptitle('Model Performance Comparison (Cross-Validation)', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(REPORTS_OUT / "plot_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved model comparison plot")

# ============================================================================
# CV FOLD SCORES VISUALIZATION
# ============================================================================

print("\nCreating CV fold scores visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for model_name, results in cv_results.items():
    fold_scores_df = pd.DataFrame(results['fold_scores'])
    
    # ROC-AUC per fold
    axes[0].plot(
        fold_scores_df['fold'], 
        fold_scores_df['roc_auc'], 
        marker='o', 
        linewidth=2, 
        markersize=8,
        label=f"{model_name.upper()} (mean={results['roc_auc']:.4f})"
    )
    
    # PR-AUC per fold
    axes[1].plot(
        fold_scores_df['fold'], 
        fold_scores_df['pr_auc'], 
        marker='o', 
        linewidth=2, 
        markersize=8,
        label=f"{model_name.upper()} (mean={results['pr_auc']:.4f})"
    )

axes[0].set_xlabel('Fold', fontsize=12)
axes[0].set_ylabel('ROC-AUC', fontsize=12)
axes[0].set_title('ROC-AUC per CV Fold', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(range(1, CV+1))

axes[1].set_xlabel('Fold', fontsize=12)
axes[1].set_ylabel('PR-AUC', fontsize=12)
axes[1].set_title('PR-AUC per CV Fold', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range(1, CV+1))

plt.tight_layout()
plt.savefig(REPORTS_OUT / "plot_cv_fold_scores.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved CV fold scores plot")



# ============================================================================
# COMPREHENSIVE SUMMARY DASHBOARD
# ============================================================================

print(f"\n{'='*80}")
print(f"{'='*80}")
print(f"  FINAL SUMMARY - Porto Seguro Safe Driver Prediction")
print(f"{'='*80}")
print(f"{'='*80}\n")

# Dataset Summary
print(f"{'─'*80}")
print("📊 DATASET SUMMARY")
print(f"{'─'*80}")
print(f"  Training samples: {len(X_train):,}")
print(f"  Test samples: {len(X_test):,}")
print(f"  Total features: {X_train.shape[1]}")
print(f"  Categorical features: {len(cat_cols)}")
print(f"  Class imbalance (positive %): {y_train.mean()*100:.2f}%")
print()

# Cross-Validation Results
print(f"{'─'*80}")
print("🔄 CROSS-VALIDATION RESULTS")
print(f"{'─'*80}")
for model_name, results in cv_results.items():
    print(f"\n  {model_name.upper()}:")
    print(f"    ROC-AUC:     {results['roc_auc']:.5f}")
    print(f"    PR-AUC:      {results['pr_auc']:.5f}")
    print(f"    Brier Score: {results['brier']:.5f}")
print()

# Holdout Test Results
print(f"{'─'*80}")
print(f"🎯 HOLDOUT TEST RESULTS - {best_model_name.upper()}")
print(f"{'─'*80}")
print(f"  ROC-AUC:     {holdout_metrics['roc_auc']:.5f}")
print(f"  PR-AUC:      {holdout_metrics['pr_auc']:.5f}")
print(f"  Brier Score: {holdout_metrics['brier']:.5f}")
print()

# Model Metadata
print(f"{'─'*80}")
print("⚙️  MODEL METADATA")
print(f"{'─'*80}")
print(f"  Best Model: {best_model_name.upper()}")
print(f"  Encoder: {metadata['encoder']}")
print(f"  Best Iteration: {metadata['best_iteration']}")
print(f"  Number of Trees: {metadata['n_trees']}")
print(f"  Training Time: {metadata['fit_time_s']:.2f} seconds")
print(f"  Prediction Time: {metadata['predict_time_ms_per_1k']:.2f} ms/1k samples")
print()

# Top Features
if feature_importance is not None and not feature_importance.empty:
    print(f"{'─'*80}")
    print("⭐ TOP 10 MOST IMPORTANT FEATURES")
    print(f"{'─'*80}")
    top10 = feature_importance.head(10)
    for idx, (feature, importance) in enumerate(top10.items(), 1):
        print(f"  {idx:2d}. {feature:30s} {importance:>10.1f}")
    print()

# Files Generated
print(f"{'─'*80}")
print("💾 FILES GENERATED")
print(f"{'─'*80}")
output_files = [
    "model_comparison_cv.csv",
    "holdout_predictions.csv",
    "holdout_metrics.csv",
    "feature_importance.csv",
    "plot_roc_curve.png",
    "plot_pr_curve.png",
    "plot_calibration.png",
    "plot_confusion_matrix.png",
    "plot_feature_importance_top30.png",
    "plot_threshold_analysis.png",
    "plot_cv_fold_scores.png",
]

if len(cv_results) > 1:
    output_files.append("plot_model_comparison.png")

for file in output_files:
    if (REPORTS_OUT / file).exists():
        print(f"  ✓ {file}")

print(f"\n📁 All files saved to: {REPORTS_OUT}")

print(f"\n{'='*80}")
print(f"{'='*80}")
print("  ✅ MODEL TRAINING & EVALUATION COMPLETE!")
print(f"{'='*80}")
print(f"{'='*80}\n")



# ============================================================================
# BONUS: COMPREHENSIVE RESULTS DASHBOARD (Single Image)
# ============================================================================

print("\nCreating comprehensive results dashboard...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. ROC Curve
ax1 = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_test.values, y_pred_test)
auc_score = roc_auc_score(y_test.values, y_pred_test)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax1.plot(fpr, tpr, linewidth=2.5, label=f'Model (AUC={auc_score:.4f})')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve', fontweight='bold', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Precision-Recall Curve
ax2 = fig.add_subplot(gs[0, 1])
precision, recall, _ = precision_recall_curve(y_test.values, y_pred_test)
ap_score = average_precision_score(y_test.values, y_pred_test)
baseline = (y_test.values == 1).sum() / len(y_test)
ax2.plot([0, 1], [baseline, baseline], 'k--', alpha=0.3, label=f'Baseline ({baseline:.3f})')
ax2.plot(recall, precision, linewidth=2.5, label=f'Model (AP={ap_score:.4f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve', fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Calibration Curve
ax3 = fig.add_subplot(gs[0, 2])
prob_true, prob_pred = calibration_curve(y_test.values, y_pred_test, n_bins=15, strategy='quantile')
ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
ax3.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=6, label='Model')
ax3.set_xlabel('Predicted Probability')
ax3.set_ylabel('True Probability')
ax3.set_title('Calibration Curve', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Feature Importance (Top 15)
ax4 = fig.add_subplot(gs[1, :])
if feature_importance is not None and not feature_importance.empty:
    top15 = feature_importance.head(15).iloc[::-1]
    colors_fi = plt.cm.viridis(np.linspace(0.3, 0.9, len(top15)))
    bars = ax4.barh(range(len(top15)), top15.values, color=colors_fi, edgecolor='black', linewidth=0.5)
    ax4.set_yticks(range(len(top15)))
    ax4.set_yticklabels(top15.index, fontsize=10)
    ax4.set_xlabel('Importance (Gain)', fontsize=11)
    ax4.set_title('Top 15 Most Important Features', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='x')
    for i, (bar, value) in enumerate(zip(bars, top15.values)):
        ax4.text(value, i, f' {value:.0f}', va='center', fontsize=9)

# 5. Prediction Distribution
ax5 = fig.add_subplot(gs[2, 0])
ax5.hist(y_pred_test[y_test.values == 0], bins=50, alpha=0.6, label='Negative Class', edgecolor='black')
ax5.hist(y_pred_test[y_test.values == 1], bins=50, alpha=0.6, label='Positive Class', edgecolor='black')
ax5.set_xlabel('Predicted Probability')
ax5.set_ylabel('Count')
ax5.set_title('Prediction Distribution by True Class', fontweight='bold', fontsize=12)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Metrics Comparison (CV vs Holdout)
ax6 = fig.add_subplot(gs[2, 1])
metrics_names = ['ROC-AUC', 'PR-AUC', 'Brier']
cv_vals = [cv_results[best_model_name]['roc_auc'], 
           cv_results[best_model_name]['pr_auc'], 
           cv_results[best_model_name]['brier']]
holdout_vals = [holdout_metrics['roc_auc'], 
                holdout_metrics['pr_auc'], 
                holdout_metrics['brier']]

x = np.arange(len(metrics_names))
width = 0.35
bars1 = ax6.bar(x - width/2, cv_vals, width, label='CV Mean', alpha=0.8, edgecolor='black')
bars2 = ax6.bar(x + width/2, holdout_vals, width, label='Holdout', alpha=0.8, edgecolor='black')

ax6.set_ylabel('Score')
ax6.set_title('CV vs Holdout Performance', fontweight='bold', fontsize=12)
ax6.set_xticks(x)
ax6.set_xticklabels(metrics_names)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# 7. Model Summary Box
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

summary_text = f"""
MODEL SUMMARY
{'─'*30}

Best Model: {best_model_name.upper()}
Encoder: {metadata['encoder']}

PERFORMANCE (Holdout)
  • ROC-AUC: {holdout_metrics['roc_auc']:.5f}
  • PR-AUC:  {holdout_metrics['pr_auc']:.5f}
  • Brier:   {holdout_metrics['brier']:.5f}

TRAINING INFO
  • CV Folds: {CV}
  • Best Iteration: {metadata['best_iteration']}
  • Trees: {metadata['n_trees']}
  • Train Time: {metadata['fit_time_s']:.1f}s
  • Pred Time: {metadata['predict_time_ms_per_1k']:.1f} ms/1k

DATASET
  • Train: {len(X_train):,} samples
  • Test: {len(X_test):,} samples
  • Features: {X_train.shape[1]}
  • Imbalance: {y_train.mean()*100:.2f}%
"""

ax7.text(0.1, 0.95, summary_text, transform=ax7.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))

# Main title
fig.suptitle(f'Porto Seguro Safe Driver Prediction - {best_model_name.upper()} Model Dashboard', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(REPORTS_OUT / "plot_comprehensive_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved comprehensive dashboard: plot_comprehensive_dashboard.png")
print(f"\n🎉 All done! Check {REPORTS_OUT} for all outputs.")



# ============================================================================
# ENVIRONMENT & REPRODUCIBILITY INFO
# ============================================================================

import platform
import importlib

print(f"\n{'='*80}")
print("ENVIRONMENT & REPRODUCIBILITY")
print(f"{'='*80}")

packages = ["numpy", "pandas", "sklearn", "lightgbm", "xgboost", "matplotlib", "seaborn"]
versions = {}

for pkg in packages:
    try:
        module = importlib.import_module(pkg)
        version = getattr(module, "__version__", "unknown")
    except ImportError:
        version = "not installed"
    versions[pkg] = version

print(f"\nPython: {platform.python_version()}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"\nPackage Versions:")
for pkg, ver in versions.items():
    print(f"  {pkg:15s}: {ver}")

print(f"\nConfiguration:")
print(f"  Random Seed: {RND}")
print(f"  CV Folds: {CV}")
print(f"  Max Estimators: {N_EST}")
print(f"  Early Stopping: {ESR}")

print(f"\n✓ All runs are reproducible with fixed seed and split indices")
print(f"{'='*80}\n")


