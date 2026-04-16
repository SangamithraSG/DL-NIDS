"""
Performance evaluation metrics and visualization for DL-NIDS.

Computes accuracy, precision, recall, F1-score, and generates 
comprehensive plots including confusion matrices and error distributions.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve,
    classification_report
)
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from utils.config import REPORTS_DIR, INT_TO_CATEGORY, CLASS_NAMES_BINARY, CLASS_NAMES_MULTICLASS
from utils.logger import get_logger

logger = get_logger(__name__)

def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_probs: Optional[np.ndarray] = None,
    num_classes: int = 2
) -> Dict[str, Any]:
    """
    Compute overall classification metrics.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # FPR and FNR (Critical for Cybersecurity)
    # Binary interpretation: 0 is Normal, 1 is Attack
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    else:
        # For multi-class, we treat 'Normal' as index 0
        # FPR = (sum of all samples from other classes predicted as Normal) / (total samples not in Normal)
        # Actually standard definition of FPR per class is:
        # FPR = (False Positives for class i) / (Total negatives for class i)
        # We focus on the "Normal" vs "Any Attack" concept here:
        mask_normal = (y_true == 0)
        mask_attack = (y_true != 0)
        
        fp = np.sum((y_pred != 0) & mask_normal)
        tn = np.sum((y_pred == 0) & mask_normal)
        fn = np.sum((y_pred == 0) & mask_attack)
        tp = np.sum((y_pred != 0) & mask_attack)
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1,
        'fpr': fpr,
        'fnr': fnr,
        'cm': cm,
        'per_class_f1': f1_class.tolist()
    }
    
    if y_probs is not None:
        try:
            if num_classes == 2:
                metrics['auc_roc'] = roc_auc_score(y_true, y_probs)
            else:
                metrics['auc_roc'] = roc_auc_score(y_true, y_probs, multi_class='ovr')
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")
            metrics['auc_roc'] = 0.0
            
    return metrics

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str = "Confusion Matrix"):
    """
    Returns a Plotly confusion matrix heatmap.
    """
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale='Blues',
        title=title
    )
    return fig

def plot_reconstruction_error(errors: np.ndarray, threshold: float, y_true: np.ndarray):
    """
    Plot reconstruction error distribution for binary classes (Autoencoder).
    """
    df = pd.DataFrame({
        'Error': errors,
        'Label': ['Normal' if y == 0 else 'Attack' for y in y_true]
    })
    
    fig = px.histogram(
        df, x="Error", color="Label",
        marginal="box", barmode="overlay",
        title="Reconstruction Error Distribution",
        color_discrete_map={'Normal': '#00CC66', 'Attack': '#FF4444'}
    )
    
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Threshold: {threshold:.4f}")
    
    return fig

def generate_report_table(metrics_list: List[Dict[str, Any]], model_names: List[str]) -> pd.DataFrame:
    """
    Creates a comparison table for multiple models.
    """
    data = []
    for name, m in zip(model_names, metrics_list):
        data.append({
            'Model': name,
            'Accuracy': f"{m['accuracy']:.4f}",
            'F1-Macro': f"{m['f1_macro']:.4f}",
            'AUC-ROC': f"{m['auc_roc']:.4f}" if 'auc_roc' in m else "N/A",
            'FPR': f"{m['fpr']:.4f}",
            'FNR': f"{m['fnr']:.4f}"
        })
    return pd.DataFrame(data)
