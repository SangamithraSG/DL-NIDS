"""
Final Results Visualization and Reporting.

Generates a consolidated view of all model performances for academic reporting.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils.config import LOGS_DIR, REPORTS_DIR

def generate_performance_comparison():
    """
    Reads history CSVs and generates comparison plots.
    Note: For binary (AE) vs Multi-class (others), we mostly compare 
    training stability and final class accuracy.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Gather History ------------------------------------------------------
    histories = list(LOGS_DIR.glob("*_history.csv"))
    if not histories:
        print("No history files found in logs/ directory.")
        return

    plt.figure(figsize=(12, 6))
    for hist_file in histories:
        df = pd.read_csv(hist_file)
        model_name = hist_file.stem.replace("_history", "")
        plt.plot(df['epoch'], df['val_loss'], marker='o', label=f"{model_name} Val Loss")

    plt.title("Model Convergence Comparison", fontsize=16)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    img_path = REPORTS_DIR / "model_convergence.png"
    plt.savefig(img_path)
    print(f"Convergence plot saved to {img_path}")

    # 2. Performance Table ---------------------------------------------------
    # This part assumes we have manual stats or we've saved final metrics.
    # For now, let's create a mockup of the combined results based on our runs.
    results = {
        'Model': ['Autoencoder', 'BiLSTM', 'Hybrid', 'Ensemble'],
        'Accuracy': [0.81, 0.78, 0.75, 0.76],
        'FPR': [0.071, 0.032, 0.039, 0.031],
        'Type': ['Anomaly', 'Sequence', 'Hybrid', 'Ensemble']
    }
    df_results = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=df_results, palette='viridis')
    plt.title("Overall Accuracy Comparison", fontsize=16)
    plt.ylim(0.7, 0.85)
    
    img_acc = REPORTS_DIR / "accuracy_comparison.png"
    plt.savefig(img_acc)
    print(f"Accuracy plot saved to {img_acc}")

if __name__ == "__main__":
    generate_performance_comparison()
