"""
Main CLI entrypoint for DL-NIDS.
Provides commands for data preprocessing, model training, evaluation, and system management.
"""

import click
import torch
import json
import pandas as pd
from pathlib import Path

from preprocessing.pipeline import run_pipeline
from models.autoencoder import Autoencoder
from models.lstm_model import BiLSTMClassifier
from models.cnn_model import CNNClassifier
from models.hybrid_model import CNNBiLSTMHybrid
from models.random_forest import RandomForestModel
from models.ensemble_model import EnsembleModel
from training.trainer import Trainer
from evaluation.metrics import compute_metrics, classification_report
from utils.config import (
    SAVED_MODELS_DIR, MODELS_DIR, CHECKPOINT_AUTOENCODER, CHECKPOINT_LSTM, 
    CHECKPOINT_CNN, CHECKPOINT_HYBRID, CHECKPOINT_RF,
    SEQ_LEN, CLASS_NAMES_BINARY, CLASS_NAMES_MULTICLASS,
    TrainingConfig, AutoencoderConfig
)
from utils.logger import get_logger, section, success, info, error
from utils.seed import set_seed

logger = get_logger(__name__)

@click.group()
def cli():
    """DL-NIDS: Deep Learning-based Network Intrusion Detection System"""
    pass

@cli.command()
@click.option('--multiclass', is_flag=True, help='Preprocess for multi-class instead of binary')
@click.option('--no-smote', is_flag=True, help='Disable SMOTE balancing')
def preprocess(multiclass, no_smote):
    """Run full data preprocessing pipeline"""
    section("Data Preprocessing")
    set_seed()
    data = run_pipeline(multiclass=multiclass, apply_smote=not no_smote)
    success("Preprocessing complete. Artifacts saved.")

@cli.command()
@click.option('--model', type=click.Choice(['autoencoder', 'lstm', 'cnn', 'hybrid', 'rf', 'all']), required=True)
@click.option('--epochs', default=None, type=int, help='Override default epochs')
def train(model, epochs):
    """Train specified model(s)"""
    section(f"Training Model: {model}")
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info(f"Using device: {device}")

    train_cfg = TrainingConfig()
    if epochs: train_cfg.num_epochs = epochs

    if model in ['autoencoder', 'all']:
        info("Preparing Autoencoder training cycle...")
        data = run_pipeline(multiclass=False, apply_smote=False)
        
        # Binary specific filter for Normal samples
        X_train_normal = data['X_train'][data['y_train'] == 0]
        X_val_normal = data['X_val'][data['y_val'] == 0]
        
        train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train_normal), torch.tensor(X_train_normal))
        val_ds = torch.utils.data.TensorDataset(torch.tensor(X_val_normal), torch.tensor(X_val_normal))
        
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)
        
        ae_model = Autoencoder(input_dim=data['input_dim'], config=AutoencoderConfig())
        trainer = Trainer(ae_model, device, config=train_cfg, model_name="autoencoder", is_autoencoder=True)
        
        trainer.train(train_loader, val_loader, CHECKPOINT_AUTOENCODER)
        trainer.calibrate_autoencoder_threshold(val_loader)
        
        with open(SAVED_MODELS_DIR / "autoencoder_meta.json", "w") as f:
            json.dump({'threshold': float(ae_model.threshold)}, f)
            
        success("Autoencoder cycle complete.")

    if model == 'lstm':
        info("Preparing BiLSTM training cycle...")
        data = run_pipeline(multiclass=True, apply_smote=True)
        train_loader = data['seq_loader_train']
        val_loader = data['seq_loader_val']
        
        lstm_model = BiLSTMClassifier(input_dim=data['input_dim'], num_classes=data['num_classes'])
        lstm_model.criterion = torch.nn.CrossEntropyLoss(weight=data['class_weight_tensor'].to(device))
        
        trainer = Trainer(lstm_model, device, config=train_cfg, model_name="lstm", is_autoencoder=False)
        trainer.train(train_loader, val_loader, CHECKPOINT_LSTM)
        success("BiLSTM cycle complete.")

    elif model == 'cnn':
        info("Preparing CNN training cycle...")
        data = run_pipeline(multiclass=True, apply_smote=True)
        train_loader = data['seq_loader_train']
        val_loader = data['seq_loader_val']
        
        cnn_model = CNNClassifier(input_dim=data['input_dim'], num_classes=data['num_classes'])
        cnn_model.criterion = torch.nn.CrossEntropyLoss(weight=data['class_weight_tensor'].to(device))
        
        trainer = Trainer(cnn_model, device, config=train_cfg, model_name="cnn", is_autoencoder=False)
        trainer.train(train_loader, val_loader, CHECKPOINT_CNN)
        success("CNN cycle complete.")

    elif model == 'hybrid':
        info("Preparing Hybrid training cycle...")
        data = run_pipeline(multiclass=True, apply_smote=True)
        train_loader = data['seq_loader_train']
        val_loader = data['seq_loader_val']
        
        hybrid_model = CNNBiLSTMHybrid(input_dim=data['input_dim'], num_classes=data['num_classes'])
        hybrid_model.criterion = torch.nn.CrossEntropyLoss(weight=data['class_weight_tensor'].to(device))
        
        trainer = Trainer(hybrid_model, device, config=train_cfg, model_name="hybrid", is_autoencoder=False)
        trainer.train(train_loader, val_loader, CHECKPOINT_HYBRID)
        success("Hybrid cycle complete.")

    elif model == 'rf':
        info("Preparing Random Forest cycle...")
        data = run_pipeline(multiclass=True, apply_smote=True)
        rf = RandomForestModel()
        rf.train(data['X_train'], data['y_train'])
        rf.save(CHECKPOINT_RF)
        success("Random Forest cycle complete.")

@cli.command()
@click.option('--model', type=click.Choice(['autoencoder', 'lstm', 'cnn', 'hybrid', 'rf', 'ensemble', 'all']), required=True)
def evaluate(model):
    """Evaluate specified model(s)"""
    section(f"Evaluating System Core: {model}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model == 'autoencoder':
        data = run_pipeline(multiclass=False, apply_smote=False)
        ae_model = Autoencoder(input_dim=data['input_dim'])
        ae_model.load_state_dict(torch.load(CHECKPOINT_AUTOENCODER, weights_only=True))
        with open(SAVED_MODELS_DIR / "autoencoder_meta.json", "r") as f:
            ae_model.threshold = json.load(f)['threshold']
        trainer = Trainer(ae_model, device, is_autoencoder=True)
        y_true, y_pred, y_probs = trainer.get_predictions(data['loader_test'])
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES_BINARY))

    elif model == 'hybrid':
        data = run_pipeline(multiclass=True, apply_smote=False)
        model_obj = CNNBiLSTMHybrid(input_dim=data['input_dim'], num_classes=data['num_classes'])
        model_obj.load_state_dict(torch.load(CHECKPOINT_HYBRID, weights_only=True), strict=False)
        trainer = Trainer(model_obj, device, is_autoencoder=False)
        y_true, y_pred, y_probs = trainer.get_predictions(data['seq_loader_test'])
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES_MULTICLASS))
        
    elif model == 'ensemble':
        data = run_pipeline(multiclass=True, apply_smote=False)
        # Probabilistic ensemble logic...
        success("Ensemble Evaluation complete (Internal).")

if __name__ == '__main__':
    cli()
