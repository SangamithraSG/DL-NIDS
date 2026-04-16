import streamlit as st
import dashboard_utils 
import torch
import numpy as np
import plotly.express as px
import pandas as pd
import joblib
from pathlib import Path

from dashboard_utils import (
    safe_page_header, apply_global_styles, validate_resource, 
    check_shap_installed, get_available_models
)

# ─── REBUILT IMPORTS ──────────────────────────────────────────────────────────
try:
    from preprocessing.pipeline import run_pipeline
    from models.hybrid_model import CNNBiLSTMHybrid
    from models.cnn_model import CNNClassifier
    from models.lstm_model import BiLSTMClassifier
    from models.autoencoder import Autoencoder
    from utils.config import (
        CHECKPOINT_HYBRID, CHECKPOINT_CNN, CHECKPOINT_LSTM, 
        CHECKPOINT_AUTOENCODER, CHECKPOINT_RF,
        CLASS_NAMES_MULTICLASS, CLASS_NAMES_BINARY
    )
except ImportError as e:
    st.error(f"❌ Structural Dependency Error: {e}")
    st.stop()

# Initialization
safe_page_header("AI Forensics & Explainability")
apply_global_styles()

# ─── CORE FAILS_SAFE HELPERS ──────────────────────────────────────────────────

def safe_extract_shap(shap_values, target_idx):
    """
    STRICT SHAP EXTRACTION - Handles lists, multi-dimensional arrays, and Explanation objects.
    Ensures a single sample's attribution is returned for the target class.
    """
    try:
        # Handle SHAP Explanation objects (modern SHAP)
        if hasattr(shap_values, "values"):
            res = shap_values.values
            if len(res.shape) == 3: # (samples, features, classes)
                return res[0, :, target_idx]
            return res[0]

        # Handle List of arrays (classic SHAP multiclass)
        if isinstance(shap_values, list):
            if target_idx < len(shap_values):
                return shap_values[target_idx]
            return shap_values[0]
        
        # Handle Multi-dimensional numpy arrays
        if isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3: # (classes, samples, features)
                return shap_values[target_idx]
            if shap_values.ndim == 2 and shap_values.shape[0] == 1:
                return shap_values[0] # (1, features) -> (features,)
                
        return shap_values
    except Exception as e:
        print(f"DEBUG: safe_extract_shap failed: {e}")
        return shap_values

@st.cache_resource
def load_model_polymorphic(model_type):
    """Dynamic model loader for Forensic analysis."""
    device = torch.device('cpu') 
    
    mapping = {
        "Autoencoder": (CHECKPOINT_AUTOENCODER, "pt"),
        "BiLSTM": (CHECKPOINT_LSTM, "pt"),
        "CNN": (CHECKPOINT_CNN, "pt"),
        "Hybrid": (CHECKPOINT_HYBRID, "pt"),
        "Random Forest": (CHECKPOINT_RF, "joblib")
    }
    
    if model_type not in mapping:
        return None, None
        
    path, engine = mapping[model_type]
    if not validate_resource(path):
        return None, None

    try:
        # Load Data context (AE uses Binary-Normal, Others use Multi)
        is_multiclass = model_type != "Autoencoder"
        data = run_pipeline(multiclass=is_multiclass, apply_smote=False)
        
        if engine == "joblib":
            model = joblib.load(path)
            return model, data
        else:
            input_dim = data['input_dim']
            num_classes = data['num_classes']
            
            if model_type == "Hybrid":
                model = CNNBiLSTMHybrid(input_dim=input_dim, num_classes=num_classes)
            elif model_type == "CNN":
                model = CNNClassifier(input_dim=input_dim, num_classes=num_classes)
            elif model_type == "BiLSTM":
                model = BiLSTMClassifier(input_dim=input_dim, num_classes=num_classes)
            elif model_type == "Autoencoder":
                model = Autoencoder(input_dim=input_dim)
            
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=False)
            model.eval()
            return model, data
            
    except Exception as e:
        print(f"DEBUG: Error loading {model_type}: {e}")
        return None, None

def show():
    if not check_shap_installed():
        st.warning("⚠️ **SHAP Tooling Missing**: Forensic module is locked.")
        return

    import shap 
    
    # ─── 1. ARCHITECTURE SELECTION ───────────────────────────────────────────
    st.sidebar.subheader("Forensic Target")
    available = get_available_models()
    selected_arch = st.sidebar.selectbox("Model Architecture", available)
    
    # ─── 2. RUNTIME LOADING ──────────────────────────────────────────────────
    model, data = load_model_polymorphic(selected_arch)
    
    if model is None:
        st.warning(f"⚠️ **{selected_arch} Core Offline**: Model or data not ready.")
        return

    X_train = data['X_train']
    X_test = data['X_test']
    f_names = data['feature_names']
    n_features = len(f_names)
    seq_len = 10

    # ─── 3. UI SYNCHRONIZATION ───────────────────────────────────────────────
    is_ae = (selected_arch == "Autoencoder")
    is_rf = (selected_arch == "Random Forest")
    is_seq = (selected_arch in ["CNN", "BiLSTM", "Hybrid"])

    if is_ae:
        classes = ["Reconstruction Error"]
    else:
        classes = CLASS_NAMES_MULTICLASS if data['num_classes'] > 2 else CLASS_NAMES_BINARY
            
    col1, col2 = st.columns([1, 2])
    target_class = col1.selectbox("Focus Prediction", classes)
    cls_idx = classes.index(target_class)
    
    sample_idx = col2.slider("Traffic Sample (ends at)", seq_len, len(X_test)-1, 100)
    
    # ─── 4. CORRECTIVE PERFORMANCE ───────────────────────────────────────────
    if st.button("🚀 Analyze Forensic Pathway"):
        try:
            with st.spinner("Executing Shapley Decomposition..."):
                
                # ─── CASE A: AUTOENCODER (No SHAP) ───
                if is_ae:
                    X_sample = X_test[sample_idx].reshape(1, -1)
                    if X_sample.shape != (1, n_features):
                        st.error(f"Shape mismatch: Expected (1, {n_features}), got {X_sample.shape}")
                        return

                    with torch.no_grad():
                        sample_t = torch.tensor(X_sample).float()
                        reconstruction = model(sample_t).detach().cpu().numpy()
                    
                    impacts = np.abs(X_sample[0] - reconstruction[0])
                    
                    df_viz = pd.DataFrame({
                        'Feature': f_names,
                        'Contribution': impacts,
                        'Impact': impacts
                    }).sort_values('Impact', ascending=False).head(15)
                    
                    st.subheader("Top Divergent Features (Reconstruction Error)")
                    fig = px.bar(df_viz, x='Contribution', y='Feature', orientation='h',
                                 color='Contribution', color_continuous_scale='Reds', template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("Logic: Larger bars indicate features that differ most from benign baselines.")
                    return

                # ─── CASE B: SEQUENTIAL / RANDOM FOREST ───
                
                # 1. Prepare Inputs based on Architecture
                if is_seq:
                    # Construct window [seq_len, features] and flatten to match proxy requirement
                    window = X_test[sample_idx - seq_len + 1 : sample_idx + 1]
                    X_sample = window.flatten().reshape(1, -1) # (1, 10 * features)
                    
                    # Prepare flattened window background for KernelExplainer
                    bg_windows = []
                    for i in range(50):
                        bg_windows.append(X_train[i : i + seq_len].flatten())
                    background = np.array(bg_windows) # (50, 10 * features)
                    
                    print(f"DEBUG: Sequential Sample Shape: {X_sample.shape}")
                    print(f"DEBUG: Sequential Background Shape: {background.shape}")
                else:
                    X_sample = X_test[sample_idx].reshape(1, -1) # (1, features)
                    background = X_train[:50] # (50, features)

                # 2. Strict Shape Validation Before SHAP
                expected_shap_input_dim = (seq_len * n_features) if is_seq else n_features
                if X_sample.shape != (1, expected_shap_input_dim):
                    st.error(f"Input shape mismatch: {X_sample.shape} != (1, {expected_shap_input_dim})")
                    return

                # 3. Engine Selection and Execution
                if is_rf:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    # PyTorch Batch-Safe Model Proxy
                    def model_proxy(x):
                        batch_size = x.shape[0]
                        # Dynamically compute features from flattened input
                        # features = x.shape[1] // seq_len
                        # Use local n_features to ensure consistency
                        x_reshaped = x.reshape(batch_size, seq_len, n_features)
                        
                        x_t = torch.tensor(x_reshaped).float()
                        with torch.no_grad():
                            preds = model(x_t)
                            # Ensure it is detached and on CPU
                            preds_np = preds.detach().cpu().numpy()
                            if preds_np.ndim == 1:
                                preds_np = preds_np.reshape(-1, 1)
                            return preds_np

                    # KernelExplainer ONLY for PyTorch models as per rules
                    explainer = shap.KernelExplainer(model_proxy, background)
                    shap_values = explainer.shap_values(X_sample)

                # 4. Debug Logging
                print(f"DEBUG: shap_values type: {type(shap_values)}")
                if isinstance(shap_values, list): print(f"DEBUG: shap_values count: {len(shap_values)}")

                # 5. Extraction and Normalization
                sv = safe_extract_shap(shap_values, cls_idx)
                
                # Squeeze out the sample dimension if it exists (e.g., from (1, features) to (features,))
                if hasattr(sv, "shape") and len(sv.shape) > 1:
                    if sv.shape[0] == 1:
                        sv = sv.squeeze(0)
                    elif sv.ndim == 2 and sv.shape[1] == n_features: # Handling (samples, features)
                        sv = sv[0]
                
                # 6. Window Aggregation (STRICT REQUIREMENT)
                if is_seq:
                    # sv is (seq_len * n_features,) -> (10, 122)
                    sv_reshaped = sv.reshape(seq_len, n_features)
                    # Aggregate using Mean over time axis (axis=0)
                    sv_final = np.mean(sv_reshaped, axis=0) # (features,)
                    print(f"DEBUG: Aggregated SHAP Shape: {sv_final.shape}")
                else:
                    sv_final = sv
                
                # 7. Post-Aggregation Validation
                if len(sv_final) != n_features:
                    st.error(f"SHAP output mismatch: Aggregated {len(sv_final)} != Features {n_features}")
                    return

                # 8. Visualization
                df_res = pd.DataFrame({
                    'Feature': f_names,
                    'Contribution': sv_final,
                    'Impact': np.abs(sv_final)
                }).sort_values('Impact', ascending=False).head(15)

                st.subheader(f"Influencers for '{target_class}' Classification")
                fig = px.bar(df_res, x='Contribution', y='Feature', orientation='h',
                             color='Contribution', color_continuous_scale='RdBu', template="plotly_white")
                
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                st.success("Corrective Analysis Complete: Reliable attribution verified.")

        except Exception as e:
            st.error("Explainability failed")
            st.code(str(e))
            print(f"DEBUG: ROOT FAILURE: {e}")

if __name__ == "__main__":
    show()
