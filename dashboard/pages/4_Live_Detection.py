import streamlit as st
import dashboard_utils 
import torch
import time
import pandas as pd

from dashboard_utils import (
    safe_page_header, apply_global_styles, validate_resource
)

# ─── REBUILT IMPORTS ──────────────────────────────────────────────────────────
try:
    from preprocessing.pipeline import run_pipeline
    from models.hybrid_model import CNNBiLSTMHybrid
    from utils.config import CHECKPOINT_HYBRID, CLASS_NAMES_MULTICLASS
except ImportError as e:
    st.error(f"❌ Core Structural Synchronization Failure: {e}")
    st.stop()

# Initialization
safe_page_header("Real-Time Prediction Hub")
apply_global_styles()

@st.cache_resource
def load_security_environment():
    """Initializes the AI inference engine with zero-crash stability."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not validate_resource(CHECKPOINT_HYBRID):
        return None, None, None
        
    try:
        # Load Data
        data_bundle = run_pipeline(multiclass=True, apply_smote=False)
        
        # Initialize Hybrid Architecture
        model = CNNBiLSTMHybrid(input_dim=data_bundle['input_dim'], num_classes=data_bundle['num_classes'])
        model.load_state_dict(torch.load(CHECKPOINT_HYBRID, map_location=device, weights_only=True), strict=False)
        model.to(device)
        model.eval()
        
        return model, data_bundle, device
    except Exception as e:
        st.error(f"🚨 Inference Core Fault: {e}")
        return None, None, None

def show():
    model, data, device = load_security_environment()
    
    if model is None:
        st.warning("⚠️ **Detection Engine Offline**: Hybrid model weights not found.")
        st.info("System Fix: Train the core via `python main.py train --model hybrid`.")
        return

    st.sidebar.subheader("Stream Configuration")
    sim_rate = st.sidebar.slider("Sampling Interval (s)", 0.2, 5.0, 1.0)
    
    if 'active' not in st.session_state:
        st.session_state.active = False

    def toggle():
        st.session_state.active = not st.session_state.active

    btn_label = "🛑 Stop Stream" if st.session_state.active else "▶️ Start Live Feed"
    st.sidebar.button(btn_label, on_click=toggle)

    col1, col2, col3 = st.columns(3)
    p_seen   = col1.empty()
    p_threats = col2.empty()
    p_status = col3.empty()

    p_seen.metric("Packets Processed", "0")
    p_threats.metric("Threats Detected", "0")
    p_status.metric("IPS Status", "ARMED")

    st.subheader("Unified Threat Management (UTM) Adaptive Stream")
    display_proxy = st.empty()
    
    if 'threat_log' not in st.session_state:
        st.session_state.threat_log = []

    if st.session_state.active:
        loader = data['seq_loader_test']
        count = 0
        alarms = 0
        
        try:
            for x, y in loader:
                if not st.session_state.active: break
                x = x.to(device)
                with torch.no_grad():
                    outputs = model(x)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)

                for i in range(len(preds)):
                    if not st.session_state.active: break
                    count += 1
                    p_label = CLASS_NAMES_MULTICLASS[preds[i].item()]
                    t_label = CLASS_NAMES_MULTICLASS[y[i].item()]
                    score   = probs[i][preds[i]].item()
                    
                    threat_level = "🟢 CLEAN" if p_label == "Normal" else f"🔥 ALERT: {p_label}"
                    if p_label != "Normal": alarms += 1
                    
                    st.session_state.threat_log.insert(0, {
                        "Time": time.strftime("%H:%M:%S"),
                        "Diagnosis": threat_level,
                        "Confidence": f"{score:.2%}",
                        "Ground Truth": t_label
                    })
                    st.session_state.threat_log = st.session_state.threat_log[:20]
                    
                    p_seen.metric("Packets Processed", str(count))
                    p_threats.metric("Threats Detected", str(alarms))
                    
                    if p_label != "Normal" and score > 0.8:
                        p_status.error("MITIGATION ACTIVE")
                    else:
                        p_status.success("ARMED")
                        
                    display_proxy.table(pd.DataFrame(st.session_state.threat_log))
                    time.sleep(sim_rate)
        except Exception as e:
            st.error(f"Stream Connectivity Error: {e}")
            st.session_state.active = False
    else:
        if st.session_state.threat_log:
            display_proxy.table(pd.DataFrame(st.session_state.threat_log))
        st.info("System Ready. Awaiting live data injection...")

if __name__ == "__main__":
    show()
