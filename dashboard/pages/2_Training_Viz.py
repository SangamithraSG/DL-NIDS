import streamlit as st
import dashboard_utils
import plotly.express as px
import pandas as pd

from dashboard_utils import safe_page_header, apply_global_styles, validate_resource
from utils.config import LOGS_DIR

# Initialization
safe_page_header("Training Diagnostics")
apply_global_styles()

@st.cache_data
def load_historical_data(file_name):
    """Safe CSV loader with caching."""
    path = LOGS_DIR / file_name
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.error(f"IO Failure ({file_name}): {e}")
    return None

def show():
    model_logs = {
        "Autoencoder": "autoencoder_history.csv",
        "BiLSTM": "lstm_history.csv",
        "CNN": "cnn_history.csv",
        "Hybrid": "hybrid_history.csv"
    }

    selection = st.selectbox("Select Target Engine", list(model_logs.keys()))
    logfile = model_logs[selection]
    
    if not validate_resource(LOGS_DIR / logfile):
        st.warning(f"⚠️ **Logs Missing**: No telemetry found for {selection}.")
        st.info(f"Resolution: Run `python main.py train --model {selection.lower()}`")
        return

    df = load_historical_data(logfile)
    
    if df is not None:
        st.subheader(f"{selection} Convergence History")
        
        # Loss Visualization
        fig = px.line(
            df, x='epoch', y=['train_loss', 'val_loss'], 
            title="Loss Reduction Curve",
            color_discrete_map={"train_loss": "#1f77b4", "val_loss": "#ff7f0e"}
        )
        
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Epoch-by-Epoch Telemetry")
        st.dataframe(df.style.highlight_min(axis=0, subset=['val_loss'], color='#00f2fe'), use_container_width=True)
    else:
        st.error("Historical data parsing failed.")

if __name__ == "__main__":
    show()
