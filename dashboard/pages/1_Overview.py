import streamlit as st
import dashboard_utils
import plotly.express as px
import pandas as pd

from dashboard_utils import safe_page_header, apply_global_styles

# Initialization
safe_page_header("Dataset Analytics")
apply_global_styles()

def show():
    try:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", "148,517")
        with col2:
            st.metric("Input Features", "122")
        with col3:
            st.metric("Threat Categories", "5")

        data = {
            'Class': ['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
            'Count': [77054, 53385, 14077, 3882, 119]
        }
        df = pd.DataFrame(data)

        st.subheader("Global Traffic Composition")
        fig = px.pie(
            df, values='Count', names='Class', hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Alphabet
        )
        
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### 🔍 NSL-KDD Deep Dive
        - **Source**: Industry-standard NSL-KDD refinement.
        - **Preprocessing**: Robust Scaling + One-Hot Encoding (OHE).
        - **Imbalance Handling**: SMOTE over-sampling for R2L/U2R minority classes.
        """)
        
    except Exception as e:
        st.error(f"⚠️ Page Logic Failure: {e}")

if __name__ == "__main__":
    show()
