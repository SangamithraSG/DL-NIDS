import streamlit as st
import dashboard_utils
import plotly.graph_objects as go
import pandas as pd

from dashboard_utils import safe_page_header, apply_global_styles

# Initialization
safe_page_header("Performance Benchmarks")
apply_global_styles()

def show():
    # Production Performance Matrix
    core_metrics = {
        'Engine': ['Autoencoder', 'BiLSTM', 'Hybrid', 'Ensemble'],
        'Accuracy': [0.81, 0.78, 0.75, 0.76],
        'FPR': [0.071, 0.032, 0.039, 0.031],
        'FNR': [0.282, 0.347, 0.368, 0.366]
    }
    df = pd.DataFrame(core_metrics)

    try:
        st.subheader("Cross-Architecture Performance Profile")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['Engine'], y=df['Accuracy'], name='Accuracy', marker_color='#1f77b4'))
        fig.add_trace(go.Bar(x=df['Engine'], y=df['FPR'], name='False Positive Rate (FPR)', marker_color='#d62728'))
        fig.add_trace(go.Bar(x=df['Engine'], y=df['FNR'], name='False Negative Rate (FNR)', marker_color='#9467bd'))

        fig.update_layout(
            barmode='group', 
            template="plotly_white", 
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tactical Scoring Matrix")
        st.table(df.set_index('Engine').style.format("{:.2%}"))

        st.info("""
        **💡 Strategic Insights:**
        - **Ensemble Core**: Delivers the prioritized "Low Alarm" profile with a 3.17% FPR.
        - **Autoencoder Core**: Exceptional at baseline pattern recognition but prone to noise in binary classification.
        - **Hybrid Core**: Optimized for complex attack sequences combining spatial and temporal features.
        """)
        
    except Exception as e:
        st.error(f"Visualization Logic Failure: {e}")

if __name__ == "__main__":
    show()
