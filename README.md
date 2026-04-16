DL-NIDS: Deep Learning Network Intrusion Detection System

🛡️ Project Overview
DL-NIDS (Deep Learning Network Intrusion Detection System) is a high-performance security framework designed to identify and mitigate cyber threats in real-time. By leveraging state-of-the-art Deep Learning architectures, the system analyzes network traffic patterns to distinguish between benign activity and sophisticated attacks (DoS, Probe, U2R, R2L).

In modern cybersecurity, traditional rule-based systems often fail against zero-day exploits. This project provides an AI-driven alternative that learns from the industry-standard NSL-KDD dataset, offering a robust defense mechanism for enterprise network environments.

📂 Project Structure
text
dl_nids/
├── dashboard/          # Streamlit UI & forensic pages
├── data/               # NSL-KDD dataset (Manually added)
├── models/             # Architecture definitions (CNN, BiLSTM, etc.)
├── preprocessing/      # Data cleansing, scaling, and SMOTE logic
├── training/           # Unified training and validation engine
├── utils/              # Configuration, logging, and seed management
├── main.py             # CLI Entrypoint for the entire system
└── requirements.txt    # Project dependencies
🚀 Setup Instructions
1. Clone the Repository
bash
git clone <repo_url>
cd dl_nids
2. Environment Configuration
Choose one of the following methods to create a clean environment:

Option A: Using Conda

bash
conda create -n dlnids python=3.10
conda activate dlnids
Option B: Using venv (Python Native)

bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
3. Install Dependencies
bash
pip install -r requirements.txt
📊 Dataset Setup (Critical)
This system requires the NSL-KDD dataset. Due to licensing and size, you must download it manually.

Download the dataset from official sources (e.g., Kaggle or UNB).
Locate the following files:
KDDTrain+.txt
KDDTest+.txt
Place these files in the dl_nids/data/ directory:
bash
dl_nids/data/KDDTrain+.txt
dl_nids/data/KDDTest+.txt
🛠️ How to Run
Step 1: Preprocessing
Generate the scaled and encoded artifacts needed for training.

bash
python main.py preprocess
Step 2: Model Training
Train a specific architecture (e.g., the Hybrid CNN-BiLSTM).

bash
python main.py train --model hybrid
Available options: autoencoder, cnn, lstm, hybrid, rf, all.

Step 3: Launch Dashboard
Start the interactive UI to visualize analytics and live detection.

bash
streamlit run dashboard/app.py
🔍 Explainability Note
IMPORTANT

Forensic explainability is limited to the Autoencoder model. Deep models like CNN and BiLSTM utilize Kernel-SHAP for attribution, which is computationally expensive and causes significant latency. For real-time stability, the system uses Autoencoder Reconstruction Error to highlight divergent features during attack scenarios.

🧪 Key Features
Multi-Model Engine: Support for CNN, BiLSTM, Hybrid (CNN+BiLSTM), and Random Forest.
Ensemble Voting: High-confidence detection using probabilistic voting.
Interactive Dashboard: Real-time traffic simulation with threat level gauges.
AI Forensics: Feature-level attribution to understand why an alert was triggered.
Imbalance Handling: Integrated SMOTE to ensure minority attacks are detected.
🔧 Troubleshooting
"No module named X" Ensure your environment is active and run: pip install -r requirements.txt

Streamlit using the wrong Python version If you see errors related to imports while running Streamlit, use: python -m streamlit run dashboard/app.py

Model not found (Checkpoint Error) You must train the model before visualizing it in the dashboard: python main.py train --model <selected_model>

Shape Mismatch during evaluation Ensure you have run python main.py preprocess with the same configuration (binary vs multiclass) as your training cycle.