# 🛡️ DL-NIDS: Deep Learning Network Intrusion Detection System

## 📌 Project Overview

DL-NIDS is a deep learning-based cybersecurity system designed to detect network intrusions in real-time. It analyzes network traffic patterns to classify activity as normal or malicious (DoS, Probe, U2R, R2L).

Traditional rule-based systems struggle with modern attacks. This system uses AI models trained on the NSL-KDD dataset to provide intelligent threat detection.

---

## 📂 Project Structure

```
dl_nids/
├── dashboard/        # Streamlit UI
├── data/             # Dataset (manual download)
├── models/           # Model architectures
├── preprocessing/    # Data processing
├── training/         # Training logic
├── utils/            # Config & helpers
├── main.py           # Entry point
└── requirements.txt
```

---

## 🚀 Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/SangamithraSG/DL-NIDS.git
cd DL-NIDS
```

---

### 2. Create Environment

#### Option A: Conda

```bash
conda create -n dlnids python=3.10
conda activate dlnids
```

#### Option B: venv

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Setup (IMPORTANT)

Download NSL-KDD dataset manually.

Required files:

* `KDDTrain+.txt`
* `KDDTest+.txt`

Place them here:

```
dl_nids/data/
```

---

## 🛠️ How to Run

### Step 1: Preprocessing

```bash
python main.py preprocess
```

---

### Step 2: Train Model

```bash
python main.py train --model hybrid
```

Available models:

```
autoencoder, cnn, lstm, hybrid, rf, all
```

---

### Step 3: Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🔍 Explainability Note

Explainability is supported **only for the Autoencoder model**.

Deep models (CNN, LSTM) require SHAP, which is computationally expensive and unstable for real-time dashboards.

Autoencoder uses **reconstruction error** for fast and interpretable insights.

---

## 🧪 Features

* Multi-model support (CNN, LSTM, Hybrid, RF)
* Ensemble detection
* Real-time Streamlit dashboard
* Autoencoder-based explainability
* SMOTE for class imbalance

---

## 🔧 Troubleshooting

### ❌ Module not found

```bash
pip install -r requirements.txt
```

---

### ❌ Streamlit using wrong Python

```bash
python -m streamlit run dashboard/app.py
```

---

### ❌ Model not found

Train model first:

```bash
python main.py train --model hybrid
```

---

## 📌 Note

Explainability for deep models was intentionally disabled to ensure system stability and performance.
