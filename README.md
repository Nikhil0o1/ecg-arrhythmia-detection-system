<div align="center">

# 🫀 CardioAI

### AI-Powered ECG Arrhythmia Detection System

Production-grade deep learning system for detecting **cardiac arrhythmias from 12-lead ECG signals** using clinical datasets and modern AI infrastructure.

Built with **PyTorch, FastAPI, and a clinical-grade React dashboard.**

---

### ⚡ Live Demo

Frontend → https://ecg-frontend-cfkf.onrender.com
API Docs → https://ecg-arrhythmia-detection-system-backend.onrender.com/docs

---

</div>

---

# 🎬 Product Demo

<p align="center">

<img src="assets/Animation.gif" width="900"/>

</p>

The platform allows clinicians or developers to upload **12-lead ECG signals** and instantly receive predictions powered by a deep learning model trained on **PTB-XL clinical ECG data**.

---

# 🖥 Clinical Dashboard

The interface is designed to simulate a **modern clinical diagnostic tool**.

### Key UI Features

• Drag-and-drop ECG upload
• Real-time AI predictions
• Confidence visualization
• Clean medical-style interface

---

# ⚙️ Tech Stack

<p align="center">

<img src="https://skillicons.dev/icons?i=python,pytorch,fastapi,react,typescript,tailwind,git,docker"/>

</p>

| Layer         | Technology                       |
| ------------- | -------------------------------- |
| Frontend      | React + TypeScript + TailwindCSS |
| Backend       | FastAPI                          |
| Deep Learning | PyTorch                          |
| Dataset       | PTB-XL                           |
| Model         | IndustryCNN                      |
| Deployment    | Vercel + Render                  |

---

# 🧠 Deep Learning Model

The project uses a custom **IndustryCNN architecture optimized for ECG signal analysis**.

### Architecture Highlights

• 12-lead ECG processing
• Residual CNN blocks
• Temporal feature extraction
• Global average pooling
• Binary classification head

Total parameters:

```
~1.2 Million
```

---

# 📊 Model Performance

Evaluation performed on the **PTB-XL test dataset**.

| Metric      | Score     |
| ----------- | --------- |
| Accuracy    | **88.8%** |
| Precision   | **93.8%** |
| Recall      | **84.9%** |
| F1 Score    | **89.1%** |
| ROC-AUC     | **95.7%** |
| Sensitivity | **84.9%** |
| Specificity | **93.4%** |

The model successfully captures temporal patterns across **multi-lead ECG signals**, enabling reliable arrhythmia detection.

---

# 🏗 System Architecture

```
ECG Signal (.npy)
        │
        ▼
Signal Preprocessing
        │
        ▼
IndustryCNN Model (PyTorch)
        │
        ▼
FastAPI Inference API
        │
        ▼
React Clinical Dashboard
```

---

# 📂 Project Structure

```
ecg-arrhythmia-detection-system
│
├── backend
│   ├── api
│   ├── inference
│   ├── training
│   ├── models
│   └── run_training.py
│
├── frontend
│   ├── components
│   ├── hooks
│   ├── pages
│   └── styles
│
├── assets
│   └── animations.gif
│
├── data
└── README.md
```

---

# 🔬 Dataset

The model was trained using the **PTB-XL dataset**, one of the largest publicly available ECG datasets.

Dataset characteristics:

```
21,000 ECG recordings
12-lead signals
10 second recordings
100 Hz sampling rate
```

Source:

PhysioNet PTB-XL ECG Dataset

---

# ⚡ Quick Start

Clone the repository.

```
git clone https://github.com/Nikhil001/ecg-arrhythmia-detection-system
cd ecg-arrhythmia-detection-system
```

Install backend dependencies.

```
pip install -r backend/requirements.txt
```

Run the backend server.

```
uvicorn api.main:app --reload
```

Run the frontend.

```
cd frontend
npm install
npm run dev
```

---

# 🚀 API Endpoints

### Health Check

```
GET /health
```

---

### ECG Prediction

```
POST /predict
```

---

### Upload ECG File

```
POST /predict-file
```

---

### Simulation Endpoint

```
POST /simulate
```

---

# 🧪 Example Prediction

Input ECG signal → 12-lead waveform

Output:

```json
{
 "probability": 0.996539,
 "prediction": 1,
 "confidence": 0.996539
}
```

---

# 🚧 Future Improvements

• Multi-class arrhythmia classification
• Transformer-based ECG models
• Real-time ECG streaming
• ECG waveform visualization

---

# 👨‍💻 Author

**Nikhil M**

AI Engineer | Deep Learning | Systems

GitHub
https://github.com/Nikhil001

---

# ⭐ Support

If you find this project interesting, consider giving it a **star ⭐**.

It helps the project grow.
