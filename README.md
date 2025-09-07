# 🌱 Smart Crop Advisory System - Machine Learning Service

The **ML Service** provides intelligent crop recommendations based on soil, climate, and environmental factors. It is powered by **Flask** and integrates with the backend.

---

## ✨ Features
- ML model predicts suitable crops based on soil and weather features
- Flask REST API for predictions
- Preprocessing utilities for clean inputs
- Exported model (`models/crop_recommender.pkl`)

---

## 🛠 Tech Stack
- **Python 3.9+**
- **Flask**
- **Scikit-learn, Pandas, NumPy**
- **Pickle** for model persistence

---

## 📂 Folder Structure

```text
smart-crop-ml/
├── models/           # Trained ML model(s)
├── services/         # Recommender service(s)
├── utils/            # Preprocessing functions
├── app.py            # Flask entry point
├── train.py          # Training script
├── requirements.txt  # Dependencies
└── .env.example      # Env vars template
```

---

## 🚀 Getting Started

### 1️⃣ Prerequisites
- Python 3.9+
- pip / venv

### 2️⃣ Installation
```bash
cd smart-crop-ml
pip install -r requirements.txt
```

### 3️⃣ Run ML Service
```bash
python app.py
```

Service will run on 👉 `http://localhost:5001` (or the port you configure)

---

## 🔄 Training the Model
```bash
python train.py
```
This script should preprocess data, train a classifier/regressor, and save the model to `models/` (e.g. `models/crop_recommender.pkl`).

---

## 🤝 Contribution
1. Fork the repo  
2. Create a new branch (`feature-xyz`)  
3. Commit changes  
4. Push branch  
5. Create a Pull Request  

---

## 📜 License
This project is licensed under the **MIT License**.
