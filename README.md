# 🚨 AccidentShield AI — Accident Detection from CCTV

## Quick Start

### Step 1 — Download Dataset
- Go to: https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage
- Download your `kaggle.json` from Kaggle → Account → API
- Place `kaggle.json` in this project folder

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run Notebook
Open `Accident_Detection_CNN.ipynb` in VS Code → Run All Cells
This will:
- Download the dataset automatically
- Train CNN from Scratch
- Train Transfer Learning (MobileNetV2)
- Fine tune the model
- Save best model to `models/`

### Step 4 — Launch App
```bash
streamlit run app.py
```

## Files
- `Accident_Detection_CNN.ipynb` — Full training pipeline
- `app.py` — Streamlit web app (image + video detection)
- `requirements.txt` — Python dependencies
- `models/` — Saved models (created after running notebook)
