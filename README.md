# 🚗 Road Accident Severity Prediction


---

## Project Structure

```
road_accident_project/
├── dataset.py               ← Dataset schema + generation (STATS19-based)
├── train_model.py           ← Full ML training pipeline (RF + SVM + MLP)
├── app.py                   ← Flask backend (routes + predict API)
├── requirements.txt         ← pip dependencies
├── road_accident_dataset.csv← Auto-generated after running train_model.py
├── model/                   ← Saved models (auto-created)
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   ├── nn_model.pkl
│   ├── scaler.pkl
│   ├── metrics.json
│   └── classification_reports.json
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── *.png                ← 5 plots (auto-generated)
└── templates/
    └── index.html
```

---

## ⚙️ Setup & Run in VS Code

### Step 1 — Create & activate virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Train models (run ONCE)
```bash
python train_model.py
```
Prints accuracy, F1, precision, recall for all 3 models.
Saves models to `/model/` and 5 plots to `/static/`.

### Step 4 — Start the web app
```bash
python app.py
```
Open: **http://127.0.0.1:5000**

---

## Dataset
- **Schema:** UK STATS19 Road Safety Data (DfT open licence)
- **Size:** 3,000 records × 9 features + 1 target
- **Split:** 80% train / 20% test (stratified)
- **Classes:** Slight (≈80%) | Serious (≈15%) | Fatal (≈5%)

## ML Models
| Model | Type | Scaling |
|-------|------|---------|
| Random Forest | Ensemble Bagging, 200 CART trees | Not required |
| SVM (RBF) | Kernel SVM, One-vs-One | StandardScaler |
| Neural Network | MLP 128→64→32, Adam, Early stopping | StandardScaler |

## Web App Tabs
- **Predict** — Interactive prediction form with probability bars
- **Dataset** — Full feature descriptions and pre-processing pipeline
- **Algorithms** — Detailed explanation with maths for all 3 models
- **Results** — Metrics table + 5 visualisation plots

---
*Academic Year 2025–26 | Session: Feb2026 – June 2026*
