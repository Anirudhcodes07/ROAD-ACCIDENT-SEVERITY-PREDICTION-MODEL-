"""
train_model.py
──────────────────────────────────────────────────────────────────────────────
Road Accident Severity Prediction
IML AAT Miniproject | BMS College of Engineering | 24AM4PCIML
Faculty In-charge: Prof. Chethana V
──────────────────────────────────────────────────────────────────────────────

ML ALGORITHMS USED
==================

1. RANDOM FOREST CLASSIFIER
   ─────────────────────────
   Type        : Ensemble Learning (Bagging)
   Base Learner: Decision Tree (CART — Classification & Regression Tree)

   How it works:
   • Builds `n_estimators` decision trees on random bootstrap samples of the
     training data (sampling with replacement — "bagging").
   • At each split, only a random subset of √p features is considered
     (where p = number of features). This de-correlates the trees.
   • Final prediction = majority vote across all trees (classification).

   Key hyperparameters (used here):
     n_estimators = 200   — number of trees; more trees → lower variance
     max_depth    = 12    — maximum depth per tree; controls overfitting
     min_samples_split= 5 — minimum samples needed to split a node
     class_weight = 'balanced' — adjusts weights inversely proportional to
                                 class frequencies (handles imbalance)
     random_state = 42

   Why suitable here:
   • Handles mixed feature types (numeric + categorical integers).
   • Naturally provides feature importance rankings.
   • Robust to outliers and missing values.
   • Consistently strong on tabular accident data (see IEEE Xplore references).

   Mathematical basis:
     Gini Impurity at node t:  G(t) = 1 − Σ p(k|t)²
     Information Gain: IG = G(parent) − Σ (|child|/|parent|)·G(child)
     Ensemble prediction: ŷ = mode{ h₁(x), h₂(x), …, hB(x) }


2. SUPPORT VECTOR MACHINE (SVM)
   ──────────────────────────────
   Type   : Kernel-based discriminative classifier
   Kernel : Radial Basis Function (RBF) — k(xᵢ, xⱼ) = exp(−γ‖xᵢ−xⱼ‖²)

   How it works:
   • Finds the hyperplane that maximises the margin between classes in a
     high-dimensional (possibly infinite) feature space induced by the kernel.
   • Soft-margin SVM allows misclassifications controlled by penalty C.
   • Multi-class: One-vs-One (OvO) strategy — trains C(C−1)/2 binary SVMs
     (here C=3 classes → 3 binary classifiers), final label by voting.

   Key hyperparameters (used here):
     kernel      = 'rbf'   — captures non-linear decision boundaries
     C           = 10      — regularisation (higher C = less margin, more fit)
     gamma       = 'scale' — 1 / (n_features × X.var()); auto-tuned
     probability = True    — enables Platt scaling for probability estimates
     random_state= 42

   Pre-processing required:
   • StandardScaler applied: z = (x − μ) / σ
     SVM is sensitive to feature scale; scaling is mandatory.

   Why suitable here:
   • Works well in moderate-dimensional spaces (~10 features).
   • RBF kernel models complex interactions (e.g., speed × alcohol).
   • Strong theoretical generalisation bounds (VC dimension theory).

   Mathematical basis:
     Primal:  min ½‖w‖² + C Σ ξᵢ    s.t. yᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ
     Dual:    max Σαᵢ − ½ ΣΣ αᵢαⱼyᵢyⱼk(xᵢ,xⱼ)
     Decision: f(x) = sign(Σ αᵢyᵢk(xᵢ,x) + b)


3. MULTILAYER PERCEPTRON (MLP / Basic Neural Network)
   ────────────────────────────────────────────────────
   Type       : Feedforward Artificial Neural Network
   Architecture: Input(9) → Dense(128, ReLU) → Dense(64, ReLU) →
                 Dense(32, ReLU) → Output(3, Softmax)

   How it works:
   • Organised in layers: input, hidden, output.
   • Each neuron computes: z = Σ (wᵢxᵢ) + b, then applies activation f(z).
   • ReLU activation: f(z) = max(0, z) — avoids vanishing gradient.
   • Softmax output: P(class k) = exp(zₖ) / Σ exp(zⱼ) — gives probabilities.
   • Trained by Backpropagation + Adam optimiser:
       - Forward pass → compute loss (cross-entropy)
       - Backward pass → compute ∂L/∂w via chain rule
       - Adam updates: combines momentum + adaptive learning rate

   Key hyperparameters (used here):
     hidden_layer_sizes = (128, 64, 32) — three hidden layers
     activation         = 'relu'
     solver             = 'adam'
     alpha              = 0.001        — L2 regularisation
     max_iter           = 500
     early_stopping     = True         — stops if val-loss doesn't improve
     random_state       = 42

   Pre-processing required:
   • StandardScaler (same as SVM) — neural nets require normalised inputs.

   Why suitable here:
   • Can learn complex non-linear feature interactions automatically.
   • Serves as a comparison point between classical ML and neural approaches.

──────────────────────────────────────────────────────────────────────────────
EVALUATION METRICS
  Accuracy  = (TP + TN) / N
  Precision = TP / (TP + FP)  [per class, macro-averaged]
  Recall    = TP / (TP + FN)  [per class, macro-averaged]
  F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)
  Confusion Matrix — rows=actual, cols=predicted
──────────────────────────────────────────────────────────────────────────────
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble      import RandomForestClassifier
from sklearn.svm           import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import (accuracy_score, f1_score, precision_score,
                                    recall_score, classification_report,
                                    confusion_matrix)
import joblib

from dataset import generate, FEATURE_COLS, SEVERITY_LABELS

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
MDL_DIR = os.path.join(BASE, "model")
STT_DIR = os.path.join(BASE, "static")
os.makedirs(MDL_DIR, exist_ok=True)
os.makedirs(STT_DIR, exist_ok=True)

CLASS_NAMES = ["Slight", "Serious", "Fatal"]
PALETTE     = {"Slight": "#2563eb", "Serious": "#d97706", "Fatal": "#dc2626"}

# ── 1. Generate Dataset ────────────────────────────────────────────────────────
print("=" * 60)
print("ROAD ACCIDENT SEVERITY PREDICTION — MODEL TRAINING")
print("=" * 60)
print("\n[1] Generating dataset (STATS19 schema, N=3000) …")
df = generate()

counts = df["Accident_Severity"].value_counts().sort_index()
print("    Class distribution:")
for k, v in counts.items():
    print(f"      {SEVERITY_LABELS[k]:8s}: {v:5d}  ({v/len(df)*100:.1f} %)")

# Save CSV for reference
df.to_csv(os.path.join(BASE, "road_accident_dataset.csv"), index=False)
print("    Saved: road_accident_dataset.csv")

# ── 2. Pre-processing ──────────────────────────────────────────────────────────
print("\n[2] Pre-processing …")
X = df[FEATURE_COLS].values
y = df["Accident_Severity"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"    Train : {len(X_train)} samples  |  Test : {len(X_test)} samples")

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)   # fit on train only
X_test_sc   = scaler.transform(X_test)         # transform test with same params
print("    StandardScaler fitted (μ and σ computed on training set only)")

# ── 3. Train Models ────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── 3a. Random Forest ─────────────────────────────────────────────────────────
print("\n[3a] Training Random Forest …")
print("     Algorithm : Ensemble Bagging — 200 CART decision trees")
print("     Criterion : Gini Impurity | max_depth=12 | class_weight=balanced")

rf = RandomForestClassifier(
    n_estimators   = 200,
    max_depth       = 12,
    min_samples_split = 5,
    class_weight    = "balanced",
    random_state    = 42,
    n_jobs          = -1
)
rf.fit(X_train, y_train)
rf_pred   = rf.predict(X_test)
rf_cv     = cross_val_score(rf, X, y, cv=cv, scoring="accuracy").mean()
rf_acc    = accuracy_score(y_test, rf_pred)
rf_f1     = f1_score(y_test, rf_pred, average="weighted")
rf_prec   = precision_score(y_test, rf_pred, average="weighted", zero_division=0)
rf_rec    = recall_score(y_test, rf_pred, average="weighted", zero_division=0)
print(f"     Test Accuracy : {rf_acc:.4f}  |  CV Accuracy : {rf_cv:.4f}")
print(f"     Precision     : {rf_prec:.4f} |  Recall      : {rf_rec:.4f}")
print(f"     Weighted F1   : {rf_f1:.4f}")

# ── 3b. SVM ───────────────────────────────────────────────────────────────────
print("\n[3b] Training SVM (RBF kernel) …")
print("     Algorithm : C-SVM | kernel=RBF | C=10 | gamma=scale (auto)")
print("     Multi-class: One-vs-One (3 binary classifiers)")
print("     Input      : StandardScaler-normalised features")

svm = SVC(
    kernel      = "rbf",
    C           = 10,
    gamma       = "scale",
    probability = True,
    class_weight= "balanced",
    random_state= 42
)
svm.fit(X_train_sc, y_train)
svm_pred  = svm.predict(X_test_sc)
svm_cv    = cross_val_score(svm, X_train_sc, y_train, cv=cv, scoring="accuracy").mean()
svm_acc   = accuracy_score(y_test, svm_pred)
svm_f1    = f1_score(y_test, svm_pred, average="weighted")
svm_prec  = precision_score(y_test, svm_pred, average="weighted", zero_division=0)
svm_rec   = recall_score(y_test, svm_pred, average="weighted", zero_division=0)
print(f"     Test Accuracy : {svm_acc:.4f}  |  CV Accuracy : {svm_cv:.4f}")
print(f"     Precision     : {svm_prec:.4f} |  Recall      : {svm_rec:.4f}")
print(f"     Weighted F1   : {svm_f1:.4f}")

# ── 3c. Neural Network ────────────────────────────────────────────────────────
print("\n[3c] Training MLP Neural Network …")
print("     Architecture : Input(9)→Dense(128,ReLU)→Dense(64,ReLU)")
print("                    →Dense(32,ReLU)→Output(3,Softmax)")
print("     Optimiser    : Adam | L2 alpha=0.001 | Early stopping ON")

nn = MLPClassifier(
    hidden_layer_sizes = (128, 64, 32),
    activation         = "relu",
    solver             = "adam",
    alpha              = 0.001,
    max_iter           = 500,
    early_stopping     = True,
    validation_fraction= 0.10,
    random_state       = 42
)
nn.fit(X_train_sc, y_train)
nn_pred   = nn.predict(X_test_sc)
nn_cv     = cross_val_score(nn, X_train_sc, y_train, cv=cv, scoring="accuracy").mean()
nn_acc    = accuracy_score(y_test, nn_pred)
nn_f1     = f1_score(y_test, nn_pred, average="weighted")
nn_prec   = precision_score(y_test, nn_pred, average="weighted", zero_division=0)
nn_rec    = recall_score(y_test, nn_pred, average="weighted", zero_division=0)
print(f"     Test Accuracy : {nn_acc:.4f}  |  CV Accuracy : {nn_cv:.4f}")
print(f"     Precision     : {nn_prec:.4f} |  Recall      : {nn_rec:.4f}")
print(f"     Weighted F1   : {nn_f1:.4f}")

# ── 4. Save Artefacts ──────────────────────────────────────────────────────────
print("\n[4] Saving models …")
joblib.dump(rf,     os.path.join(MDL_DIR, "rf_model.pkl"))
joblib.dump(svm,    os.path.join(MDL_DIR, "svm_model.pkl"))
joblib.dump(nn,     os.path.join(MDL_DIR, "nn_model.pkl"))
joblib.dump(scaler, os.path.join(MDL_DIR, "scaler.pkl"))

metrics = {
    "Random Forest": {
        "accuracy": round(rf_acc, 4),  "cv_accuracy": round(rf_cv, 4),
        "f1": round(rf_f1, 4), "precision": round(rf_prec, 4), "recall": round(rf_rec, 4)
    },
    "SVM (RBF)": {
        "accuracy": round(svm_acc, 4), "cv_accuracy": round(svm_cv, 4),
        "f1": round(svm_f1, 4), "precision": round(svm_prec, 4), "recall": round(svm_rec, 4)
    },
    "Neural Network": {
        "accuracy": round(nn_acc, 4),  "cv_accuracy": round(nn_cv, 4),
        "f1": round(nn_f1, 4), "precision": round(nn_prec, 4), "recall": round(nn_rec, 4)
    },
}
with open(os.path.join(MDL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Classification reports
reports = {
    "Random Forest":  classification_report(y_test, rf_pred,  target_names=CLASS_NAMES),
    "SVM (RBF)":      classification_report(y_test, svm_pred, target_names=CLASS_NAMES),
    "Neural Network": classification_report(y_test, nn_pred,  target_names=CLASS_NAMES),
}
with open(os.path.join(MDL_DIR, "classification_reports.json"), "w") as f:
    json.dump(reports, f, indent=2)

print("    ✅ Models, scaler, metrics saved to /model/")

# ── 5. Plots ───────────────────────────────────────────────────────────────────
print("\n[5] Generating plots …")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})

# 5a. Model Comparison (Accuracy + F1)
fig, ax = plt.subplots(figsize=(8, 4.5))
model_names = list(metrics.keys())
accs  = [v["accuracy"] * 100 for v in metrics.values()]
f1s   = [v["f1"] * 100        for v in metrics.values()]
cvs   = [v["cv_accuracy"]*100  for v in metrics.values()]
x = np.arange(len(model_names)); w = 0.26
b1 = ax.bar(x - w, accs, w, label="Test Accuracy %", color="#2563eb", alpha=0.88)
b2 = ax.bar(x,     f1s,  w, label="Weighted F1 %",   color="#16a34a", alpha=0.88)
b3 = ax.bar(x + w, cvs,  w, label="5-Fold CV Acc %", color="#9333ea", alpha=0.88)
ax.set_ylim(60, 100); ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel("Score (%)"); ax.set_title("Model Comparison — Test Accuracy, F1 & CV Accuracy", fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
for b in list(b1)+list(b2)+list(b3):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
            f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
plt.tight_layout()
plt.savefig(os.path.join(STT_DIR, "model_comparison.png"), dpi=130)
plt.close()

# 5b. Confusion Matrices (all 3 side-by-side)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (pred, title) in zip(axes, [
        (rf_pred,  "Random Forest"),
        (svm_pred, "SVM (RBF)"),
        (nn_pred,  "Neural Network")]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                linewidths=.5, linecolor="white")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix\n{title}", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(STT_DIR, "confusion_matrix.png"), dpi=130)
plt.close()

# 5c. Feature Importance (RF)
fi  = rf.feature_importances_
idx = np.argsort(fi)
colors_fi = ["#2563eb" if fi[i] >= np.median(fi) else "#93c5fd" for i in idx]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh([FEATURE_COLS[i] for i in idx], fi[idx] * 100, color=colors_fi, edgecolor="white")
ax.set_xlabel("Importance (%)"); ax.set_title("Feature Importance — Random Forest", fontsize=11, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(STT_DIR, "feature_importance.png"), dpi=130)
plt.close()

# 5d. Class Distribution
fig, ax = plt.subplots(figsize=(5, 4))
cnt = df["Accident_Severity"].value_counts().sort_index()
bars = ax.bar(CLASS_NAMES, cnt.values,
              color=["#2563eb","#d97706","#dc2626"], alpha=0.85, edgecolor="white")
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+20,
            f"{b.get_height()}\n({b.get_height()/cnt.sum()*100:.1f}%)",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Count"); ax.set_title("Accident Severity Distribution (N=3000)", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(STT_DIR, "severity_distribution.png"), dpi=130)
plt.close()

# 5e. Precision / Recall per class (RF)
report_dict = {}
for i, name in enumerate(CLASS_NAMES):
    mask = y_test == i
    report_dict[name] = {
        "Precision": precision_score(y_test==i, rf_pred==i, zero_division=0),
        "Recall":    recall_score(y_test==i, rf_pred==i, zero_division=0),
        "F1":        f1_score(y_test==i, rf_pred==i, zero_division=0),
    }
metrics_per_class = np.array([[v["Precision"], v["Recall"], v["F1"]]
                               for v in report_dict.values()])
fig, ax = plt.subplots(figsize=(7, 4))
x2 = np.arange(len(CLASS_NAMES)); w2 = 0.25
ax.bar(x2-w2, metrics_per_class[:,0]*100, w2, label="Precision", color="#2563eb", alpha=0.85)
ax.bar(x2,    metrics_per_class[:,1]*100, w2, label="Recall",    color="#16a34a", alpha=0.85)
ax.bar(x2+w2, metrics_per_class[:,2]*100, w2, label="F1 Score",  color="#d97706", alpha=0.85)
ax.set_xticks(x2); ax.set_xticklabels(CLASS_NAMES)
ax.set_ylabel("Score (%)"); ax.set_ylim(0,110)
ax.set_title("Precision / Recall / F1 per Class — Random Forest", fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(STT_DIR, "per_class_metrics.png"), dpi=130)
plt.close()

print("    ✅ 5 plots saved to /static/")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("  Run:  python app.py")
print("  Open: http://127.0.0.1:5000")
print("="*60)
