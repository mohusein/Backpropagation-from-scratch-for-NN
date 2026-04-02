import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.rcParams.update({
    'figure.facecolor': '#080c14',
    'axes.facecolor':   '#0d1117',
    'axes.edgecolor':   '#1e2d5e',
    'axes.labelcolor':  '#94a3b8',
    'text.color':       '#e2e8f0',
    'xtick.color':      '#64748b',
    'ytick.color':      '#64748b',
    'grid.color':       '#111827',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
})

print("All libraries loaded successfully.")
print(f"NumPy  : {np.__version__}")
print(f"Pandas : {pd.__version__}")


# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Full dataset shape : {X.shape}")
print(f"Class distribution : {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"Feature names (first 6): {list(data.feature_names[:6])}")

# Train / test split (stratified)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standard scaling (fit on train only)
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test       = scaler.transform(X_test)

# Validation split from train (used inside BO objective)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
)

print(f"\nTrain      : {X_train.shape[0]} samples")
print(f"Validation : {X_val.shape[0]}  samples")
print(f"Test       : {X_test.shape[0]}  samples")
print(f"Features   : {X_train.shape[1]}")

# Quick EDA
df = pd.DataFrame(X_train_full, columns=data.feature_names)
df['target'] = y_train_full

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Class balance
counts = pd.Series(y_train_full).value_counts()
colors = ['#ef4444', '#3b82f6']
axes[0].bar(['Malignant', 'Benign'], counts.values, color=colors, edgecolor='#1e2d5e', linewidth=0.8)
axes[0].set_title('Class Distribution (Train)', fontsize=12, color='#e2e8f0', pad=10)
axes[0].set_ylabel('Count', fontsize=10)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 2, str(v), ha='center', va='bottom', color='#e2e8f0', fontsize=11)

# Feature correlation heatmap (first 10 features)
corr = df.iloc[:, :10].corr()
sns.heatmap(corr, ax=axes[1], cmap='Blues', linewidths=0.3,
            linecolor='#0d1117', annot=False, cbar_kws={'shrink': 0.8})
axes[1].set_title('Feature Correlation (first 10)', fontsize=12, color='#e2e8f0', pad=10)
axes[1].tick_params(labelsize=7)

plt.suptitle('Dataset Overview', color='#e2e8f0', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
