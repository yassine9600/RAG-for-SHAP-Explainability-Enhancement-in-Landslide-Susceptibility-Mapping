import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, f1_score, make_scorer)
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*80)
print("LANDSLIDE SUSCEPTIBILITY PREDICTION - LightGBM")
print("="*80)

# ============================================================================
# STEP 2: LOAD DATA
# ============================================================================
print("\n[STEP 2] Loading Data...")
print("-"*80)

file_path = '/content/SU_2_5_c4.xlsx'
df = pd.read_excel(file_path)

print(f"✓ Data loaded successfully!")
print(f"  Dataset Shape: {df.shape}")

print(f"\n  Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"    {i}. {col}")

print(f"\n  First 5 rows:")
print(df.head())

print(f"\n  Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("    ✓ No missing values found")
else:
    print(missing[missing > 0])

# ============================================================================
# STEP 3: SEPARATE FEATURES AND TARGET
# ============================================================================
print("\n[STEP 3] Separating Features and Target...")
print("-"*80)

possible_label_names = ['label', 'class', 'target', 'landslide', 'susceptibility']
label_col = None

for col in df.columns:
    if col.lower() in possible_label_names:
        label_col = col
        break

if label_col is None:
    label_col = df.columns[-1]
    print(f"⚠ Label column not identified. Using last column: '{label_col}'")
else:
    print(f"✓ Label column identified: '{label_col}'")

X = df.drop(columns=[label_col])
y = df[label_col]

print(f"\n  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")
print(f"\n  Class Distribution:")
print(y.value_counts())
print(f"\n  Class Proportions:")
print(y.value_counts(normalize=True))

# ============================================================================
# STEP 4: DROP UNWANTED COLUMNS (Slope, VS30, TWI)
# ============================================================================
print("\n[STEP 4] Dropping 'Slope', 'VS30', and 'TWI' Columns")
print("-"*80)

cols_to_drop = ['Slope', 'VS30', 'TWI']
existing_cols_to_drop = [col for col in cols_to_drop if col in X.columns]

if existing_cols_to_drop:
    X = X.drop(columns=existing_cols_to_drop)
    print(f"✓ {existing_cols_to_drop} columns dropped successfully!")
    print(f"  Remaining features count: {X.shape[1]}")
else:
    print(f"⚠ None of the specified columns {cols_to_drop} were found in the dataset")

print(f"\nFinal Features:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

feature_names = list(X.columns)

# ============================================================================
# STEP 5: NORMALIZATION
# ============================================================================
print("\n[STEP 5] Feature Normalization")
print("-"*80)

scaler = MinMaxScaler()
X_normalized = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

print(f"✓ Min-Max normalization complete! All features scaled to [0, 1]")

# ============================================================================
# STEP 6: TRAIN-TEST SPLIT (70-30)
# ============================================================================
print("\n[STEP 6] Train-Test Split (70-30)")
print("-"*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"✓ Data split complete!")
print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test set:     {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\n  Training class distribution:\n{y_train.value_counts()}")
print(f"\n  Test class distribution:\n{y_test.value_counts()}")

# ============================================================================
# STEP 7: LightGBM TRAINING WITH HYPERPARAMETER TUNING
# ============================================================================
print("\n[STEP 7] Training LightGBM with Hyperparameter Optimization")
print("="*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, pos_label=1)
sample_weights = np.where(y_train == 1, 3.5, 1.0)

lgb_params = {
    'num_leaves':        [15, 31, 63, 127],
    'max_depth':         [3, 5, 7, 10],
    'learning_rate':     [0.01, 0.03, 0.05, 0.1],
    'n_estimators':      [100, 300, 500, 700, 1000],
    'min_child_samples': [10, 20, 30, 50],
    'subsample':         [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree':  [0.6, 0.7, 0.8, 0.9],
    'reg_alpha':         [0, 5, 10, 20],
    'reg_lambda':        [0, 10, 20, 40]
}

lgb_model = lgb.LGBMClassifier(
    objective='binary',
    is_unbalance=True,
    random_state=42,
    verbose=-1
)

print("Starting LightGBM hyperparameter tuning...")
lgb_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=lgb_params,
    n_iter=100,
    scoring=f1_scorer,
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

lgb_search.fit(X_train, y_train, sample_weight=sample_weights)
best_model = lgb_search.best_estimator_
print(f"\n✓ LightGBM training complete!")
print(f"  Best CV F1:     {lgb_search.best_score_:.4f}")
print(f"  Best Params:    {lgb_search.best_params_}")

# ============================================================================
# STEP 8: EVALUATION
# ============================================================================
print("\n[STEP 8] Evaluating LightGBM Model")
print("="*80)

y_train_pred  = best_model.predict(X_train)
y_train_proba = best_model.predict_proba(X_train)[:, 1]
y_test_pred   = best_model.predict(X_test)
y_test_proba  = best_model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test,  y_test_pred)
train_auc = roc_auc_score(y_train,  y_train_proba)
test_auc  = roc_auc_score(y_test,   y_test_proba)
train_f1  = f1_score(y_train, y_train_pred, pos_label=1)
test_f1   = f1_score(y_test,  y_test_pred,  pos_label=1)

cm = confusion_matrix(y_test, y_test_pred)
recall    = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
precision = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
overfit   = train_auc - test_auc

print(f"\n  {'Metric':<25} {'Train':>10} {'Test':>10}")
print(f"  {'-'*45}")
print(f"  {'Accuracy':<25} {train_acc:>10.4f} {test_acc:>10.4f}")
print(f"  {'AUC-ROC':<25} {train_auc:>10.4f} {test_auc:>10.4f}")
print(f"  {'F1 Score':<25} {train_f1:>10.4f} {test_f1:>10.4f}")
print(f"  {'Recall (Landslide)':<25} {'':>10} {recall:>10.4f}")
print(f"  {'Precision (Landslide)':<25} {'':>10} {precision:>10.4f}")
print(f"  {'Overfitting Gap (AUC)':<25} {'':>10} {overfit:>10.4f}")

print(f"\n  Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['No Landslide', 'Landslide']))

# ============================================================================
# STEP 9: VISUALIZATIONS
# ============================================================================
print("\n[STEP 9] Generating Visualizations")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('LightGBM — Landslide Susceptibility Model', fontsize=16, fontweight='bold')

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
axes[0].plot(fpr, tpr, color='#9B5DE5', linewidth=3,
             label=f'LightGBM (AUC = {test_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='#999999', linewidth=1.8,
             linestyle='--', label='Random Classifier (AUC = 0.5000)')
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# --- Confusion Matrix ---
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[1],
            xticklabels=['No Landslide', 'Landslide'],
            yticklabels=['No Landslide', 'Landslide'],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# --- Feature Importance ---
importance = best_model.feature_importances_
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
fi_df = fi_df.sort_values('Importance', ascending=True)
axes[2].barh(fi_df['Feature'], fi_df['Importance'],
             color='#9B5DE5', alpha=0.8, edgecolor='black')
axes[2].set_xlabel('Importance', fontsize=12)
axes[2].set_title('Feature Importance', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('lgbm_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'lgbm_results.png'")
plt.show()

# ============================================================================
# STEP 10: EXPORT — SINGLE PKL BUNDLE
# ============================================================================
print("\n[STEP 10] Exporting Model Bundle")
print("="*80)

model_bundle = {
    # ── Core objects needed for inference ──────────────────────────────────
    'model':         best_model,          # trained LGBMClassifier
    'scaler':        scaler,              # fitted MinMaxScaler

    # ── Feature information ─────────────────────────────────────────────────
    'feature_names': feature_names,       # list of feature names (post-drop)
    'dropped_cols':  cols_to_drop,        # columns removed before training
    'label_col':     label_col,           # name of the target column

    # ── Hyperparameter search results ───────────────────────────────────────
    'best_params':   lgb_search.best_params_,
    'best_cv_f1':    lgb_search.best_score_,

    # ── Performance metrics ─────────────────────────────────────────────────
    'metrics': {
        'train_accuracy':  train_acc,
        'test_accuracy':   test_acc,
        'train_auc':       train_auc,
        'test_auc':        test_auc,
        'train_f1':        train_f1,
        'test_f1':         test_f1,
        'recall':          recall,
        'precision':       precision,
        'overfitting_auc': overfit,
        'confusion_matrix': cm,
    },

    # ── Dataset info ────────────────────────────────────────────────────────
    'data_info': {
        'n_samples':      len(X),
        'n_features':     X.shape[1],
        'train_size':     X_train.shape[0],
        'test_size':      X_test.shape[0],
        'class_counts':   y.value_counts().to_dict(),
    },

    # ── Meta ────────────────────────────────────────────────────────────────
    'model_name':     'LightGBM',
    'training_date':  pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
}

bundle_path = 'lgbm_landslide_bundle.pkl'
joblib.dump(model_bundle, bundle_path)
print(f"✓ Model bundle saved as: '{bundle_path}'")
print(f"\n  Bundle contains:")
for key in model_bundle:
    print(f"    • {key}")

# ============================================================================
# HOW TO LOAD AND USE IN ANOTHER APP
# ============================================================================
print("""
╔══════════════════════════════════════════════════════════════════╗
║              HOW TO USE THE BUNDLE IN ANOTHER APP               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  import joblib, pandas as pd                                     ║
║                                                                  ║
║  bundle = joblib.load('lgbm_landslide_bundle.pkl')               ║
║                                                                  ║
║  model         = bundle['model']                                 ║
║  scaler        = bundle['scaler']                                ║
║  feature_names = bundle['feature_names']                         ║
║                                                                  ║
║  # Prepare new data (must have same columns as feature_names)    ║
║  X_new = pd.DataFrame([your_data], columns=feature_names)        ║
║  X_scaled = scaler.transform(X_new)                             ║
║                                                                  ║
║  pred  = model.predict(X_scaled)          # 0 or 1              ║
║  proba = model.predict_proba(X_scaled)[:, 1]  # probability     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

print("="*80)
print(f"🏆 LightGBM | Test AUC: {test_auc:.4f} | Test F1: {test_f1:.4f} | Recall: {recall:.4f}")
print("="*80)
