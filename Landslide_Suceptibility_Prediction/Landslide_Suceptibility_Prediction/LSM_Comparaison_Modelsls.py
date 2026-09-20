import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve, f1_score, make_scorer)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*80)
print("LANDSLIDE SUSCEPTIBILITY PREDICTION - MULTI-MODEL COMPARISON")
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
print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Display column names
print(f"\n  Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"    {i}. {col}")

# Display first few rows
print(f"\n  First 5 rows:")
print(df.head())

# Check for missing values
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

# Identify the label column
possible_label_names = ['label', 'class', 'target', 'landslide', 'susceptibility']
label_col = None

for col in df.columns:
    if col.lower() in possible_label_names:
        label_col = col
        break

# If not found, assume last column is the label
if label_col is None:
    label_col = df.columns[-1]
    print(f"⚠ Label column not explicitly identified. Using last column: '{label_col}'")
else:
    print(f"✓ Label column identified: '{label_col}'")

# Separate features and target
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

cols_to_drop = ['Slope','VS30','TWI']
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

# ============================================================================
# STEP 5: NORMALIZATION
# ============================================================================
print("\n[STEP 5] Feature Normalization")
print("-"*80)
print("Applying Min-Max Normalization: (v - vmin) / (vmax - vmin)")

scaler = MinMaxScaler()
X_normalized = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

print(f"✓ Min-Max normalization complete!")
print(f"  All features scaled to range [0, 1]")

# ============================================================================
# STEP 6: TRAIN-TEST SPLIT (70-30)
# ============================================================================
print("\n[STEP 6] Train-Test Split (70-30)")
print("-"*80)

# Split: Train vs Test (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"✓ Data split complete!")
print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%) affinities")

print(f"\n  Training set class distribution:")
print(y_train.value_counts())
print(f"\n  Test set class distribution:")
print(y_test.value_counts())

# ============================================================================
# STEP 7: MODEL TRAINING - ALL MODELS
# ============================================================================
print("\n[STEP 7] Training Multiple Models with Hyperparameter Optimization")
print("="*80)

# Setup cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, pos_label=1)

# Calculate class weights
base_scale = (y_train == 0).sum() / (y_train == 1).sum()
sample_weights = np.where(y_train == 1, 3.5, 1.0)

# Dictionary to store all models and results
models = {}
results = {}

# ----------------------------------------------------------------------------
# MODEL 1: XGBoost
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("MODEL 1: XGBoost")
print("="*80)

xgb_params = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.03, 0.05, 0.07],
    'n_estimators': [500, 700, 1000, 1200],
    'min_child_weight': [10, 15, 20],
    'gamma': [0.3, 0.5, 0.7, 1.0],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'reg_alpha': [5, 10, 20, 30],
    'reg_lambda': [10, 20, 30, 50]
}

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=3.5,
    random_state=42,
    use_label_encoder=False
)

print("Starting XGBoost hyperparameter tuning...")
xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_params,
    n_iter=100,
    scoring=f1_scorer,
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train, sample_weight=sample_weights)
models['XGBoost'] = xgb_search.best_estimator_
print(f"✓ XGBoost training complete! Best CV F1: {xgb_search.best_score_:.4f}")

# ----------------------------------------------------------------------------
# MODEL 2: K-Nearest Neighbors (KNN)
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("MODEL 2: K-Nearest Neighbors (KNN)")
print("="*80)

knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
    'leaf_size': [20, 30, 40, 50]
}

knn_model = KNeighborsClassifier()

print("Starting KNN hyperparameter tuning...")
knn_search = RandomizedSearchCV(
    estimator=knn_model,
    param_distributions=knn_params,
    n_iter=50,
    scoring=f1_scorer,
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

knn_search.fit(X_train, y_train)
models['KNN'] = knn_search.best_estimator_
print(f"✓ KNN training complete! Best CV F1: {knn_search.best_score_:.4f}")

# ----------------------------------------------------------------------------
# MODEL 3: Random Forest
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("MODEL 3: Random Forest")
print("="*80)

rf_params = {
    'n_estimators': [100, 200, 300, 500, 700],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2', 0.5, 0.7],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 3.5}]
}

rf_model = RandomForestClassifier(random_state=42)

print("Starting Random Forest hyperparameter tuning...")
rf_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_params,
    n_iter=100,
    scoring=f1_scorer,
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_train, y_train)
models['Random Forest'] = rf_search.best_estimator_
print(f"✓ Random Forest training complete! Best CV F1: {rf_search.best_score_:.4f}")

# ----------------------------------------------------------------------------
# MODEL 4: AdaBoost
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("MODEL 4: AdaBoost")
print("="*80)

ada_params = {
    'n_estimators': [50, 100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
    'algorithm': ['SAMME', 'SAMME.R']
}

ada_model = AdaBoostClassifier(random_state=42)

print("Starting AdaBoost hyperparameter tuning...")
ada_search = RandomizedSearchCV(
    estimator=ada_model,
    param_distributions=ada_params,
    n_iter=50,
    scoring=f1_scorer,
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

ada_search.fit(X_train, y_train, sample_weight=sample_weights)
models['AdaBoost'] = ada_search.best_estimator_
print(f"✓ AdaBoost training complete! Best CV F1: {ada_search.best_score_:.4f}")

# ----------------------------------------------------------------------------
# MODEL 5: LightGBM
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("MODEL 5: LightGBM")
print("="*80)

lgb_params = {
    'num_leaves': [15, 31, 63, 127],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'n_estimators': [100, 300, 500, 700, 1000],
    'min_child_samples': [10, 20, 30, 50],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 5, 10, 20],
    'reg_lambda': [0, 10, 20, 40]
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
models['LightGBM'] = lgb_search.best_estimator_
print(f"✓ LightGBM training complete! Best CV F1: {lgb_search.best_score_:.4f}")

# ============================================================================
# STEP 8: EVALUATE ALL MODELS
# ============================================================================
print("\n[STEP 8] Evaluating All Models")
print("="*80)

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    train_f1 = f1_score(y_train, y_train_pred, pos_label=1)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    landslide_recall = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    landslide_precision = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0

    # Store results
    results[model_name] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'recall': landslide_recall,
        'precision': landslide_precision,
        'confusion_matrix': cm,
        'y_test_proba': y_test_proba
    }

    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    print(f"  Recall: {landslide_recall:.4f}")
    print(f"  Precision: {landslide_precision:.4f}")

# ============================================================================
# STEP 9: MODEL COMPARISON AND BEST MODEL SELECTION
# ============================================================================
print("\n[STEP 9] Model Comparison")
print("="*80)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test AUC': [results[m]['test_auc'] for m in results.keys()],
    'Test F1': [results[m]['test_f1'] for m in results.keys()],
    'Test Accuracy': [results[m]['test_acc'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Overfitting (AUC)': [results[m]['train_auc'] - results[m]['test_auc'] for m in results.keys()]
})

comparison_df = comparison_df.sort_values('Test AUC', ascending=False)

print("\nModel Performance Comparison:")
print(comparison_df.to_string(index=False))

# Identify best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = models[best_model_name]
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"    Test AUC: {comparison_df.iloc[0]['Test AUC']:.4f}")
print(f"    Test F1: {comparison_df.iloc[0]['Test F1']:.4f}")

# ============================================================================
# STEP 10: SAVE BEST MODEL
# ============================================================================
print("\n[STEP 10] Saving Best Model")
print("="*80)

# Save model
model_filename = f'best_landslide_model_{best_model_name.lower().replace(" ", "_")}.pkl'
joblib.dump(best_model, model_filename)
print(f"✓ Best model saved as: {model_filename}")

# Save scaler
scaler_filename = 'feature_scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"✓ Feature scaler saved as: {scaler_filename}")

# Save feature names
feature_names_filename = 'feature_names.pkl'
joblib.dump(list(X.columns), feature_names_filename)
print(f"✓ Feature names saved as: {feature_names_filename}")

# Create model info file
model_info = {
    'best_model_name': best_model_name,
    'test_auc': comparison_df.iloc[0]['Test AUC'],
    'test_f1': comparison_df.iloc[0]['Test F1'],
    'test_accuracy': comparison_df.iloc[0]['Test Accuracy'],
    'recall': comparison_df.iloc[0]['Recall'],
    'precision': comparison_df.iloc[0]['Precision'],
    'feature_names': list(X.columns),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

info_filename = 'model_info.pkl'
joblib.dump(model_info, info_filename)
print(f"✓ Model info saved as: {info_filename}")

print("\n📦 Model Package Ready for Deployment:")
print(f"    1. {model_filename} - Trained model")
print(f"    2. {scaler_filename} - Feature scaler")
print(f"    3. {feature_names_filename} - Feature names")
print(f"    4. {info_filename} - Model metadata")

# ============================================================================
# STEP 11: VISUALIZATIONS
# ============================================================================
print("\n[STEP 11] Generating Visualizations")
print("="*80)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 1. AUC Comparison (Main Chart)
ax1 = plt.subplot(2, 3, 1)
model_names = comparison_df['Model'].values
test_aucs = comparison_df['Test AUC'].values
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

bars = ax1.barh(model_names, test_aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('AUC Score', fontsize=12, fontweight='bold')
ax1.set_title('Model AUC Comparison (Test Set)', fontsize=14, fontweight='bold')
ax1.set_xlim([0, 1])
ax1.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, auc) in enumerate(zip(bars, test_aucs)):
    ax1.text(auc + 0.01, bar.get_y() + bar.get_height()/2,
             f'{auc:.4f}', va='center', fontsize=10, fontweight='bold')

# Highlight best model
best_idx = 0
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

# 2. ROC Curves for All Models
ax2 = plt.subplot(2, 3, 2)
for i, (model_name, color) in enumerate(zip(results.keys(), colors)):
    fpr, tpr, _ = roc_curve(y_test, results[model_name]['y_test_proba'])
    ax2.plot(fpr, tpr, color=color, lw=2,
             label=f'{model_name} (AUC={results[model_name]["test_auc"]:.3f})')

ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc="lower right", fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. F1 Score Comparison
ax3 = plt.subplot(2, 3, 3)
f1_scores = comparison_df['Test F1'].values
bars = ax3.barh(model_names, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
ax3.set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
ax3.set_xlim([0, 1])
ax3.grid(True, alpha=0.3, axis='x')

for bar, f1 in zip(bars, f1_scores):
    ax3.text(f1 + 0.01, bar.get_y() + bar.get_height()/2,
             f'{f1:.4f}', va='center', fontsize=10, fontweight='bold')

bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

# 4. Precision-Recall Comparison
ax4 = plt.subplot(2, 3, 4)
precisions = comparison_df['Precision'].values
recalls = comparison_df['Recall'].values

x = np.arange(len(model_names))
width = 0.35

bars1 = ax4.bar(x - width/2, precisions, width, label='Precision',
                color='skyblue', alpha=0.8, edgecolor='black')
bars2 = ax4.bar(x + width/2, recalls, width, label='Recall',
                color='salmon', alpha=0.8, edgecolor='black')

ax4.set_ylabel('Score', fontsize=11)
ax4.set_title('Precision vs Recall by Model', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(model_names, rotation=45, ha='right')
ax4.legend()
ax4.set_ylim([0, 1.1])
ax4.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 5. Overfitting Analysis
ax5 = plt.subplot(2, 3, 5)
overfitting = comparison_df['Overfitting (AUC)'].values * 100
bars = ax5.barh(model_names, overfitting, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.set_xlabel('Overfitting Gap (%)', fontsize=12, fontweight='bold')
ax5.set_title('Overfitting Analysis (Train-Test AUC Gap)', fontsize=14, fontweight='bold')
ax5.axvline(x=8, color='red', linestyle='--', linewidth=2, label='Target Threshold (8%)')
ax5.grid(True, alpha=0.3, axis='x')
ax5.legend()

for bar, gap in zip(bars, overfitting):
    ax5.text(gap + 0.3, bar.get_y() + bar.get_height()/2,
             f'{gap:.2f}%', va='center', fontsize=10, fontweight='bold')

# 6. Best Model Confusion Matrix
ax6 = plt.subplot(2, 3, 6)
cm_best = results[best_model_name]['confusion_matrix']
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax6,
            xticklabels=['No Landslide', 'Landslide'],
            yticklabels=['No Landslide', 'Landslide'],
            annot_kws={'size': 12, 'weight': 'bold'})
ax6.set_ylabel('Actual', fontsize=11)
ax6.set_xlabel('Predicted', fontsize=11)
ax6.set_title(f'Best Model Confusion Matrix\n({best_model_name})',
              fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
print("✓ Comparison visualizations saved as 'model_comparison_results.png'")
plt.show()

# STANDALONE ROC CURVES CHART
print("\n[STANDALONE] Generating High-Quality ROC Curves Chart...")
roc_colors = {'XGBoost': '#E63946', 'KNN': '#457B9D', 'Random Forest': '#2A9D8F', 'AdaBoost': '#E9C46A', 'LightGBM': '#9B5DE5'}
sorted_models = sorted(results.keys(), key=lambda m: results[m]['test_auc'], reverse=True)
fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
for model_name in sorted_models:
    fpr, tpr, _ = roc_curve(y_test, results[model_name]['y_test_proba'])
    ax_roc.plot(fpr, tpr, color=roc_colors.get(model_name, '#FFFFFF'), linewidth=3, label=f'{model_name} (AUC = {results[model_name]["test_auc"]:.4f})')
ax_roc.plot([0, 1], [0, 1], color='#999999', linewidth=1.8, linestyle='--', label='Random Classifier (AUC = 0.5000)')
ax_roc.set_xlabel('False Positive Rate', fontsize=14)
ax_roc.set_ylabel('True Positive Rate', fontsize=14)
ax_roc.set_title('ROC Curves — Model Comparison', fontsize=20, fontweight='bold')
ax_roc.legend(loc='lower right')
plt.savefig('ROC_Curves_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 12: FINAL SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY REPORT")
print("="*80)
print(f"Features used: {X.shape[1]}")
print(f"🏆 BEST MODEL: {best_model_name} (AUC: {comparison_df.iloc[0]['Test AUC']:.4f})")
print("="*80)