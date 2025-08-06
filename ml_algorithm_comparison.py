# -*- coding: utf-8 -*-
"""
Machine Learning Algorithm Comparison for Habit Imputation

This script compares different ML algorithms to find the best one for
imputing missing 'habit' values in the bat predation risk dataset.

Algorithms tested:
1. Baseline models (Majority class, Random, Constant)
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)
6. Logistic Regression
7. Naive Bayes
8. Gradient Boosting
9. XGBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=== MACHINE LEARNING ALGORITHM COMPARISON ===")
print("Comparing different ML algorithms for habit imputation...")

#%% Load and prepare data
print("\n[STEP 1] Loading and preparing data...")

# Load the prepared dataset
try:
    dataset1_for_knn = pd.read_csv('datasets/produced/dataset1_prepared_for_knn.csv')
    print("✓ Loaded datasets/produced/dataset1_prepared_for_knn.csv")
except FileNotFoundError:
    print("❌ Error: datasets/produced/dataset1_prepared_for_knn.csv not found!")
    print("Please run investigation_a_preprocessing.py first to create this file.")
    exit()

# Separate complete and missing data
habit_missing = dataset1_for_knn['habit'].isnull()
df_complete = dataset1_for_knn[~habit_missing].copy()
df_missing = dataset1_for_knn[habit_missing].copy()

print(f"Complete observations: {len(df_complete)}")
print(f"Missing observations: {len(df_missing)}")

# Select features for ML algorithms
feature_cols = [
    'bat_landing_to_food',
    'seconds_after_rat_arrival', 
    'risk',
    'reward',
    'hours_after_sunset',
    'month',
    'season'
]

# Prepare feature matrices
X_complete = df_complete[feature_cols].values
X_missing = df_missing[feature_cols].values

# Check for NaN values in features (excluding habit)
print(f"\nChecking for NaN values in features:")
for i, col in enumerate(feature_cols):
    nan_count = np.isnan(X_complete[:, i]).sum()
    print(f"  {col}: {nan_count} NaN values")

# Only impute if there are actual NaN values
if np.isnan(X_complete).any():
    print(f"\nFound NaN values in features. Applying imputation...")
    imputer = SimpleImputer(strategy='median')
    X_complete = imputer.fit_transform(X_complete)
    X_missing = imputer.transform(X_missing)
else:
    print(f"\nNo NaN values found in features. Skipping imputation.")
    # Convert to float if needed
    X_complete = X_complete.astype(float)
    X_missing = X_missing.astype(float)

# Normalize/Standardize features for better ML performance
print(f"\nNormalizing features for better ML performance...")
scaler = StandardScaler()
X_complete_scaled = scaler.fit_transform(X_complete)
X_missing_scaled = scaler.transform(X_missing)

print(f"Feature scaling completed:")
print(f"  Original feature ranges: {X_complete.min(axis=0)} to {X_complete.max(axis=0)}")
print(f"  Scaled feature ranges: {X_complete_scaled.min(axis=0)} to {X_complete_scaled.max(axis=0)}")

# Encode habit values for classification
le = LabelEncoder()
y_complete = le.fit_transform(df_complete['habit'])

# Get class names for reporting
class_names = le.classes_
print(f"\nHabit classes: {class_names}")
print(f"Class distribution:")
for i, class_name in enumerate(class_names):
    count = (y_complete == i).sum()
    print(f"  {class_name}: {count} ({count/len(y_complete)*100:.1f}%)")

#%% Define algorithms to test
print("\n[STEP 2] Defining algorithms to test...")

algorithms = {
    'Majority Class': DummyClassifier(strategy='most_frequent'),
    'Random': DummyClassifier(strategy='uniform'),
    'Constant (fast)': DummyClassifier(strategy='constant', constant=0),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'KNN (k=11)': KNeighborsClassifier(n_neighbors=11, weights='distance'),
    'KNN (k=20)': KNeighborsClassifier(n_neighbors=20, weights='distance'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
}

#%% Test algorithms
print("\n[STEP 3] Testing algorithms with cross-validation...")

results = {}
cv_folds = 5
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

for name, model in algorithms.items():
    print(f"\nTesting {name}...")
    
    # Use scaled features for all algorithms
    X_train = X_complete_scaled
    
    # Perform cross-validation
    cv_scores = cross_validate(model, X_train, y_complete, 
                              cv=cv, 
                              scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                              return_train_score=True)
    
    # Store results
    results[name] = {
        'test_accuracy': cv_scores['test_accuracy'].mean(),
        'test_accuracy_std': cv_scores['test_accuracy'].std(),
        'train_accuracy': cv_scores['train_accuracy'].mean(),
        'test_precision': cv_scores['test_precision_macro'].mean(),
        'test_recall': cv_scores['test_recall_macro'].mean(),
        'test_f1': cv_scores['test_f1_macro'].mean(),
        'overfitting': cv_scores['train_accuracy'].mean() - cv_scores['test_accuracy'].mean()
    }
    
    print(f"  Test Accuracy: {results[name]['test_accuracy']:.4f} (+/- {results[name]['test_accuracy_std']*2:.4f})")
    print(f"  Train Accuracy: {results[name]['train_accuracy']:.4f}")
    print(f"  Overfitting: {results[name]['overfitting']:.4f}")

#%% Create results comparison
print("\n[STEP 4] Creating results comparison...")

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('test_accuracy', ascending=False)

print("\n=== ALGORITHM COMPARISON RESULTS ===")
print(results_df.round(4))

#%% Visualize results
print("\n[STEP 5] Creating visualizations...")

# Create plots folder if it doesn't exist
import os
plots_folder = 'plots'
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

# Set white background style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Plot 1: Test Accuracy Comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.patch.set_facecolor('white')

# Subplot 1: Test Accuracy
accuracies = results_df['test_accuracy']
accuracies_std = results_df['test_accuracy_std']
colors = ['green' if x > 0.5 else 'orange' if x > 0.4 else 'red' for x in accuracies]

bars = ax1.bar(range(len(accuracies)), accuracies, yerr=accuracies_std*2, 
               color=colors, alpha=0.7, capsize=5)
ax1.set_title('Test Accuracy by Algorithm', fontsize=14, fontweight='bold', color='black')
ax1.set_xlabel('Algorithms', color='black')
ax1.set_ylabel('Test Accuracy', color='black')
ax1.set_xticks(range(len(accuracies)))
ax1.set_xticklabels(accuracies.index, rotation=45, ha='right', color='black')
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(colors='black')

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', color='black')

# Subplot 2: Overfitting Analysis
overfitting = results_df['overfitting']
colors = ['green' if x < 0.1 else 'orange' if x < 0.2 else 'red' for x in overfitting]

bars = ax2.bar(range(len(overfitting)), overfitting, color=colors, alpha=0.7)
ax2.set_title('Overfitting Analysis (Train - Test Accuracy)', fontsize=14, fontweight='bold', color='black')
ax2.set_xlabel('Algorithms', color='black')
ax2.set_ylabel('Overfitting Score', color='black')
ax2.set_xticks(range(len(overfitting)))
ax2.set_xticklabels(overfitting.index, rotation=45, ha='right', color='black')
ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
ax2.grid(axis='y', alpha=0.3)
ax2.legend()
ax2.tick_params(colors='black')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, overfitting)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold', color='black')

# Subplot 3: Precision vs Recall
ax3.scatter(results_df['test_precision'], results_df['test_recall'], 
           s=100, alpha=0.7, c=range(len(results_df)), cmap='viridis')
for i, name in enumerate(results_df.index):
    ax3.annotate(name, (results_df['test_precision'].iloc[i], results_df['test_recall'].iloc[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8, color='black')
ax3.set_title('Precision vs Recall', fontsize=14, fontweight='bold', color='black')
ax3.set_xlabel('Precision (Macro)', color='black')
ax3.set_ylabel('Recall (Macro)', color='black')
ax3.grid(True, alpha=0.3)
ax3.tick_params(colors='black')

# Subplot 4: F1 Score
f1_scores = results_df['test_f1']
colors = ['green' if x > 0.5 else 'orange' if x > 0.4 else 'red' for x in f1_scores]

bars = ax4.bar(range(len(f1_scores)), f1_scores, color=colors, alpha=0.7)
ax4.set_title('F1 Score (Macro) by Algorithm', fontsize=14, fontweight='bold', color='black')
ax4.set_xlabel('Algorithms', color='black')
ax4.set_ylabel('F1 Score', color='black')
ax4.set_xticks(range(len(f1_scores)))
ax4.set_xticklabels(f1_scores.index, rotation=45, ha='right', color='black')
ax4.set_ylim(0, 1)
ax4.grid(axis='y', alpha=0.3)
ax4.tick_params(colors='black')

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold', color='black')

plt.tight_layout()

# Save the comparison plot
comparison_plot_filename = os.path.join(plots_folder, 'ml_algorithm_comparison.png')
plt.savefig(comparison_plot_filename, dpi=300, bbox_inches='tight')
print(f"✓ Saved comparison plot to: {comparison_plot_filename}")

plt.show()

#%% Detailed analysis of best algorithm
print("\n[STEP 6] Detailed analysis of best algorithm...")

best_algorithm_name = results_df.index[0]
best_algorithm = algorithms[best_algorithm_name]

print(f"\nBest Algorithm: {best_algorithm_name}")
print(f"Test Accuracy: {results_df.loc[best_algorithm_name, 'test_accuracy']:.4f}")
print(f"Overfitting: {results_df.loc[best_algorithm_name, 'overfitting']:.4f}")

# Train the best model on full data using scaled features
best_model = best_algorithm.fit(X_complete_scaled, y_complete)

# Predict missing values using scaled features
y_pred = best_model.predict(X_missing_scaled)
predicted_habits = le.inverse_transform(y_pred)

# Create imputed dataset
df_imputed = dataset1_for_knn.copy()
df_imputed.loc[habit_missing, 'habit'] = predicted_habits

# Show imputation results
print(f"\nImputation Results using {best_algorithm_name}:")
imputed_dist = df_imputed['habit'].value_counts()
for habit, count in imputed_dist.items():
    print(f"  {habit}: {count}")

# Show imputation results summary
print(f"\nImputation Results Summary using {best_algorithm_name}:")
imputed_dist = df_imputed['habit'].value_counts()
for habit, count in imputed_dist.items():
    print(f"  {habit}: {count}")

print(f"\n✓ Best algorithm identified: {best_algorithm_name}")
print(f"✓ Test Accuracy: {results_df.loc[best_algorithm_name, 'test_accuracy']:.4f}")
print(f"✓ Overfitting: {results_df.loc[best_algorithm_name, 'overfitting']:.4f}")

# Print final recommendations
print("\n=== FINAL RECOMMENDATIONS ===")
print(f"1. Best Algorithm: {best_algorithm_name}")
print(f"   - Test Accuracy: {results_df.loc[best_algorithm_name, 'test_accuracy']:.4f}")
print(f"   - Overfitting: {results_df.loc[best_algorithm_name, 'overfitting']:.4f}")

# Check if any algorithm has acceptable performance
acceptable_algorithms = results_df[results_df['test_accuracy'] > 0.6]
if len(acceptable_algorithms) > 0:
    print(f"\n2. Acceptable Algorithms (Accuracy > 0.6):")
    for name in acceptable_algorithms.index:
        print(f"   - {name}: {acceptable_algorithms.loc[name, 'test_accuracy']:.4f}")
else:
    print(f"\n2. No algorithm achieved accuracy > 0.6")
    print(f"   - Best: {best_algorithm_name} ({results_df.loc[best_algorithm_name, 'test_accuracy']:.4f})")

# Check for overfitting
low_overfitting = results_df[results_df['overfitting'] < 0.1]
if len(low_overfitting) > 0:
    print(f"\n3. Algorithms with Low Overfitting (< 0.1):")
    for name in low_overfitting.index:
        print(f"   - {name}: {low_overfitting.loc[name, 'overfitting']:.4f}")

print(f"\n4. Next Steps:")
print(f"   - Use {best_algorithm_name} for imputation")
print(f"   - Consider feature engineering if accuracy is still low")
print(f"   - Check data quality and class balance")
print(f"   - Try ensemble methods combining multiple algorithms")

print(f"\n✓ Algorithm comparison completed!")
print(f"✓ Files created:")
print(f"   - {comparison_plot_filename}") 