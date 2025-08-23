# -*- coding: utf-8 -*-
"""
Simple ML Algorithm Comparison for Bat vs Rat Binary Classification

This script compares different ML algorithms to classify:
- "rat" behaviors (predator-focused responses) → label 0
- "bat" behaviors (competition-focused responses) → label 1

Uses Dataset1 only for clean, simple analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings('ignore')

print("=== SIMPLE BAT VS RAT BINARY CLASSIFICATION ===")
print("Using Dataset1 only for clean analysis")

#%% Load and prepare data
print("\n[STEP 1] Loading data...")

try:
    # Load only dataset1 (no complex merging)
    dataset1 = pd.read_csv('dataset1.csv')
    print(f"Loaded dataset1.csv: {dataset1.shape}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure dataset1.csv is in the current directory.")
    exit()

#%% Prepare binary classification data
print("\n[STEP 2] Preparing binary classification data...")

# Filter for clear "rat" and "bat" entries (no ambiguous ones)
original_rat = dataset1[dataset1['habit'] == 'rat'].copy()
original_bat = dataset1[dataset1['habit'].str.contains('bat', na=False) & 
                        (dataset1['habit'] != 'bat_and_rat')].copy()

print(f"Ground truth samples:")
print(f"  Rat entries: {len(original_rat)}")
print(f"  Bat entries: {len(original_bat)}")

# Combine ground truth data
ground_truth = pd.concat([original_rat, original_bat], ignore_index=True)

# Create binary labels: 0 = rat, 1 = bat
ground_truth['binary_label'] = (ground_truth['habit'] != 'rat').astype(int)

print(f"\nBinary label distribution:")
print(f"  Rat (0): {(ground_truth['binary_label'] == 0).sum()}")
print(f"  Bat (1): {(ground_truth['binary_label'] == 1).sum()}")

#%% Simple feature set (Dataset1 only)
print("\n[STEP 3] Using simple feature set...")

# Basic features from Dataset1
feature_cols = [
    'bat_landing_to_food',      # vigilance level
    'seconds_after_rat_arrival', # timing
    'risk',                     # risk level
    'reward',                   # reward outcome
    'hours_after_sunset',       # time context
    'month'                     # seasonal context
]

print(f"Feature set ({len(feature_cols)} features):")
for i, feat in enumerate(feature_cols, 1):
    print(f"  {i}. {feat}")

# Prepare features
X = ground_truth[feature_cols].values
y = ground_truth['binary_label'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features standardized successfully")

#%% Define algorithms to test
print("\n[STEP 4] Testing algorithms...")

# Simple algorithm set with KNN k=1-20
algorithms = {
    'Majority Class': DummyClassifier(strategy='most_frequent'),
    'KNN (k=1)': KNeighborsClassifier(n_neighbors=1),
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),
    'KNN (k=10)': KNeighborsClassifier(n_neighbors=10),
    'KNN (k=15)': KNeighborsClassifier(n_neighbors=15),
    'KNN (k=20)': KNeighborsClassifier(n_neighbors=20),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5),
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Naive Bayes': GaussianNB()
}

#%% Test algorithms
results = {}
cv_folds = 5
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

for name, model in algorithms.items():
    print(f"Testing {name}...")
    
    try:
        # Cross-validation
        cv_scores = cross_validate(model, X_scaled, y, 
                                  cv=cv, 
                                  scoring=['accuracy'],
                                  return_train_score=True)
        
        # Store results
        results[name] = {
            'test_accuracy': cv_scores['test_accuracy'].mean(),
            'test_accuracy_std': cv_scores['test_accuracy'].std(),
            'train_accuracy': cv_scores['train_accuracy'].mean(),
            'overfitting': cv_scores['train_accuracy'].mean() - cv_scores['test_accuracy'].mean()
        }
        
        print(f"  Test Accuracy: {results[name]['test_accuracy']:.3f} (+/- {results[name]['test_accuracy_std']*2:.3f})")
        print(f"  Overfitting: {results[name]['overfitting']:.3f}")
        
    except Exception as e:
        print(f"  Error: {e}")

#%% Results analysis
print("\n[STEP 5] Results summary...")

# Convert to DataFrame and sort by accuracy
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('test_accuracy', ascending=False)

print("\n=== ALGORITHM COMPARISON RESULTS ===")
print("Algorithm                Test_Acc  Train_Acc  Overfitting")
print("-" * 55)
for name, row in results_df.iterrows():
    print(f"{name:<20} {row['test_accuracy']:.3f}     {row['train_accuracy']:.3f}      {row['overfitting']:.3f}")

#%% Best algorithm selection
print("\n[STEP 6] Best algorithm selection...")

# Find algorithms with low overfitting (< 0.1)
low_overfitting = results_df[results_df['overfitting'] < 0.1]
if len(low_overfitting) > 0:
    # Among low overfitting algorithms, pick highest accuracy
    best_algorithm = low_overfitting.sort_values('test_accuracy', ascending=False).index[0]
    print(f"\nBest Algorithm (Low Overfitting): {best_algorithm}")
else:
    # If all have high overfitting, just pick highest accuracy
    best_algorithm = results_df.index[0]
    print(f"\nBest Algorithm (Highest Accuracy): {best_algorithm}")

best_results = results_df.loc[best_algorithm]
print(f"  Test Accuracy: {best_results['test_accuracy']:.3f}")
print(f"  Overfitting: {best_results['overfitting']:.3f}")

#%% Save simple visualization
print("\n[STEP 7] Creating visualization...")

import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Simple bar plot
plt.figure(figsize=(12, 6))
plt.style.use('default')

# Plot test accuracies
algorithms_to_plot = results_df.index[:10]  # Top 10
accuracies = [results_df.loc[name, 'test_accuracy'] for name in algorithms_to_plot]
colors = ['red' if results_df.loc[name, 'overfitting'] > 0.1 else 'green' for name in algorithms_to_plot]

bars = plt.bar(range(len(algorithms_to_plot)), accuracies, color=colors, alpha=0.7)
plt.title('ML Algorithm Comparison (Red=High Overfitting, Green=Low Overfitting)', fontweight='bold')
plt.xlabel('Algorithms')
plt.ylabel('Test Accuracy')
plt.xticks(range(len(algorithms_to_plot)), algorithms_to_plot, rotation=45, ha='right')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)

# Add accuracy labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/ml_algorithm_comparison.png', dpi=300, bbox_inches='tight')
print("Saved plot to: plots/ml_algorithm_comparison.png")
plt.show()

#%% Final recommendations
print("\n=== FINAL RECOMMENDATIONS ===")
print(f"1. Best Algorithm: {best_algorithm}")
print(f"   - Test Accuracy: {best_results['test_accuracy']:.3f}")
print(f"   - Overfitting: {best_results['overfitting']:.3f}")

# Show top 5 algorithms
print(f"\n2. Top 5 Algorithms:")
for i, (name, row) in enumerate(results_df.head(5).iterrows(), 1):
    overfitting_status = "Low" if row['overfitting'] < 0.1 else "High"
    print(f"   {i}. {name}: {row['test_accuracy']:.3f} (Overfitting: {overfitting_status})")

print(f"\n3. KNN Performance Analysis:")
knn_results = results_df[results_df.index.str.contains('KNN')].sort_values('test_accuracy', ascending=False)
if len(knn_results) > 0:
    best_knn = knn_results.index[0]
    print(f"   - Best KNN: {best_knn}")
    print(f"   - Accuracy: {knn_results.loc[best_knn, 'test_accuracy']:.3f}")
    print(f"   - Overfitting: {knn_results.loc[best_knn, 'overfitting']:.3f}")

print(f"\n4. Implementation Ready:")
print(f"   - Use {best_algorithm} for rat vs bat classification")
print(f"   - Feature set: {len(feature_cols)} simple features from Dataset1")
print(f"   - No Dataset2 merge complexity needed")

print(f"\nAlgorithm comparison completed!")