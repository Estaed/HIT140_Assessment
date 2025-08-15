# -*- coding: utf-8 -*-
#%%
"""
Machine Learning Algorithm Comparison for Habit Imputation

This script compares different ML algorithms to find the best one for
imputing missing 'habit' values in the bat predation risk dataset.

IMPORTANT: This is a TESTING/BENCHMARKING tool that finds the best algorithm.
The actual implementation and imputation happens in investigation_A.py Phase 3.

WORKFLOW:
1. Run this script first to compare algorithms and find the best one
2. Use the best algorithm parameters in investigation_A.py Phase 3
3. The accuracy results here are from cross-validation testing, not final implementation

NOTE: LightGBM often outperforms Gradient Boosting on structured data like this.
If LightGBM works properly, it may achieve higher accuracy and should be considered
as the primary choice for final implementation.

Algorithms tested:
1. Baseline models (Majority class, Random, Constant)
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)
6. Logistic Regression
7. Naive Bayes
8. Gradient Boosting
9. LightGBM (potentially best performer)
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
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=== MACHINE LEARNING ALGORITHM COMPARISON ===")
print("Comparing different ML algorithms for habit imputation...")

#%% Load and prepare data
print("\n[STEP 1] Loading and preparing data...")

# Load the prepared dataset
try:
    # Load the cleaned dataset exported from investigation_A.py
    cleaned_filename = 'datasets/dataset1_cleaned_for_ml.csv'
    dataset1_for_knn = pd.read_csv(cleaned_filename)
    print(f"✓ Loaded cleaned dataset from: {cleaned_filename}")
    
except FileNotFoundError:
    print(f"❌ Error: {cleaned_filename} not found!")
    print("Please run investigation_A.py first to create this file.")
    print("The file will be created after Phase 2 (habit classification) completes.")
    exit()
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("Please run investigation_A.py first to process the data.")
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

# Analyze class balance and complexity
print(f"\n=== CLASS BALANCE ANALYSIS ===")
print(f"Total classes: {len(class_names)}")

# Get class counts and find most/least frequent
class_counts = np.bincount(y_complete)
most_frequent_idx = class_counts.argmax()
least_frequent_idx = class_counts.argmin()

print(f"Most frequent class: {class_names[most_frequent_idx]}")
print(f"Least frequent class: {class_names[least_frequent_idx]}")

# Check if class imbalance is causing accuracy issues
min_class_count = class_counts.min()
max_class_count = class_counts.max()
imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')

print(f"Class imbalance ratio (max/min): {imbalance_ratio:.2f}")
if imbalance_ratio > 10:
    print("⚠️  WARNING: High class imbalance detected! This can cause accuracy issues.")
    print("   Solutions: Use class_weight='balanced' or SMOTE resampling")
elif imbalance_ratio > 5:
    print("⚠️  Moderate class imbalance detected. Consider using balanced class weights.")
else:
    print("✓ Class balance looks reasonable.")

#%% Define algorithms to test
print("\n[STEP 2] Defining algorithms to test...")
print("NOTE: LightGBM often outperforms Gradient Boosting on structured data.")
print("If LightGBM works properly, it may achieve higher accuracy than Gradient Boosting.")
print("="*60)

algorithms = {
    'Majority Class': DummyClassifier(strategy='most_frequent'),
    'Random': DummyClassifier(strategy='uniform'),
    'Constant (fast)': DummyClassifier(strategy='constant', constant=0),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, weights='distance'),   # Standard value
    'KNN (k=7)': KNeighborsClassifier(n_neighbors=7, weights='distance'),   # Good balance
    'KNN (k=11)': KNeighborsClassifier(n_neighbors=11, weights='distance'), # Larger but reasonable
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6, class_weight='balanced'),  # Reduced depth
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, class_weight='balanced'),  # Reduced depth
    'SVM (Linear)': SVC(kernel='linear', random_state=42, class_weight='balanced'),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42, class_weight='balanced'),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, 
        max_depth=3,  # Reduced from 5 to prevent overfitting
        learning_rate=0.05,  # Reduced from 0.1 to prevent overfitting
        subsample=0.7,  # Reduced from 0.8 to prevent overfitting
        random_state=42
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
        force_col_wise=True  # Better compatibility with scikit-learn
    )
}

#%% Test algorithms
print("\n[STEP 3] Testing algorithms with cross-validation...")

results = {}
cv_folds = 5
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

for name, model in algorithms.items():
    print(f"\nTesting {name}...")
    
    try:
        # Use scaled features for all algorithms
        X_train = X_complete_scaled
        
        # Special handling for LightGBM
        if name == 'LightGBM':
            print(f"    Debug: LightGBM data types - X: {X_train.dtype}, y: {y_complete.dtype}")
            print(f"    Debug: LightGBM shapes - X: {X_train.shape}, y: {y_complete.shape}")
            print(f"    Debug: LightGBM unique y values: {np.unique(y_complete)}")
            
            # Ensure data is properly formatted for LightGBM
            X_train_lgb = X_train.astype(np.float32)
            y_complete_lgb = y_complete.astype(np.int32)
            print(f"    Debug: Converted X to {X_train_lgb.dtype}, y to {y_complete_lgb.dtype}")
            
            # Check for any infinite values
            if np.any(np.isinf(X_train_lgb)):
                print(f"    Warning: Found infinite values in features, replacing with large finite values")
                X_train_lgb = np.nan_to_num(X_train_lgb, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Use the cleaned data for LightGBM
            X_train = X_train_lgb
            y_complete = y_complete_lgb
        
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
        
    except Exception as e:
        print(f"  ❌ Error testing {name}: {e}")
        if name == 'LightGBM':
            print(f"    Debug: LightGBM failed. Trying alternative approach...")
            try:
                # Try with even simpler LightGBM config
                simple_lgb = lgb.LGBMClassifier(
                    n_estimators=20,
                    max_depth=2,
                    random_state=42,
                    verbose=-1
                )
                cv_scores = cross_validate(simple_lgb, X_train, y_complete, 
                                          cv=cv, 
                                          scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                                          return_train_score=True)
                
                results[name] = {
                    'test_accuracy': cv_scores['test_accuracy'].mean(),
                    'test_accuracy_std': cv_scores['test_accuracy'].std(),
                    'train_accuracy': cv_scores['train_accuracy'].mean(),
                    'test_precision': cv_scores['test_precision_macro'].mean(),
                    'test_recall': cv_scores['test_recall_macro'].mean(),
                    'test_f1': cv_scores['test_f1_macro'].mean(),
                    'overfitting': cv_scores['train_accuracy'].mean() - cv_scores['test_accuracy'].mean()
                }
                print(f"    ✓ LightGBM alternative config succeeded!")
                print(f"    Test Accuracy: {results[name]['test_accuracy']:.4f}")
            except Exception as e2:
                print(f"    ❌ LightGBM alternative also failed: {e2}")
                print(f"    Skipping LightGBM for this comparison.")
                # Store NaN results for failed algorithms
                results[name] = {
                    'test_accuracy': np.nan,
                    'test_accuracy_std': np.nan,
                    'train_accuracy': np.nan,
                    'test_precision': np.nan,
                    'test_recall': np.nan,
                    'test_f1': np.nan,
                    'overfitting': np.nan
                }
        else:
            # Store NaN results for failed algorithms
            results[name] = {
                'test_accuracy': np.nan,
                'test_accuracy_std': np.nan,
                'train_accuracy': np.nan,
                'test_precision': np.nan,
                'test_recall': np.nan,
                'test_f1': np.nan,
                'overfitting': np.nan
            }

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
# Filter out NaN values for plotting
valid_accuracies = results_df.dropna(subset=['test_accuracy'])
accuracies = valid_accuracies['test_accuracy']
accuracies_std = valid_accuracies['test_accuracy_std']

if len(accuracies) > 0:
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
else:
    ax1.text(0.5, 0.5, 'No valid accuracy data to plot', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Test Accuracy by Algorithm', fontsize=14, fontweight='bold', color='black')

# Subplot 2: Overfitting Analysis
# Filter out NaN values for plotting
valid_overfitting = results_df.dropna(subset=['overfitting'])
overfitting = valid_overfitting['overfitting']

if len(overfitting) > 0:
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
else:
    ax2.text(0.5, 0.5, 'No valid overfitting data to plot', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Overfitting Analysis (Train - Test Accuracy)', fontsize=14, fontweight='bold', color='black')

# Subplot 3: Precision vs Recall
# Filter out NaN values for plotting
valid_data = results_df.dropna()
if len(valid_data) > 0:
    ax3.scatter(valid_data['test_precision'], valid_data['test_recall'], 
               s=100, alpha=0.7, c=range(len(valid_data)), cmap='viridis')
    for i, name in enumerate(valid_data.index):
        ax3.annotate(name, (valid_data['test_precision'].iloc[i], valid_data['test_recall'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, color='black')
ax3.set_title('Precision vs Recall', fontsize=14, fontweight='bold', color='black')
ax3.set_xlabel('Precision (Macro)', color='black')
ax3.set_ylabel('Recall (Macro)', color='black')
ax3.grid(True, alpha=0.3)
ax3.tick_params(colors='black')

# Subplot 4: F1 Score
# Filter out NaN values for plotting
valid_f1 = results_df.dropna(subset=['test_f1'])
f1_scores = valid_f1['test_f1']

if len(f1_scores) > 0:
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
else:
    ax4.text(0.5, 0.5, 'No valid F1 score data to plot', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('F1 Score (Macro) by Algorithm', fontsize=14, fontweight='bold', color='black')

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
print(f"   - Use {best_algorithm_name} for imputation in investigation_A.py Phase 3")
print(f"   - The accuracy results here are from testing, not final implementation")
print(f"   - Consider feature engineering if accuracy is still low")
print(f"   - Check data quality and class balance")
print(f"   - Try ensemble methods combining multiple algorithms")

# Special note about LightGBM
if 'LightGBM' in results and not np.isnan(results['LightGBM']['test_accuracy']):
    print(f"\n5. LightGBM Performance:")
    lgb_acc = results['LightGBM']['test_accuracy']
    if lgb_acc > results_df.iloc[0]['test_accuracy']:
        print(f"   ⭐ LightGBM outperformed Gradient Boosting! ({lgb_acc:.4f} vs {results_df.iloc[0]['test_accuracy']:.4f})")
        print(f"   Consider using LightGBM instead of Gradient Boosting for final implementation.")
    else:
        print(f"   LightGBM accuracy: {lgb_acc:.4f} (Gradient Boosting still better)")
elif 'LightGBM' in results:
    print(f"\n5. LightGBM Issue:")
    print(f"   ❌ LightGBM failed to run properly - this may be due to:")
    print(f"   - Data type compatibility issues")
    print(f"   - Feature scaling problems")
    print(f"   - Class imbalance affecting LightGBM more severely")
    print(f"   - Consider trying LightGBM with different parameters or data preprocessing")

print(f"\n✓ Algorithm comparison completed!")
print(f"✓ Files created:")
print(f"   - {comparison_plot_filename}")

#%% Analysis of accuracy drop and solutions
print(f"\n=== ACCURACY DROP ANALYSIS ===")
print(f"Previous accuracy: ~0.73")
print(f"Current accuracy: {results_df.iloc[0]['test_accuracy']:.3f}")
accuracy_drop = 0.73 - results_df.iloc[0]['test_accuracy']
print(f"Accuracy drop: {accuracy_drop:.3f}")

print(f"\n=== POSSIBLE CAUSES ===")
print(f"1. Increased number of habit categories:")
print(f"   - Previous: Likely fewer categories")
print(f"   - Current: {len(class_names)} categories")
print(f"   - More categories = harder classification")

print(f"\n2. Class imbalance:")
print(f"   - Imbalance ratio: {imbalance_ratio:.2f}")
if imbalance_ratio > 5:
    print(f"   - High imbalance detected - this significantly reduces accuracy")

print(f"\n3. Feature complexity:")
print(f"   - Current features: {len(feature_cols)}")
print(f"   - Feature names: {feature_cols}")

print(f"\n=== SOLUTIONS TO IMPROVE ACCURACY ===")
print(f"1. Class Balance Solutions:")
print(f"   ✓ Added class_weight='balanced' to algorithms")
print(f"   - Consider SMOTE resampling for severe imbalance")
print(f"   - Use stratified sampling in cross-validation")

print(f"\n2. Feature Engineering:")
print(f"   - Add interaction features (e.g., risk * reward)")
print(f"   - Create time-based features (e.g., time of day patterns)")
print(f"   - Add seasonal patterns")

print(f"\n3. Algorithm Tuning:")
print(f"   - Hyperparameter optimization for best algorithms")
print(f"   - Ensemble methods (voting, stacking)")
print(f"   - Feature selection to remove noise")

print(f"\n4. Data Quality:")
print(f"   - Check for outliers in features")
print(f"   - Verify feature distributions")
print(f"   - Consider reducing categories if accuracy remains low")

print(f"\n=== RECOMMENDED NEXT STEPS ===")
print(f"1. Run this updated comparison with balanced class weights")
print(f"2. If accuracy is still low, consider reducing to 4-6 habit categories")
print(f"3. Add feature engineering (interactions, time patterns)")
print(f"4. Use hyperparameter tuning on best algorithms")
print(f"5. Consider ensemble methods for final imputation")

print(f"\n=== LIGHTGBM TROUBLESHOOTING ===")
print(f"If LightGBM continues to fail, try these solutions:")
print(f"1. Data preprocessing:")
print(f"   - Ensure all features are numeric (no categorical strings)")
print(f"   - Check for infinite values in scaled features")
print(f"   - Verify class labels are properly encoded (0 to n-1)")
print(f"2. LightGBM parameters:")
print(f"   - Try even simpler config: n_estimators=10, max_depth=1")
print(f"   - Use different boosting_type: 'gbdt' or 'dart'")
print(f"   - Check LightGBM version compatibility with scikit-learn")
print(f"3. Alternative approach:")
print(f"   - Use CatBoost as LightGBM alternative")
print(f"   - Try XGBoost with different parameters")
print(f"   - Consider ensemble of multiple LightGBM configurations") 
# %%
