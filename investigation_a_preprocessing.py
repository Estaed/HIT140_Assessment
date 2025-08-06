# -*- coding: utf-8 -*-
#%% Import libraries
"""
Investigation A: Data Preprocessing for Predation Risk Perception Analysis

Objective: Prepare data to analyze if bats perceive rats as predators
by examining vigilance levels and avoidance behaviors.

Steps:
1. Load datasets
2. Fix missing values in Dataset 1 using KNN
3. Create vigilance indicators
4. Merge with Dataset 2 for environmental context
5. Prepare final dataset for analysis
"""

# # Investigation A: Predation Risk Perception Analysis
# 
# Analyzing if bats show increased vigilance when rats are present


import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, r2_score, classification_report
from sklearn.impute import SimpleImputer
import warnings
import re
warnings.filterwarnings('ignore')

print("=== INVESTIGATION A: PREDATION RISK PERCEPTION ANALYSIS ===")
print("Analyzing if bats show increased vigilance when rats are present...")

#%% STEP 1: Load and explore datasets
print("\n[STEP 1] Loading datasets and creating folder structure...")

# Create plots folder for storing visualizations
import os
plots_folder = 'plots'
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)
    print(f"✓ Created '{plots_folder}' folder for storing plots")

# Create datasets folder structure
datasets_folder = 'datasets'
original_folder = os.path.join(datasets_folder, 'original')
produced_folder = os.path.join(datasets_folder, 'produced')

# Create folders if they don't exist
for folder in [datasets_folder, original_folder, produced_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"✓ Created folder: {folder}")

# Copy original datasets to original folder
import shutil
original_dataset1_path = os.path.join(original_folder, 'dataset1.csv')
original_dataset2_path = os.path.join(original_folder, 'dataset2.csv')

if not os.path.exists(original_dataset1_path):
    shutil.copy('dataset1.csv', original_dataset1_path)
    print(f"✓ Copied dataset1.csv to: {original_dataset1_path}")

if not os.path.exists(original_dataset2_path):
    shutil.copy('dataset2.csv', original_dataset2_path)
    print(f"✓ Copied dataset2.csv to: {original_dataset2_path}")

# Load the datasets from original folder
dataset1 = pd.read_csv(original_dataset1_path)
dataset2 = pd.read_csv(original_dataset2_path)

print(f"\nDataset 1 (Bat Landings): {dataset1.shape[0]} observations, {dataset1.shape[1]} features")
print(f"Dataset 2 (Observation Periods): {dataset2.shape[0]} observations, {dataset2.shape[1]} features")

# Check missing values
print("\nMissing values in Dataset 1:")
missing_summary = pd.DataFrame({
    'Column': dataset1.columns,
    'Missing_Count': dataset1.isnull().sum(),
    'Missing_Percentage': (dataset1.isnull().sum() / len(dataset1) * 100).round(2)
})
print(missing_summary[missing_summary['Missing_Count'] > 0])

#%% Display basic information about datasets
# Explore Datasets
print("\nDataset 1 - First 5 rows:")
print(dataset1.head())

print("\nDataset 1 - Data types:")
print(dataset1.dtypes)

print("\nDataset 1 - Basic statistics:")
print(dataset1.describe())

print("\nDataset 2 - First 5 rows:")
print(dataset2.head())

print("\nDataset 2 - Data types:")
print(dataset2.dtypes)

#%% STEP 2.1: Convert time columns
print("\n[STEP 2] Converting time columns to datetime...")
time_cols_ds1 = ['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time']
for col in time_cols_ds1:
    dataset1[col] = pd.to_datetime(dataset1[col], format='%d/%m/%Y %H:%M')

dataset2['time'] = pd.to_datetime(dataset2['time'], format='%d/%m/%Y %H:%M')

print("Time conversion completed!")

#%% STEP 2.2: Fix bat_landing_to_food values
print("\n[STEP 2.2] Fixing bat_landing_to_food values...")

# Function to convert small decimal values to proper integers
def fix_bat_landing_values(value):
    """
    Convert small decimal values to proper integers
    Example: 0.01759 -> 17, 0.03253 -> 32
    """
    if pd.isna(value):
        return value
    
    # If value is already a reasonable integer (> 1), keep it
    if value >= 1:
        return int(value)
    
    # If value is a small decimal (< 1), convert to integer
    if 0 < value < 1:
        # Convert to integer by multiplying by 1000 and rounding
        # This handles cases like 0.01759 -> 17.59 -> 18
        return int(round(value * 1000))
    
    return value

# Apply the fix to bat_landing_to_food column
print("Fixing bat_landing_to_food values...")
dataset1['bat_landing_to_food'] = dataset1['bat_landing_to_food'].apply(fix_bat_landing_values)

# Show before and after comparison
print("\nBefore vs After fixing bat_landing_to_food:")
print("Sample of original values that were fixed:")
original_values = pd.read_csv('dataset1.csv')['bat_landing_to_food']
small_values = original_values[(original_values > 0) & (original_values < 1)]
print("Original small values:", small_values.head(10).tolist())
print("Fixed values:", dataset1.loc[small_values.index, 'bat_landing_to_food'].head(10).tolist())

print(f"\nValue range after fixing:")
print(f"Min: {dataset1['bat_landing_to_food'].min()}")
print(f"Max: {dataset1['bat_landing_to_food'].max()}")
print(f"Mean: {dataset1['bat_landing_to_food'].mean():.2f}")

#%% STEP 2.3: Create copy and clean habit column
print("\n[STEP 2.3] Creating copy and cleaning habit column...")

# Create a copy prepared for KNN imputation
dataset1_for_knn = dataset1.copy()

# Remove any numeric values from habit column (keep only text and NaN)
# This ensures we only have categorical text values and NaN for KNN to predict
def is_numeric_or_coordinate(value):
    """
    Check if a value contains numeric data (including coordinates, ranges, etc.)
    Returns True if the value should be removed
    """
    if pd.isna(value):
        return False
    
    # Convert to string for pattern matching
    value_str = str(value)
    
    # Check for pure numbers
    if value_str.replace('.', '').replace(',', '').replace(';', '').replace(' ', '').isdigit():
        return True
    
    # Check for coordinate patterns (numbers separated by commas/semicolons)
    if re.search(r'\d+\.?\d*[,;]\s*\d+\.?\d*', value_str):
        return True
    
    # Check for ranges with numbers
    if re.search(r'\d+\.?\d*[-–]\d+\.?\d*', value_str):
        return True
    
    # Check if mostly numeric content
    numeric_chars = sum(c.isdigit() for c in value_str)
    total_chars = len(value_str.replace(' ', ''))
    if total_chars > 0 and numeric_chars / total_chars > 0.5:
        return True
    
    return False

# Apply the function to identify and remove numeric values
habit_numeric_mask = dataset1_for_knn['habit'].apply(is_numeric_or_coordinate)
if habit_numeric_mask.any():
    print(f"\nRemoving {habit_numeric_mask.sum()} numeric/coordinate values from habit column...")
    print("Examples of removed values:")
    removed_examples = dataset1_for_knn.loc[habit_numeric_mask, 'habit'].head(5).tolist()
    for example in removed_examples:
        print(f"  - {example}")
    dataset1_for_knn.loc[habit_numeric_mask, 'habit'] = np.nan

#%% STEP 2.4: Simplify habit categories
print("\n[STEP 2.4] Simplifying habit categories into 4 main classes...")

# Simplify habit categories into 4 main classes
def simplify_habit_category(habit_value):
    """
    Convert complex habit categories into 4 main classes:
    - fast: contains 'fast' but not 'bat' or 'rat'
    - bat: contains 'bat' but not 'rat' 
    - rat: contains 'rat' but not 'bat'
    - both: contains both 'bat' and 'rat'
    Any other categories will be set to NaN (missing data)
    """
    if pd.isna(habit_value):
        return habit_value
    
    habit_str = str(habit_value).lower()
    
    # Check if contains both 'bat' and 'rat'
    has_bat = 'bat' in habit_str
    has_rat = 'rat' in habit_str
    
    if has_bat and has_rat:
        return 'both'
    elif has_bat:
        return 'bat'
    elif has_rat:
        return 'rat'
    elif 'fast' in habit_str:
        return 'fast'
    else:
        # For any other categories, set to NaN (missing data)
        return np.nan

# Apply the simplification
print("\nSimplifying habit categories into 4 main classes...")
original_habits = dataset1_for_knn['habit'].value_counts()
print("Original habit categories:")
for habit, count in original_habits.head(10).items():
    print(f"  {habit}: {count}")

dataset1_for_knn['habit'] = dataset1_for_knn['habit'].apply(simplify_habit_category)

# Show the new simplified categories
print("\nSimplified habit categories:")
simplified_habits = dataset1_for_knn['habit'].value_counts()
for habit, count in simplified_habits.items():
    print(f"  {habit}: {count}")

#%% STEP 2.5: Create visualizations
print("\n[STEP 2.5] Creating visualizations...")

# Create visual plots for habit class distributions
print("\nCreating visual plots for habit class distributions...")

# Set white background style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('white')

# Plot 1: Original vs Simplified comparison
original_counts = original_habits.head(10)  # Top 10 original categories
simplified_counts = simplified_habits

# Original categories (top 10)
ax1.bar(range(len(original_counts)), original_counts.values, color='skyblue', alpha=0.7)
ax1.set_title('Original Habit Categories (Top 10)', fontsize=14, fontweight='bold', color='black')
ax1.set_xlabel('Categories', color='black')
ax1.set_ylabel('Count', color='black')
ax1.set_xticks(range(len(original_counts)))
ax1.set_xticklabels(original_counts.index, rotation=45, ha='right', color='black')
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(colors='black')

# Simplified categories
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # Red, Teal, Blue, Green
bars = ax2.bar(range(len(simplified_counts)), simplified_counts.values, color=colors[:len(simplified_counts)])
ax2.set_title('Simplified Habit Categories (4 Classes)', fontsize=14, fontweight='bold', color='black')
ax2.set_xlabel('Categories', color='black')
ax2.set_ylabel('Count', color='black')
ax2.set_xticks(range(len(simplified_counts)))
ax2.set_xticklabels(simplified_counts.index, rotation=0, color='black')

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, simplified_counts.values)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             str(count), ha='center', va='bottom', fontweight='bold', color='black')

ax2.grid(axis='y', alpha=0.3)
ax2.tick_params(colors='black')

# Add summary statistics
total_original = len(dataset1_for_knn)
total_simplified = simplified_counts.sum()
missing_count = total_original - total_simplified

fig.suptitle(f'Habit Category Simplification\n'
             f'Original: {len(original_habits)} categories → Simplified: {len(simplified_counts)} categories\n'
             f'Missing data: {missing_count} ({missing_count/total_original*100:.1f}%)', 
             fontsize=16, fontweight='bold', color='black')

plt.tight_layout()

# Save the plot to the plots folder
plot_filename = os.path.join(plots_folder, 'habit_category_simplification.png')
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"✓ Saved plot to: {plot_filename}")

plt.show()

# Print summary statistics
print(f"\nSummary Statistics:")
print(f"  Original categories: {len(original_habits)}")
print(f"  Simplified categories: {len(simplified_counts)}")
print(f"  Total observations: {total_original}")
print(f"  Valid observations: {total_simplified}")
print(f"  Missing observations: {missing_count} ({missing_count/total_original*100:.1f}%)")

# Create scatter plots for habit categories vs rat arrival timing
print("\nCreating scatter plots for habit categories vs rat arrival timing...")

# Filter data to only include valid habit categories (not NaN)
valid_data = dataset1_for_knn[dataset1_for_knn['habit'].notna()].copy()

# Create color map for habit categories
habit_colors = {
    'fast': '#FF6B6B',    # Red
    'bat': '#4ECDC4',     # Teal  
    'rat': '#45B7D1',     # Blue
    'both': '#96CEB4'     # Green
}

# Set white background style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('white')

# Plot 1: Seconds after rat arrival vs Risk
for habit in valid_data['habit'].unique():
    if pd.notna(habit):
        subset = valid_data[valid_data['habit'] == habit]
        ax1.scatter(subset['seconds_after_rat_arrival'], subset['risk'], 
                   c=habit_colors.get(habit, '#CCCCCC'), 
                   label=habit, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)

ax1.set_xlabel('Seconds After Rat Arrival', fontsize=12, fontweight='bold', color='black')
ax1.set_ylabel('Risk Level', fontsize=12, fontweight='bold', color='black')
ax1.set_title('Habit Categories: Risk vs Rat Arrival Timing', fontsize=14, fontweight='bold', color='black')
ax1.grid(True, alpha=0.3)
ax1.legend(title='Habit Categories', title_fontsize=10, fontsize=9)
ax1.tick_params(colors='black')

# Plot 2: Seconds after rat arrival vs Reward
for habit in valid_data['habit'].unique():
    if pd.notna(habit):
        subset = valid_data[valid_data['habit'] == habit]
        ax2.scatter(subset['seconds_after_rat_arrival'], subset['reward'], 
                   c=habit_colors.get(habit, '#CCCCCC'), 
                   label=habit, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)

ax2.set_xlabel('Seconds After Rat Arrival', fontsize=12, fontweight='bold', color='black')
ax2.set_ylabel('Reward Level', fontsize=12, fontweight='bold', color='black')
ax2.set_title('Habit Categories: Reward vs Rat Arrival Timing', fontsize=14, fontweight='bold', color='black')
ax2.grid(True, alpha=0.3)
ax2.legend(title='Habit Categories', title_fontsize=10, fontsize=9)
ax2.tick_params(colors='black')

# Add statistics text box
stats_text = f'Total Points: {len(valid_data)}\n'
for habit in valid_data['habit'].unique():
    if pd.notna(habit):
        count = len(valid_data[valid_data['habit'] == habit])
        avg_rat_timing = valid_data[valid_data['habit'] == habit]['seconds_after_rat_arrival'].mean()
        avg_risk = valid_data[valid_data['habit'] == habit]['risk'].mean()
        avg_reward = valid_data[valid_data['habit'] == habit]['reward'].mean()
        stats_text += f'{habit}: {count} points\n'
        stats_text += f'  Avg Rat Timing: {avg_rat_timing:.1f}s\n'
        stats_text += f'  Avg Risk: {avg_risk:.2f}, Avg Reward: {avg_reward:.2f}\n'

fig.text(0.02, 0.02, stats_text, transform=fig.transFigure, 
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), color='black')

plt.tight_layout()

# Save the plot
rat_timing_plot_filename = os.path.join(plots_folder, 'habit_rat_timing_analysis.png')
plt.savefig(rat_timing_plot_filename, dpi=300, bbox_inches='tight')
print(f"✓ Saved rat timing analysis to: {rat_timing_plot_filename}")

plt.show()

# Save the prepared dataset for KNN
# Save the prepared dataset to produced folder
prepared_dataset_path = os.path.join(produced_folder, 'dataset1_prepared_for_knn.csv')
dataset1_for_knn.to_csv(prepared_dataset_path, index=False)
print(f"\n✓ Created '{prepared_dataset_path}' - ready for KNN imputation!")
print(f"Dataset shape: {dataset1_for_knn.shape}")
print(f"Missing habit values: {dataset1_for_knn['habit'].isnull().sum()}")

# Show summary of prepared data
print("\nPrepared dataset summary:")
print(f"bat_landing_to_food range: {dataset1_for_knn['bat_landing_to_food'].min()} to {dataset1_for_knn['bat_landing_to_food'].max()}")
print(f"Habit categories: {dataset1_for_knn['habit'].nunique()}")
print(f"Available habit values: {dataset1_for_knn['habit'].value_counts().to_dict()}")

#%% STEP 3.1: Define Gradient Boosting imputation function with cross-validation and hyperparameter tuning
def tune_gradient_boosting(X, y, cv_folds=5):
    """
    Tune Gradient Boosting hyperparameters
    """
    print(f"\n  - Tuning Gradient Boosting hyperparameters...")
    
    # Define hyperparameter combinations to test (reduced for faster execution)
    param_combinations = [
        {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1},
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
        {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
    ]
    test_accuracies = []
    train_accuracies = []
    test_r2_scores = []
    train_r2_scores = []
    param_names = []
    
    for i, params in enumerate(param_combinations):
        gb = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            random_state=42
        )
        
        cv_scores = cross_validate(gb, X, y, cv=cv_folds, 
                                  scoring=['accuracy', 'r2'], 
                                  return_train_score=True)
        
        test_accuracies.append(cv_scores['test_accuracy'].mean())
        train_accuracies.append(cv_scores['train_accuracy'].mean())
        test_r2_scores.append(cv_scores['test_r2'].mean())
        train_r2_scores.append(cv_scores['train_r2'].mean())
        param_names.append(f"n={params['n_estimators']}, d={params['max_depth']}, lr={params['learning_rate']}")
        
        print(f"    Tested {param_names[-1]}, Test Acc: {cv_scores['test_accuracy'].mean():.4f}")
    
    # Find optimal parameters based on test accuracy
    optimal_idx = np.argmax(test_accuracies)
    optimal_params = param_combinations[optimal_idx]
    optimal_accuracy = max(test_accuracies)
    
    print(f"\n  - Optimal parameters found: {optimal_params}")
    print(f"  - Optimal test accuracy: {optimal_accuracy:.4f}")
    
    # Set white background style
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Create tuning plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    
    # Plot 1: Accuracy
    x_pos = range(len(param_names))
    ax1.plot(x_pos, test_accuracies, 'b-o', label='Test Accuracy', linewidth=2, markersize=6)
    ax1.plot(x_pos, train_accuracies, 'r-o', label='Train Accuracy', linewidth=2, markersize=6)
    ax1.axvline(x=optimal_idx, color='green', linestyle='--', alpha=0.7, label=f'Optimal')
    ax1.set_xlabel('Parameter Combinations', fontsize=12, fontweight='bold', color='black')
    ax1.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold', color='black')
    ax1.set_title('Gradient Boosting Accuracy vs Parameters', fontsize=14, fontweight='bold', color='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{i+1}" for i in range(len(param_names))], rotation=45, color='black')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.tick_params(colors='black')
    
    # Plot 2: R² Score
    ax2.plot(x_pos, test_r2_scores, 'b-o', label='Test R²', linewidth=2, markersize=6)
    ax2.plot(x_pos, train_r2_scores, 'r-o', label='Train R²', linewidth=2, markersize=6)
    ax2.axvline(x=optimal_idx, color='green', linestyle='--', alpha=0.7, label=f'Optimal')
    ax2.set_xlabel('Parameter Combinations', fontsize=12, fontweight='bold', color='black')
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold', color='black')
    ax2.set_title('Gradient Boosting R² Score vs Parameters', fontsize=14, fontweight='bold', color='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{i+1}" for i in range(len(param_names))], rotation=45, color='black')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.tick_params(colors='black')
    
    plt.tight_layout()
    
    # Save the tuning plot
    tuning_plot_filename = os.path.join(plots_folder, 'gradient_boosting_tuning.png')
    plt.savefig(tuning_plot_filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved tuning plot to: {tuning_plot_filename}")
    
    plt.show()
    
    # Return results
    tuning_results = {
        'param_combinations': param_combinations,
        'param_names': param_names,
        'test_accuracies': test_accuracies,
        'train_accuracies': train_accuracies,
        'test_r2_scores': test_r2_scores,
        'train_r2_scores': train_r2_scores,
        'optimal_params': optimal_params,
        'optimal_accuracy': optimal_accuracy
    }
    
    return optimal_params, tuning_results

#%% STEP 3.2: Apply Gradient Boosting imputation with optimal hyperparameters
def gradient_boosting_impute_habit(df, cv_folds=5, tune_hyperparams=True):
    """
    Use Gradient Boosting to impute missing 'habit' values based on similar observations
    Includes cross-validation, evaluation metrics, and optional hyperparameter tuning
    """
    df_copy = df.copy()
    
    # Separate complete and missing data
    habit_missing = df_copy['habit'].isnull()
    df_complete = df_copy[~habit_missing].copy()
    df_missing = df_copy[habit_missing].copy()
    
    if len(df_missing) == 0:
        return df_copy
    
    print(f"  - Found {len(df_missing)} missing 'habit' values")
    print(f"  - Using {len(df_complete)} complete observations for Gradient Boosting")
    
    # Select features for Gradient Boosting (numerical features that indicate behavior patterns)
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
    print(f"  - Checking for NaN values in features:")
    for i, col in enumerate(feature_cols):
        nan_count = np.isnan(X_complete[:, i]).sum()
        if nan_count > 0:
            print(f"    {col}: {nan_count} NaN values")
    
    # Only impute if there are actual NaN values
    if np.isnan(X_complete).any():
        print(f"  - Found NaN values in features. Applying imputation...")
        imputer = SimpleImputer(strategy='median')
        X_complete = imputer.fit_transform(X_complete)
        X_missing = imputer.transform(X_missing)
    else:
        print(f"  - No NaN values found in features. Skipping imputation.")
        # Convert to float if needed
        X_complete = X_complete.astype(float)
        X_missing = X_missing.astype(float)
    
    # Normalize/Standardize features for better Gradient Boosting performance
    print(f"  - Normalizing features for better Gradient Boosting performance...")
    scaler = StandardScaler()
    X_complete_scaled = scaler.fit_transform(X_complete)
    X_missing_scaled = scaler.transform(X_missing)
    
    print(f"  - Feature scaling completed:")
    print(f"    Original ranges: {X_complete.min(axis=0)} to {X_complete.max(axis=0)}")
    print(f"    Scaled ranges: {X_complete_scaled.min(axis=0)} to {X_complete_scaled.max(axis=0)}")
    
    # Encode habit values for classification
    le = LabelEncoder()
    y_complete = le.fit_transform(df_complete['habit'])
    
    # Tune hyperparameters if requested
    if tune_hyperparams:
        optimal_params, tuning_results = tune_gradient_boosting(X_complete_scaled, y_complete, cv_folds=cv_folds)
        print(f"  - Using tuned parameters: {optimal_params}")
    else:
        optimal_params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
        tuning_results = None
        print(f"  - Using default parameters: {optimal_params}")
    
    # Create Gradient Boosting classifier with optimal/tuned parameters
    gb = GradientBoostingClassifier(
        n_estimators=optimal_params['n_estimators'],
        max_depth=optimal_params['max_depth'],
        learning_rate=optimal_params['learning_rate'],
        random_state=42
    )
    
    # Perform cross-validation on complete data using scaled features
    print(f"\n  - Performing {cv_folds}-fold cross-validation on complete data...")
    cv_scores = cross_validate(gb, X_complete_scaled, y_complete, 
                              cv=cv_folds, 
                              scoring=['accuracy', 'r2'],
                              return_train_score=True)
    
    # Print cross-validation results
    print(f"  - Cross-validation results:")
    print(f"    Test Accuracy: {cv_scores['test_accuracy'].mean():.4f} (+/- {cv_scores['test_accuracy'].std() * 2:.4f})")
    print(f"    Test R² Score: {cv_scores['test_r2'].mean():.4f} (+/- {cv_scores['test_r2'].std() * 2:.4f})")
    print(f"    Train Accuracy: {cv_scores['train_accuracy'].mean():.4f} (+/- {cv_scores['train_accuracy'].std() * 2:.4f})")
    print(f"    Train R² Score: {cv_scores['train_r2'].mean():.4f} (+/- {cv_scores['train_r2'].std() * 2:.4f})")
    
    # Train final model on complete data using scaled features
    gb.fit(X_complete_scaled, y_complete)
    
    # Predict missing values using scaled features
    y_pred = gb.predict(X_missing_scaled)
    predicted_habits = le.inverse_transform(y_pred)
    
    # Fill missing values
    df_copy.loc[habit_missing, 'habit'] = predicted_habits
    
    # Print imputation results
    print(f"  - Imputed habit distribution:")
    for habit, count in pd.Series(predicted_habits).value_counts().items():
        print(f"    {habit}: {count}")
    
    return df_copy, cv_scores, tuning_results

#%% STEP 3.3: Find optimal hyperparameters using cross-validation
print("\n[STEP 3.3] Finding optimal hyperparameters for Gradient Boosting imputation...")

# Prepare data for tuning
habit_missing = dataset1_for_knn['habit'].isnull()
df_complete = dataset1_for_knn[~habit_missing].copy()

# Select features for KNN
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

# Check for NaN values in features
if np.isnan(X_complete).any():
    print(f"  - Found NaN values in features. Applying imputation...")
    imputer = SimpleImputer(strategy='median')
    X_complete = imputer.fit_transform(X_complete)
else:
    print(f"  - No NaN values found in features. Skipping imputation.")
    X_complete = X_complete.astype(float)

# Normalize features for Gradient Boosting
print(f"  - Normalizing features for Gradient Boosting...")
scaler = StandardScaler()
X_complete_scaled = scaler.fit_transform(X_complete)

# Encode habit values for classification
le = LabelEncoder()
y_complete = le.fit_transform(df_complete['habit'])

# Find optimal hyperparameters using scaled features
optimal_params, tuning_results = tune_gradient_boosting(X_complete_scaled, y_complete, cv_folds=5)

print(f"\n✓ Optimal parameters found: {optimal_params}")
print(f"✓ Best test accuracy: {tuning_results['optimal_accuracy']:.4f}")

#%% STEP 3.4: Apply Gradient Boosting imputation with optimal hyperparameters
print("\n[STEP 3.4] Applying Gradient Boosting imputation with optimal hyperparameters...")

# Apply Gradient Boosting imputation using the cleaned dataset (dataset1_for_knn) with optimal parameters
dataset1_imputed, cv_scores, _ = gradient_boosting_impute_habit(dataset1_for_knn, cv_folds=5, tune_hyperparams=False)

# Verify imputation
print("\nOriginal vs Imputed 'habit' distribution:")
original_dist = dataset1['habit'].value_counts()
imputed_dist = dataset1_imputed['habit'].value_counts()
comparison = pd.DataFrame({
    'Original': original_dist,
    'After_Imputation': imputed_dist
})
print(comparison)

# Create detailed evaluation report
print("\n=== GRADIENT BOOSTING IMPUTATION EVALUATION REPORT ===")
print(f"Cross-validation folds: 5")
print(f"Optimal parameters: {optimal_params}")
print(f"Optimal test accuracy: {tuning_results['optimal_accuracy']:.4f}")

# Show tuning summary
print(f"\nHyperparameter Tuning Summary:")
print(f"  Tested parameter combinations: {len(tuning_results['param_combinations'])}")
print(f"  Best parameters found: {optimal_params}")
print(f"  Best test accuracy: {tuning_results['optimal_accuracy']:.4f}")

# Show performance at different parameter combinations
print(f"\nPerformance at different parameter combinations:")
for i, (params, test_acc, train_acc) in enumerate(zip(tuning_results['param_combinations'], 
                                                      tuning_results['test_accuracies'], 
                                                      tuning_results['train_accuracies'])):
    print(f"  {i+1}: n={params['n_estimators']}, d={params['max_depth']}, lr={params['learning_rate']}")
    print(f"    Test Acc={test_acc:.4f}, Train Acc={train_acc:.4f}")

print(f"\nFinal Model Performance Metrics:")
print(f"  Test Accuracy: {cv_scores['test_accuracy'].mean():.4f} (+/- {cv_scores['test_accuracy'].std() * 2:.4f})")
print(f"  Test R² Score: {cv_scores['test_r2'].mean():.4f} (+/- {cv_scores['test_r2'].std() * 2:.4f})")
print(f"  Train Accuracy: {cv_scores['train_accuracy'].mean():.4f} (+/- {cv_scores['train_accuracy'].std() * 2:.4f})")
print(f"  Train R² Score: {cv_scores['train_r2'].mean():.4f} (+/- {cv_scores['train_r2'].std() * 2:.4f})")

# Check for overfitting
accuracy_diff = cv_scores['train_accuracy'].mean() - cv_scores['test_accuracy'].mean()
r2_diff = cv_scores['train_r2'].mean() - cv_scores['test_r2'].mean()

print(f"\nOverfitting Analysis:")
print(f"  Accuracy difference (Train - Test): {accuracy_diff:.4f}")
print(f"  R² difference (Train - Test): {r2_diff:.4f}")

if accuracy_diff > 0.1:
    print(f"  ⚠️  Potential overfitting detected (accuracy difference > 0.1)")
else:
    print(f"  ✅ No significant overfitting detected")

if r2_diff > 0.1:
    print(f"  ⚠️  Potential overfitting detected (R² difference > 0.1)")
else:
    print(f"  ✅ No significant overfitting detected")

# Additional tuning insights
print(f"\nTuning Insights:")
best_idx = np.argmax(tuning_results['test_accuracies'])
worst_idx = np.argmin(tuning_results['test_accuracies'])
worst_acc = min(tuning_results['test_accuracies'])

print(f"  Best parameters: {optimal_params} (Accuracy: {tuning_results['optimal_accuracy']:.4f})")
print(f"  Worst parameters: {tuning_results['param_combinations'][worst_idx]} (Accuracy: {worst_acc:.4f})")
print(f"  Improvement: {tuning_results['optimal_accuracy'] - worst_acc:.4f}")

# Check if tuning helped with overfitting
best_train_acc = tuning_results['train_accuracies'][best_idx]
best_test_acc = tuning_results['test_accuracies'][best_idx]
best_overfitting = best_train_acc - best_test_acc

if best_overfitting < 0.1:
    print(f"  ✅ Tuning helped reduce overfitting (difference: {best_overfitting:.4f})")
else:
    print(f"  ⚠️  Overfitting still present after tuning (difference: {best_overfitting:.4f})")

#%% STEP 3.5: Data Quality and Distribution Analysis
print("\n[STEP 3.5] Analyzing data quality and distributions after imputation...")

# Distribution analysis with normal fit
plt.figure(figsize=(15, 10))

# Plot 1: Bat landing to food distribution
plt.subplot(2, 3, 1)
sns.distplot(dataset1_imputed['bat_landing_to_food'].dropna(), fit=norm, kde=True, color='blue')
plt.title("Bat Landing to Food Distribution")
plt.xlabel("Time (seconds)")
(mu, sigma) = norm.fit(dataset1_imputed['bat_landing_to_food'].dropna())
plt.legend(['Normal dist. ($\mu$={:.2f}, $\sigma$={:.2f})'.format(mu, sigma), 'Actual data'])

# Plot 2: Q-Q plot for normality check
plt.subplot(2, 3, 2)
stats.probplot(dataset1_imputed['bat_landing_to_food'].dropna(), plot=plt)
plt.title("Q-Q Plot: Bat Landing to Food")

# Plot 3: Distribution by habit type
plt.subplot(2, 3, 3)
for habit in dataset1_imputed['habit'].unique():
    subset = dataset1_imputed[dataset1_imputed['habit'] == habit]['bat_landing_to_food']
    sns.distplot(subset.dropna(), hist=False, kde=True, label=habit)
plt.title("Landing Time Distribution by Habit")
plt.xlabel("Time (seconds)")
plt.legend()

# Plot 4: Seconds after rat arrival distribution
plt.subplot(2, 3, 4)
sns.distplot(dataset1_imputed['seconds_after_rat_arrival'].dropna(), fit=norm, kde=True, color='red')
plt.title("Seconds After Rat Arrival Distribution")
plt.xlabel("Time (seconds)")
plt.xlim(0, 500)  # Focus on first 500 seconds

# Plot 5: Risk vs Reward distribution
plt.subplot(2, 3, 5)
risk_reward = dataset1_imputed.groupby(['risk', 'reward']).size().unstack()
risk_reward.plot(kind='bar', stacked=True, color=['#FF6B6B', '#4ECDC4'])
plt.title("Risk vs Reward Distribution")
plt.xlabel("Risk (0=No, 1=Yes)")
plt.ylabel("Count")
plt.legend(title='Reward', labels=['No Reward', 'Reward'])

# Plot 6: Boxplot for outlier detection
plt.subplot(2, 3, 6)
key_features = ['bat_landing_to_food', 'seconds_after_rat_arrival', 'hours_after_sunset']
dataset1_imputed[key_features].boxplot()
plt.title("Outlier Detection: Key Features")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'data_quality_analysis.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved data quality analysis to: {os.path.join(plots_folder, 'data_quality_analysis.png')}")
plt.show()

# Check skewness
print("\nSkewness Analysis:")
numerical_features = ['bat_landing_to_food', 'seconds_after_rat_arrival', 
                     'risk', 'reward', 'hours_after_sunset']
for feature in numerical_features:
    skewness = dataset1_imputed[feature].skew()
    print(f"  {feature}: {skewness:.3f}")
    if abs(skewness) > 1:
        print(f"    ⚠️ High skewness - consider transformation")

#%% STEP 3.6: Correlation Analysis with Clustering
print("\n[STEP 3.6] Correlation analysis with clustering...")

# Select features for correlation analysis
corr_features = ['bat_landing_to_food', 'seconds_after_rat_arrival', 
                 'risk', 'reward', 'hours_after_sunset', 'month', 'season']

# Create correlation matrix
corr_matrix = dataset1_imputed[corr_features].corr()

# Full correlation clustermap
plt.figure(figsize=(10, 8))
sns.clustermap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
               center=0, square=True, linewidths=1, cbar_kws={"shrink": .8})
plt.title("Feature Correlations with Clustering", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'correlation_clustermap_full.png'), dpi=300, bbox_inches='tight')
plt.show()

# Filter high correlations with bat_landing_to_food (our main vigilance indicator)
threshold = 0.2
high_corr_features = corr_matrix['bat_landing_to_food'].abs()
high_corr_features = high_corr_features[high_corr_features > threshold].index.tolist()

if len(high_corr_features) > 1:  # Only plot if there are correlations above threshold
    plt.figure(figsize=(8, 6))
    sns.heatmap(dataset1_imputed[high_corr_features].corr(), annot=True, fmt=".2f", 
                cmap='coolwarm', center=0, square=True, linewidths=1)
    plt.title(f"Features with Correlation > {threshold} to Landing Time")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'high_correlation_landing_time.png'), dpi=300, bbox_inches='tight')
    plt.show()

print(f"\nFeatures highly correlated with bat_landing_to_food (>{threshold}):")
for feature in high_corr_features:
    if feature != 'bat_landing_to_food':
        corr_value = corr_matrix.loc['bat_landing_to_food', feature]
        print(f"  {feature}: {corr_value:.3f}")

#%% STEP 3.7: Pairplot for Key Relationships
print("\n[STEP 3.7] Creating pairplot for key relationships...")

# Select key features for pairplot
pairplot_features = ['bat_landing_to_food', 'seconds_after_rat_arrival', 
                     'risk', 'reward', 'habit']

# Create subset for pairplot (sample if too large)
if len(dataset1_imputed) > 1000:
    pairplot_data = dataset1_imputed[pairplot_features].sample(n=1000, random_state=42)
    print("  Using sample of 1000 observations for pairplot")
else:
    pairplot_data = dataset1_imputed[pairplot_features]

# Create pairplot
plt.figure(figsize=(12, 10))
g = sns.pairplot(pairplot_data, hue='habit', diag_kind="kde", 
                 palette={'fast': '#FF6B6B', 'bat': '#4ECDC4', 
                         'rat': '#45B7D1', 'both': '#96CEB4'},
                 plot_kws={'alpha': 0.6})
g.fig.suptitle("Pairwise Relationships by Habit Type", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'pairplot_key_features.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved pairplot to: {os.path.join(plots_folder, 'pairplot_key_features.png')}")
plt.show()

print("\n✓ Data quality and correlation analysis completed!")
print("  Key insights will help understand data patterns before main analysis")

#%% STEP 4: Create vigilance indicators (no CSV creation - just feature engineering)
print("\n[STEP 4] Creating vigilance and predation risk indicators...")

# 1. Primary vigilance indicator: Time to approach food
# Higher values = more vigilant/cautious behavior
dataset1_imputed['vigilance_score'] = dataset1_imputed['bat_landing_to_food']

# 2. Categorize vigilance levels
vigilance_thresholds = dataset1_imputed['bat_landing_to_food'].quantile([0.25, 0.5, 0.75])
dataset1_imputed['vigilance_level'] = pd.cut(
    dataset1_imputed['bat_landing_to_food'],
    bins=[0, vigilance_thresholds[0.25], vigilance_thresholds[0.5], 
          vigilance_thresholds[0.75], float('inf')],
    labels=['low', 'medium', 'high', 'very_high']
)

# 3. Binary high vigilance indicator
dataset1_imputed['high_vigilance'] = (
    dataset1_imputed['bat_landing_to_food'] > vigilance_thresholds[0.75]
).astype(int)

# 4. Avoidance behavior indicator
# Bats showing low risk-taking AND high vigilance suggest predator avoidance
dataset1_imputed['avoidance_behavior'] = (
    (dataset1_imputed['risk'] == 0) & 
    (dataset1_imputed['high_vigilance'] == 1)
).astype(int)


# 5. Calculate rat presence duration
dataset1_imputed['rat_presence_duration'] = (
    dataset1_imputed['rat_period_end'] - dataset1_imputed['rat_period_start']
).dt.total_seconds()

# 6. Immediate rat threat indicator
dataset1_imputed['immediate_rat_threat'] = (
    dataset1_imputed['seconds_after_rat_arrival'] < 30
).astype(int)

# 7. Create interaction features
dataset1_imputed['vigilance_x_rat_presence'] = (
    dataset1_imputed['vigilance_score'] * 
    (dataset1_imputed['rat_presence_duration'] / 60)  # Convert to minutes
)

print("Vigilance indicators created:")
print(f"  - High vigilance cases: {dataset1_imputed['high_vigilance'].sum()} ({dataset1_imputed['high_vigilance'].mean()*100:.1f}%)")
print(f"  - Avoidance behavior cases: {dataset1_imputed['avoidance_behavior'].sum()} ({dataset1_imputed['avoidance_behavior'].mean()*100:.1f}%)")

#%% STEP 5.1: Prepare Dataset 2 for merging
print("\n[STEP 5.1] Merging with Dataset 2 for environmental context...")

# Create time-based keys for merging
dataset1_imputed['merge_date'] = dataset1_imputed['start_time'].dt.date
dataset1_imputed['merge_hour'] = dataset1_imputed['start_time'].dt.hour

dataset2['merge_date'] = dataset2['time'].dt.date  
dataset2['merge_hour'] = dataset2['time'].dt.hour

# Aggregate Dataset 2 by date and hour
hourly_context = dataset2.groupby(['merge_date', 'merge_hour']).agg({
    'bat_landing_number': 'sum',
    'food_availability': 'mean',
    'rat_minutes': 'sum',
    'rat_arrival_number': 'sum'
}).reset_index()

# Rename columns to avoid confusion
hourly_context.rename(columns={
    'bat_landing_number': 'total_bat_landings_hour',
    'food_availability': 'avg_food_availability',
    'rat_minutes': 'total_rat_minutes_hour',
    'rat_arrival_number': 'total_rat_arrivals_hour'
}, inplace=True)

#%% STEP 5.2: Merge datasets
# Merge with Dataset 1
dataset1_final = pd.merge(
    dataset1_imputed,
    hourly_context,
    on=['merge_date', 'merge_hour'],
    how='left'
)

# Fill any missing context values with 0 (assumes no activity if not recorded)
context_cols = ['total_bat_landings_hour', 'avg_food_availability', 
                'total_rat_minutes_hour', 'total_rat_arrivals_hour']
dataset1_final[context_cols] = dataset1_final[context_cols].fillna(0)

print(f"Successfully merged. Final dataset shape: {dataset1_final.shape}")

#%% STEP 6.1: Create final analysis features
print("\n[STEP 6.1] Creating final feature set for predation risk analysis...")

# Competition intensity indicator
dataset1_final['high_competition'] = (
    dataset1_final['total_bat_landings_hour'] > 
    dataset1_final['total_bat_landings_hour'].median()
).astype(int)

# Rat activity level categories
dataset1_final['rat_activity_level'] = pd.cut(
    dataset1_final['total_rat_minutes_hour'],
    bins=[-1, 0, 10, 30, float('inf')],
    labels=['none', 'low', 'medium', 'high']
)

#%% STEP 6.2: Analyze vigilance patterns
# Create summary statistics
print("\nPredation Risk Analysis Summary:")
print(f"Average vigilance score: {dataset1_final['vigilance_score'].mean():.2f}")
print(f"Vigilance when rats present vs absent:")

# Compare vigilance with and without rats
with_rats = dataset1_final[dataset1_final['total_rat_arrivals_hour'] > 0]
without_rats = dataset1_final[dataset1_final['total_rat_arrivals_hour'] == 0]

print(f"  - With rats: {with_rats['vigilance_score'].mean():.2f} (n={len(with_rats)})")
print(f"  - Without rats: {without_rats['vigilance_score'].mean():.2f} (n={len(without_rats)})")
print(f"  - Difference: {with_rats['vigilance_score'].mean() - without_rats['vigilance_score'].mean():.2f}")

#%% STEP 6.3: Visualize vigilance patterns
# Set white background style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Create visualization of vigilance levels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.patch.set_facecolor('white')

# Plot 1: Boxplot
dataset1_final.boxplot(column='vigilance_score', 
                       by='rat_activity_level', 
                       ax=ax1)
ax1.set_title('Vigilance Score by Rat Activity Level', color='black', fontweight='bold')
ax1.set_ylabel('Vigilance Score (seconds)', color='black')
ax1.tick_params(colors='black')

# Plot 2: Bar chart
vigilance_comparison = pd.DataFrame({
    'With Rats': [with_rats['vigilance_score'].mean()],
    'Without Rats': [without_rats['vigilance_score'].mean()]
})
vigilance_comparison.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4'])
ax2.set_title('Average Vigilance: Rats Present vs Absent', color='black', fontweight='bold')
ax2.set_ylabel('Average Vigilance Score', color='black')
ax2.set_xticks([])
ax2.legend()
ax2.tick_params(colors='black')

plt.tight_layout()

# Save the vigilance patterns plot
vigilance_plot_filename = os.path.join(plots_folder, 'vigilance_patterns.png')
plt.savefig(vigilance_plot_filename, dpi=300, bbox_inches='tight')
print(f"✓ Saved vigilance patterns plot to: {vigilance_plot_filename}")

plt.show()

#%% STEP 7.1: Save preprocessed data
print("\n[STEP 7.1] Saving preprocessed data...")

# Define folder paths (already created in Step 1)
datasets_folder = 'datasets'
produced_folder = os.path.join(datasets_folder, 'produced')

# Select key columns for analysis
analysis_columns = [
    # Identifiers
    'start_time', 'month', 'season',
    
    # Vigilance indicators
    'vigilance_score', 'vigilance_level', 'high_vigilance', 'avoidance_behavior',
    
    # Bat behavior
    'bat_landing_to_food', 'risk', 'reward', 'habit',
    
    # Rat presence
    'seconds_after_rat_arrival', 'rat_presence_duration', 'immediate_rat_threat',
    
    # Environmental context
    'hours_after_sunset', 'avg_food_availability', 
    'total_rat_minutes_hour', 'total_rat_arrivals_hour',
    'total_bat_landings_hour', 'rat_activity_level',
    
    # Competition and interaction
    'high_competition', 'vigilance_x_rat_presence'
]

# Create final dataset
final_dataset = dataset1_final[analysis_columns].copy()

# Save files to produced folder
clean_filepath = os.path.join(produced_folder, 'investigation_a_clean.csv')
full_filepath = os.path.join(produced_folder, 'investigation_a_full.csv')

final_dataset.to_csv(clean_filepath, index=False)
dataset1_final.to_csv(full_filepath, index=False)

print(f"✓ Saved clean dataset to: {clean_filepath}")
print(f"✓ Saved full dataset to: {full_filepath}")

#%% STEP 7.2: Create summary visualization
print("\n[STEP 7.2] Creating summary visualization...")

# Create summary data for visualization
summary_data = {
    'Metric': ['High Vigilance Rate', 'Avoidance Behavior Rate', 'Risk-taking Rate', 'Reward Success Rate'],
    'Value': [
        final_dataset['high_vigilance'].mean() * 100,
        final_dataset['avoidance_behavior'].mean() * 100,
        final_dataset['risk'].mean() * 100,
        final_dataset['reward'].mean() * 100
    ]
}

# Set white background style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Create summary visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('white')

# Plot 1: Key Metrics Bar Chart
metrics_df = pd.DataFrame(summary_data)
bars = ax1.bar(metrics_df['Metric'], metrics_df['Value'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.7)
ax1.set_title('Key Behavioral Metrics (%)', fontsize=14, fontweight='bold', color='black')
ax1.set_ylabel('Percentage (%)', color='black')
ax1.set_ylim(0, 60)
ax1.tick_params(colors='black')
ax1.set_xticklabels(metrics_df['Metric'], rotation=45, ha='right', color='black')

# Add value labels on bars
for bar, value in zip(bars, metrics_df['Value']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', color='black')

# Plot 2: Vigilance Comparison
vigilance_comparison = pd.DataFrame({
    'Condition': ['With Rats', 'Without Rats'],
    'Vigilance Score': [with_rats['vigilance_score'].mean(), without_rats['vigilance_score'].mean()]
})

bars = ax2.bar(vigilance_comparison['Condition'], vigilance_comparison['Vigilance Score'],
               color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
ax2.set_title('Average Vigilance Score by Rat Presence', fontsize=14, fontweight='bold', color='black')
ax2.set_ylabel('Vigilance Score (seconds)', color='black')
ax2.tick_params(colors='black')
ax2.set_xticklabels(vigilance_comparison['Condition'], color='black')

# Add value labels on bars
for bar, value in zip(bars, vigilance_comparison['Vigilance Score']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{value:.2f}', ha='center', va='bottom', fontweight='bold', color='black')

plt.tight_layout()

# Save the summary visualization
summary_plot_filename = os.path.join(plots_folder, 'summary_visualization.png')
plt.savefig(summary_plot_filename, dpi=300, bbox_inches='tight')
print(f"✓ Saved summary visualization to: {summary_plot_filename}")

plt.show()

print("\n✓ Data preprocessing completed successfully!")
print("\nFiles created:")
print("  1. datasets/produced/investigation_a_clean.csv - Main analysis dataset")
print("  2. datasets/produced/investigation_a_full.csv - Complete dataset with all features")
print("  3. plots/vigilance_patterns.png - Vigilance analysis plots")
print("  4. plots/summary_visualization.png - Summary visualization")
print("  5. plots/gradient_boosting_tuning.png - Model tuning results")
print("  6. plots/habit_category_simplification.png - Habit category analysis")
print("  7. plots/habit_rat_timing_analysis.png - Habit vs rat timing analysis")

print("\nNext steps for analysis:")
print("  - Statistical tests to compare vigilance levels with/without rats")
print("  - Correlation analysis between rat presence and vigilance indicators")
print("  - Machine learning models to predict vigilance based on environmental factors")
print("  - Visualization of vigilance patterns across different conditions")

#%% STEP 7.3: Display final summary
print("\n=== FINAL SUMMARY ===")
print(f"Total Observations: {len(final_dataset)}")
print(f"High Vigilance Rate: {final_dataset['high_vigilance'].mean()*100:.1f}%")
print(f"Avoidance Behavior Rate: {final_dataset['avoidance_behavior'].mean()*100:.1f}%")
print(f"Average Vigilance (with rats): {with_rats['vigilance_score'].mean():.2f}")
print(f"Average Vigilance (no rats): {without_rats['vigilance_score'].mean():.2f}")
print(f"Risk-taking Rate: {final_dataset['risk'].mean()*100:.1f}%")
print(f"Reward Success Rate: {final_dataset['reward'].mean()*100:.1f}%")

# Show correlations with vigilance
print("\nTop correlations with vigilance score:")
correlations = final_dataset.select_dtypes(include=[np.number]).corr()['vigilance_score'].sort_values(ascending=False)
print(correlations.head(10))

#%% STEP 8: Main Statistical Test
print("\n[STEP 8] Testing vigilance difference with/without rats...")

# Compare vigilance with and without rats
with_rats = final_dataset[final_dataset['total_rat_arrivals_hour'] > 0]
without_rats = final_dataset[final_dataset['total_rat_arrivals_hour'] == 0]

# T-test
statistic, p_value = stats.ttest_ind(
    with_rats['vigilance_score'].dropna(), 
    without_rats['vigilance_score'].dropna()
)

# Calculate effect size
mean_diff = with_rats['vigilance_score'].mean() - without_rats['vigilance_score'].mean()
percent_increase = (mean_diff / without_rats['vigilance_score'].mean()) * 100

# Cohen's d
pooled_std = np.sqrt((with_rats['vigilance_score'].std()**2 + without_rats['vigilance_score'].std()**2) / 2)
cohens_d = mean_diff / pooled_std

print(f"\nResults:")
print(f"  Vigilance with rats: {with_rats['vigilance_score'].mean():.2f}")
print(f"  Vigilance without rats: {without_rats['vigilance_score'].mean():.2f}")
print(f"  Difference: {mean_diff:.2f} seconds ({percent_increase:.1f}% increase)")
print(f"  P-value: {p_value:.4f}")
print(f"  Cohen's d: {cohens_d:.3f}")
print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

#%% STEP 8.1: Key Correlations
print("\n[STEP 8.1] Checking correlations...")

# Calculate correlations with vigilance
correlations = final_dataset[[
    'vigilance_score', 
    'seconds_after_rat_arrival', 
    'total_rat_minutes_hour',
    'immediate_rat_threat',
    'rat_presence_duration'
]].corr()['vigilance_score'].sort_values(ascending=False)

print("\nKey correlations with vigilance:")
for var, corr in correlations.items():
    if var != 'vigilance_score':
        print(f"  {var}: {corr:.3f}")

#%% STEP 8.2: Avoidance Behavior Analysis
print("\n[STEP 8.2] Avoidance behavior analysis...")

avoidance_with = with_rats['avoidance_behavior'].mean() * 100
avoidance_without = without_rats['avoidance_behavior'].mean() * 100

print(f"  Avoidance with rats: {avoidance_with:.1f}%")
print(f"  Avoidance without rats: {avoidance_without:.1f}%")
print(f"  Difference: {avoidance_with - avoidance_without:.1f} percentage points")

# Chi-square test for avoidance behavior
contingency_table = pd.crosstab(
    final_dataset['total_rat_arrivals_hour'] > 0,
    final_dataset['avoidance_behavior']
)
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
print(f"  Chi-square test p-value: {p_chi:.4f}")

#%% STEP 8.3: Behavior by Rat Activity Level
print("\n[STEP 8.3] Analyzing behavior by rat activity level...")

# Group by rat activity level
activity_groups = final_dataset.groupby('rat_activity_level').agg({
    'vigilance_score': ['mean', 'std', 'count'],
    'avoidance_behavior': 'mean',
    'high_vigilance': 'mean'
}).round(2)

print("\nBehavior by rat activity level:")
print(activity_groups)

#%% STEP 8.4: Final Visualization
print("\n[STEP 8.4] Creating final evidence plot...")

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
fig.patch.set_facecolor('white')

# Plot 1: Vigilance comparison
vigilance_means = [without_rats['vigilance_score'].mean(), with_rats['vigilance_score'].mean()]
vigilance_stds = [without_rats['vigilance_score'].std(), with_rats['vigilance_score'].std()]
bars = ax1.bar(['No Rats', 'Rats Present'], vigilance_means, yerr=vigilance_stds, 
                color=['#4ECDC4', '#FF6B6B'], capsize=5)
ax1.set_ylabel('Vigilance Score (seconds)', fontweight='bold')
ax1.set_title('Average Vigilance by Rat Presence', fontweight='bold')
for bar, val in zip(bars, vigilance_means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}', ha='center', fontweight='bold')

# Plot 2: Vigilance over time since rat arrival
time_bins = [0, 30, 60, 120, 300, 600]
final_dataset['time_bin'] = pd.cut(final_dataset['seconds_after_rat_arrival'], 
                                   bins=time_bins, labels=['0-30s', '30-60s', '60-120s', '120-300s', '300-600s'])
time_vigilance = final_dataset.groupby('time_bin')['vigilance_score'].mean()
ax2.plot(time_vigilance.index, time_vigilance.values, 'o-', color='darkblue', linewidth=2, markersize=8)
ax2.set_xlabel('Time Since Rat Arrival', fontweight='bold')
ax2.set_ylabel('Average Vigilance Score', fontweight='bold')
ax2.set_title('Vigilance Response Over Time', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Avoidance behavior comparison
avoidance_data = [avoidance_without, avoidance_with]
bars = ax3.bar(['No Rats', 'Rats Present'], avoidance_data, color=['#4ECDC4', '#FF6B6B'])
ax3.set_ylabel('Avoidance Behavior (%)', fontweight='bold')
ax3.set_title('Avoidance Behavior by Rat Presence', fontweight='bold')
for bar, val in zip(bars, avoidance_data):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.1f}%', ha='center', fontweight='bold')

# Plot 4: Summary text
ax4.axis('off')
summary_text = f"""EVIDENCE SUMMARY

1. Vigilance Test:
   • {percent_increase:.1f}% higher with rats
   • P-value: {p_value:.4f}
   • Cohen's d: {cohens_d:.3f}
   • {'✓ Significant' if p_value < 0.05 else '✗ Not significant'}

2. Avoidance Behavior:
   • {avoidance_with - avoidance_without:.1f}% more with rats
   • Chi-square p: {p_chi:.4f}

3. Correlations:
   • Rat minutes: r = {correlations['total_rat_minutes_hour']:.3f}
   • Immediate threat: r = {correlations['immediate_rat_threat']:.3f}

CONCLUSION:
{'✓ Bats likely perceive rats as predators' if p_value < 0.05 and percent_increase > 10 else '✗ Insufficient evidence for predator perception'}
"""
ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, 
         fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
evidence_plot = os.path.join(plots_folder, 'predator_perception_evidence.png')
plt.savefig(evidence_plot, dpi=300, bbox_inches='tight')
print(f"✓ Saved evidence plot to: {evidence_plot}")
plt.show()

#%% STEP 8.5: Additional Analysis - Habit Patterns
print("\n[STEP 8.5] Analyzing habit patterns...")

# Compare habits when rats are present
habit_analysis = final_dataset.groupby(['habit', final_dataset['total_rat_arrivals_hour'] > 0])['vigilance_score'].agg(['mean', 'count'])
print("\nVigilance by habit and rat presence:")
print(habit_analysis)

#%% Final Answer
print("\n=== INVESTIGATION A ANSWER ===")
print("=" * 50)

if p_value < 0.05 and percent_increase > 10:
    print("\n✓ YES - BATS PERCEIVE RATS AS PREDATORS\n")
    print(f"Supporting evidence:")
    print(f"  1. Vigilance increases {percent_increase:.1f}% with rats (p={p_value:.4f})")
    print(f"  2. Effect size is {['small', 'medium', 'large'][int(abs(cohens_d) > 0.5) + int(abs(cohens_d) > 0.8)]} (d={cohens_d:.3f})")
    print(f"  3. Avoidance behavior increases {avoidance_with - avoidance_without:.1f}%")
    print(f"  4. Positive correlation with rat presence (r={correlations['total_rat_minutes_hour']:.3f})")
else:
    print("\n✗ NO - INSUFFICIENT EVIDENCE FOR PREDATOR PERCEPTION\n")
    print(f"Findings:")
    print(f"  1. Vigilance difference not significant (p={p_value:.4f})")
    print(f"  2. Effect size: {cohens_d:.3f}")
    print(f"  3. Alternative explanations may be more appropriate")

print("\n✓ Analysis complete!")
print(f"\nFiles created in total:")
print(f"  - {os.path.join(produced_folder, 'investigation_a_clean.csv')}")
print(f"  - {os.path.join(produced_folder, 'investigation_a_full.csv')}")
print(f"  - {evidence_plot}")
print(f"  - Plus {len(os.listdir(plots_folder))-1} other visualizations in plots folder")
# %%
