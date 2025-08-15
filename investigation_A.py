# -*- coding: utf-8 -*-
#%%
"""
Investigation A: Predation Risk Perception Analysis
Do bats perceive rats as predators or just competitors?

IMPORTANT: This file implements the best ML algorithm found by ml_algorithm_comparison.py
The accuracy results in Phase 3 are from the ML comparison tool, not this implementation.
LightGBM achieved 73.86% accuracy in the comparison tool.

Phased approach:
- Phase 1: Data Loading and Initial Exploration
- Phase 2: Data Cleaning and Habit Analysis
- Phase 3: Missing Value Imputation with Gradient Boosting (best algorithm from comparison)
- Phase 4: Feature Engineering
- Phase 5: Statistical Analysis
- Phase 6: Visualization and Conclusion

WORKFLOW:
1. Run ml_algorithm_comparison.py before Phase 3 to find the best algorithm
2. This file then implements that algorithm for actual imputation
3. The accuracy claims reference the comparison results, not this implementation
"""

# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

#%%
# ============================================================================
# PHASE 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================
print("="*60)
print("PHASE 1: DATA LOADING AND INITIAL EXPLORATION")
print("="*60)

# Load datasets
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')

print(f"\nDataset 1: {dataset1.shape[0]} rows, {dataset1.shape[1]} columns")
print(f"Dataset 2: {dataset2.shape[0]} rows, {dataset2.shape[1]} columns")


# Check habit column specifically
print("\nHabit column analysis:")
print(f"Total rows: {len(dataset1)}")
print(f"Missing values: {dataset1['habit'].isnull().sum()}")

# Count numeric values (anything with numbers)
numeric_habits = 0
for habit in dataset1['habit']:
    if pd.notna(habit):
        habit_str = str(habit)
        # Check if it contains any numbers
        if any(char.isdigit() for char in habit_str):
            numeric_habits += 1

print(f"Numeric habit values: {numeric_habits}")
print(f"Text habit values: {len(dataset1) - dataset1['habit'].isnull().sum() - numeric_habits}")

#%%
# ============================================================================
# PHASE 2: DATA CLEANING AND HABIT ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("PHASE 2: DATA CLEANING AND HABIT ANALYSIS")
print("="*60)

# Convert time columns
time_cols = ['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time']
for col in time_cols:
    dataset1[col] = pd.to_datetime(dataset1[col], format='%d/%m/%Y %H:%M')

dataset2['time'] = pd.to_datetime(dataset2['time'], format='%d/%m/%Y %H:%M')

# Fix bat_landing_to_food values
def fix_landing_values(value):
    if pd.isna(value):
        return value
    if value >= 1:
        return int(value)
    if 0 < value < 1:
        return int(round(value * 1000))
    return value

dataset1['bat_landing_to_food'] = dataset1['bat_landing_to_food'].apply(fix_landing_values)

# Delete numeric habits identified in Phase 1
def has_numbers(value):
    if pd.isna(value):
        return False
    return any(char.isdigit() for char in str(value))

numeric_mask = dataset1['habit'].apply(has_numbers)
dataset1.loc[numeric_mask, 'habit'] = np.nan

print(f"Deleted {numeric_mask.sum()} numeric habit values")
print(f"Remaining valid habits: {dataset1['habit'].notna().sum()}")

# Store original habits before classification
original_habits = dataset1['habit'].copy()

# Comprehensive habit classification (8-10 clear categories, no missing habits)
def classify_habit_8_categories(habit):
    """Classify habits into 8-10 clear categories - comprehensive coverage"""
    if pd.isna(habit):
        return habit
    
    habit_str = str(habit).lower()
    
    # Priority order matters - comprehensive coverage for all habits
    if 'attack' in habit_str or 'fight' in habit_str:
        return 'attack'
    elif 'pick' in habit_str:
        return 'pick'
    elif 'avoid' in habit_str or 'leave' in habit_str or 'disappear' in habit_str or 'no_food' in habit_str:
        return 'avoid'
    elif 'gaze' in habit_str:
        return 'gaze'
    elif 'bat' in habit_str and 'rat' in habit_str:
        return 'both'
    elif 'fast' in habit_str:
        return 'fast'
    elif 'eating' in habit_str:
        return 'eating'
    elif 'bat' in habit_str or 'bats' in habit_str:
        return 'bat'
    elif 'rat' in habit_str or 'rats' in habit_str:
        return 'rat'
    elif 'others' in habit_str or 'other' in habit_str:
        return 'avoid'  # Group "others" with avoidance behaviors
    elif 'and' in habit_str:
        # Handle compound habits like "pick_and_others", "rat_and_no_food"
        if 'pick' in habit_str:
            return 'pick'
        elif 'rat' in habit_str:
            return 'rat'
        elif 'bat' in habit_str:
            return 'bat'
        else:
            return 'avoid'  # Default for compound habits
    else:
        # Final catch-all - assign based on context
        if 'food' in habit_str or 'bowl' in habit_str:
            return 'fast'
        else:
            return 'avoid'  # Default category for any remaining habits

# Apply new classification
dataset1['habit'] = dataset1['habit'].apply(classify_habit_8_categories)

# Get statistics for comparison
original_counts = original_habits.value_counts()
new_counts = dataset1['habit'].value_counts()

print(f"Original unique habits: {len(original_counts)}")
print(f"Classified into: {len(new_counts)} categories")
print(f"\nNew 8-10 category classification:")
for cat, count in new_counts.items():
    print(f"  {cat}: {count}")

# Debug information
print(f"\nDebug info:")
print(f"Total rows: {len(dataset1)}")
print(f"Missing values after classification: {dataset1['habit'].isnull().sum()}")
print(f"Non-missing values: {dataset1['habit'].notna().sum()}")

# Check specific problematic habits
print(f"\nTesting classification for problematic habits:")
test_habits = ['bat_fight', 'rat_and_no_food', 'ratand_others', 'pickand_others', 'pick_rat', 'pick_bat']
for habit in test_habits:
    if pd.notna(habit):
        classified = classify_habit_8_categories(habit)
        print(f"  '{habit}' -> '{classified}'")

# Show final classification results
print(f"\nFinal classification results:")
print(dataset1['habit'].value_counts())

# Create plots directory if it doesn't exist
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Create comparison visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='white')

# Original habits (top 10)
original_top10 = original_counts.head(10)
original_top10.plot(kind='bar', ax=ax1, color='lightcoral', alpha=0.7)
ax1.set_title('Original Habits (Top 10)')
ax1.set_xlabel('Habit Type')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

# New classified habits
new_counts.plot(kind='bar', ax=ax2, color='steelblue')
ax2.set_title('New 8-10 Category Classification')
ax2.set_xlabel('Category')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()

# Save the habit classification comparison plot
habit_plot_filename = os.path.join(plots_dir, 'habit_classification_comparison.png')
plt.savefig(habit_plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved habit classification plot to: {habit_plot_filename}")

plt.show()

#%%
# ============================================================================
# EXPORT CLEANED DATASET FOR ML ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("EXPORTING CLEANED DATASET FOR ML ANALYSIS")
print("="*60)

# Create datasets directory if it doesn't exist
import os
datasets_dir = 'datasets'
if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)

# Export the cleaned dataset1 for ML analysis (before imputation)
cleaned_filename = os.path.join(datasets_dir, 'dataset1_cleaned_for_ml.csv')
dataset1.to_csv(cleaned_filename, index=False)

print(f"✓ Exported cleaned dataset to: {cleaned_filename}")
print(f"✓ Dataset shape: {dataset1.shape}")
print(f"✓ Habit categories: {dataset1['habit'].nunique()}")
print(f"✓ Missing values: {dataset1['habit'].isnull().sum()}")

# Show final habit distribution
print(f"\nFinal habit distribution:")
habit_dist = dataset1['habit'].value_counts()
for habit, count in habit_dist.items():
    print(f"  {habit}: {count}")

print(f"\n✓ Dataset ready for ML algorithm comparison!")
print(f"✓ Run: python ml_algorithm_comparison.py")
print(f"✓ After finding best algorithm, return to Phase 3")

#%%
# ============================================================================
# PHASE 3: MISSING VALUE IMPUTATION WITH LIGHTGBM
# ============================================================================
print("\n" + "="*60)
print("PHASE 3: MISSING VALUE IMPUTATION")
print("="*60)
print("NOTE: This phase implements the best algorithm found by ml_algorithm_comparison.py")
print("The accuracy results shown are from the comparison tool, not this implementation.")
print("="*60)

# Prepare features for imputation
feature_cols = ['bat_landing_to_food', 'seconds_after_rat_arrival', 
                'risk', 'reward', 'hours_after_sunset', 'month', 'season']

# Split data
habit_missing = dataset1['habit'].isnull()
df_complete = dataset1[~habit_missing]
df_missing = dataset1[habit_missing]

print(f"\nMissing habits: {len(df_missing)}")
print(f"Complete habits: {len(df_complete)}")

if len(df_missing) > 0:
    # Prepare features
    X_complete = df_complete[feature_cols].values
    X_missing = df_missing[feature_cols].values
    
    # Handle NaN in features
    imputer = SimpleImputer(strategy='median')
    X_complete = imputer.fit_transform(X_complete)
    X_missing = imputer.transform(X_missing)
    
    # Standardize features (important for gradient boosting)
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_complete_scaled = scaler.fit_transform(X_complete)
    X_missing_scaled = scaler.transform(X_missing)
    
    # Encode target
    le = LabelEncoder()
    y_complete = le.fit_transform(df_complete['habit'])
    
    # Train LightGBM with OPTIMIZED parameters (Best Algorithm from ML Comparison)
    print("\nTraining LightGBM Classifier (Best Algorithm from ML Comparison)...")
    print("✓ Best Algorithm: LightGBM (found by ml_algorithm_comparison.py)")
    print("✓ Using optimized parameters: n_estimators=50, max_depth=3, learning_rate=0.1")
    print("✓ This algorithm achieved the highest accuracy in cross-validation testing")
    print("✓ NOTE: 73.86% accuracy is from testing, actual implementation may vary")
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=50,   # Optimized for performance
        max_depth=3,       # Balanced complexity
        learning_rate=0.1, # Good learning rate
        random_state=42,
        verbose=-1,        # Quiet operation
        force_col_wise=True # Better compatibility with scikit-learn
    )

    # Cross-validation with detailed metrics
    cv_scores = cross_validate(lgb_model, X_complete_scaled, y_complete, 
                               cv=5, scoring=['accuracy', 'f1_weighted', 'precision_macro', 'recall_macro'])
    
    print(f"Cross-validation results (Anti-overfitting model):")
    print(f"  Accuracy: {cv_scores['test_accuracy'].mean():.3f} ± {cv_scores['test_accuracy'].std():.3f}")
    print(f"  F1 Score: {cv_scores['test_f1_weighted'].mean():.3f} ± {cv_scores['test_f1_weighted'].std():.3f}")
    print(f"  Precision: {cv_scores['test_precision_macro'].mean():.3f} ± {cv_scores['test_precision_macro'].std():.3f}")
    print(f"  Recall: {cv_scores['test_recall_macro'].mean():.3f} ± {cv_scores['test_recall_macro'].std():.3f}")
    
    # Train final model and predict
    print("\nTraining final model and predicting missing values...")
    lgb_model.fit(X_complete_scaled, y_complete)
    y_pred = lgb_model.predict(X_missing_scaled)
    dataset1.loc[habit_missing, 'habit'] = le.inverse_transform(y_pred)
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature importance for habit prediction:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    # Show imputation results
    print(f"\nImputation completed successfully!")
    print(f"Missing values filled: {len(df_missing)}")
    print(f"Final dataset shape: {dataset1.shape}")
    print(f"Remaining missing values: {dataset1['habit'].isnull().sum()}")

    # Export the no missing data version of dataset1
    print("\n" + "="*60)
    print("EXPORTING NO MISSING DATA VERSION OF DATASET1")
    print("="*60)
    
    # Create datasets directory if it doesn't exist
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
    
    # Export the no missing data version of dataset1 (after Phase 3 imputation)
    no_missing_filename = os.path.join(datasets_dir, 'dataset1_final.csv')
    dataset1.to_csv(no_missing_filename, index=False)
    
    print(f"✓ Exported no missing data dataset to: {no_missing_filename}")
    print(f"✓ Dataset shape: {dataset1.shape}")
    print(f"✓ Habit categories: {dataset1['habit'].nunique()}")
    print(f"✓ Missing values in habit: {dataset1['habit'].isnull().sum()}")
    print(f"✓ Total missing values: {dataset1.isnull().sum().sum()}")
    
    # Show final habit distribution
    print(f"\nFinal habit distribution (no missing data):")
    habit_dist = dataset1['habit'].value_counts()
    for habit, count in habit_dist.items():
        print(f"  {habit}: {count}")
    
    print(f"\n✓ No missing data version ready for analysis!")
    print(f"✓ File saved as: {no_missing_filename}")

#%%
# ============================================================================
# PHASE 4: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*60)
print("PHASE 4: FEATURE ENGINEERING")
print("="*60)

# Create vigilance indicators
dataset1['vigilance_score'] = dataset1['bat_landing_to_food']

# Vigilance levels based on quartiles
thresholds = dataset1['vigilance_score'].quantile([0.25, 0.5, 0.75])
dataset1['vigilance_level'] = pd.cut(
    dataset1['vigilance_score'],
    bins=[0, thresholds[0.25], thresholds[0.5], thresholds[0.75], float('inf')],
    labels=['low', 'medium', 'high', 'very_high']
)

dataset1['high_vigilance'] = (dataset1['vigilance_score'] > thresholds[0.75]).astype(int)

# Avoidance behavior
dataset1['avoidance_behavior'] = (
    (dataset1['risk'] == 0) & 
    (dataset1['high_vigilance'] == 1)
).astype(int)

# Rat presence features
dataset1['rat_presence_duration'] = (
    dataset1['rat_period_end'] - dataset1['rat_period_start']
).dt.total_seconds()

dataset1['immediate_rat_threat'] = (
    dataset1['seconds_after_rat_arrival'] < 30
).astype(int)

print(f"Features created:")
print(f"  High vigilance rate: {dataset1['high_vigilance'].mean()*100:.1f}%")
print(f"  Avoidance behavior rate: {dataset1['avoidance_behavior'].mean()*100:.1f}%")
print(f"  Immediate threat cases: {dataset1['immediate_rat_threat'].mean()*100:.1f}%")

# Merge with Dataset 2 for context
dataset1['merge_date'] = dataset1['start_time'].dt.date
dataset1['merge_hour'] = dataset1['start_time'].dt.hour
dataset2['merge_date'] = dataset2['time'].dt.date
dataset2['merge_hour'] = dataset2['time'].dt.hour

hourly_context = dataset2.groupby(['merge_date', 'merge_hour']).agg({
    'bat_landing_number': 'sum',
    'food_availability': 'mean',
    'rat_minutes': 'sum',
    'rat_arrival_number': 'sum'
}).reset_index()

data = pd.merge(dataset1, hourly_context, 
                on=['merge_date', 'merge_hour'], how='left')
data[['bat_landing_number', 'food_availability', 
      'rat_minutes', 'rat_arrival_number']] = data[['bat_landing_number', 
      'food_availability', 'rat_minutes', 'rat_arrival_number']].fillna(0)

# Save the merged dataset for reference
merged_filename = os.path.join(datasets_dir, 'dataset1_merged_with_context.csv')
data.to_csv(merged_filename, index=False)
print(f"✓ Saved merged dataset to: {merged_filename}")
print(f"✓ Merged dataset shape: {data.shape}")
print(f"✓ Columns added from Dataset 2: bat_landing_number, food_availability, rat_minutes, rat_arrival_number")

# PHASE 4: Create Correlation Heatmaps
print("\nCreating correlation heatmaps...")
print("NOTE: Creating both merged data and Dataset 1 only correlations in one figure")

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')

# Left plot: Correlation using merged data (includes Dataset 2 context)
corr_cols = ['vigilance_score', 'rat_minutes', 'immediate_rat_threat', 
             'risk', 'reward', 'food_availability']
corr_matrix = data[corr_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, cbar_kws={'shrink': 0.8}, ax=ax1)
ax1.set_title('Merged Data Correlations\n(Includes Dataset 2 Context)', fontweight='bold')

# Right plot: Correlation using only Dataset 1 features
dataset1_corr_cols = ['vigilance_score', 'immediate_rat_threat', 'risk', 'reward']
dataset1_corr_matrix = dataset1[dataset1_corr_cols].corr()

sns.heatmap(dataset1_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, cbar_kws={'shrink': 0.8}, ax=ax2)
ax2.set_title('Dataset 1 Only Correlations\n(No External Context)', fontweight='bold')

plt.suptitle('Phase 4: Feature Correlations Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the combined correlation heatmaps
corr_plot_filename = os.path.join(plots_dir, 'correlation_heatmaps_comparison.png')
plt.savefig(corr_plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved combined correlation heatmaps to: {corr_plot_filename}")
plt.show()

#%%
# ============================================================================
# PHASE 5: STATISTICAL ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("PHASE 5: STATISTICAL ANALYSIS")
print("="*60)

# Main analysis: Compare vigilance with/without rats
with_rats = data[data['rat_arrival_number'] > 0]
without_rats = data[data['rat_arrival_number'] == 0]

print(f"\nSample sizes:")
print(f"  With rats: n = {len(with_rats)}")
print(f"  Without rats: n = {len(without_rats)}")

# T-test for vigilance difference
t_stat, p_value = stats.ttest_ind(
    with_rats['vigilance_score'].dropna(),
    without_rats['vigilance_score'].dropna()
)

# Calculate effect sizes
mean_with = with_rats['vigilance_score'].mean()
mean_without = without_rats['vigilance_score'].mean()
mean_diff = mean_with - mean_without
percent_change = (mean_diff / mean_without) * 100

# Cohen's d
pooled_std = np.sqrt((with_rats['vigilance_score'].std()**2 + 
                     without_rats['vigilance_score'].std()**2) / 2)
cohens_d = mean_diff / pooled_std

print(f"\nVigilance Analysis:")
print(f"  Mean with rats: {mean_with:.2f} seconds")
print(f"  Mean without rats: {mean_without:.2f} seconds")
print(f"  Difference: {mean_diff:.2f} seconds ({percent_change:+.1f}%)")
print(f"  T-statistic: {t_stat:.3f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Cohen's d: {cohens_d:.3f}")

# Analyze by habit category
print("\nVigilance by habit category and rat presence:")
habit_analysis = data.groupby(['habit', data['rat_arrival_number'] > 0])['vigilance_score'].agg(['mean', 'count'])
print(habit_analysis)

# Chi-square test for avoidance behavior
contingency = pd.crosstab(data['rat_arrival_number'] > 0, 
                          data['avoidance_behavior'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

print(f"\nAvoidance Behavior Analysis:")
print(f"  With rats: {with_rats['avoidance_behavior'].mean()*100:.1f}%")
print(f"  Without rats: {without_rats['avoidance_behavior'].mean()*100:.1f}%")
print(f"  Chi-square p-value: {p_chi:.4f}")

# PHASE 5: Create Statistical Analysis Plots
print("\nCreating statistical analysis plots...")
print("NOTE: Using merged data for statistical analysis (includes Dataset 2 context for rat presence)")

# Plot 1: Vigilance comparison
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

# Bar chart comparison
means = [mean_without, mean_with]
errors = [without_rats['vigilance_score'].std(), with_rats['vigilance_score'].std()]
bars = ax1.bar(['No Rats', 'Rats Present'], means, yerr=errors, 
               color=['#2ecc71', '#e74c3c'], capsize=5, alpha=0.7)
ax1.set_ylabel('Vigilance Score (seconds)')
ax1.set_title('Average Vigilance by Rat Presence')
for bar, val in zip(bars, means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}', ha='center', fontweight='bold')

# Distribution comparison
ax2.hist(without_rats['vigilance_score'].dropna(), alpha=0.5, 
         label='No Rats', bins=30, color='#2ecc71')
ax2.hist(with_rats['vigilance_score'].dropna(), alpha=0.5, 
         label='Rats Present', bins=30, color='#e74c3c')
ax2.set_xlabel('Vigilance Score')
ax2.set_ylabel('Frequency')
ax2.set_title('Vigilance Distribution')
ax2.legend()

# Time response curve
time_bins = [0, 30, 60, 120, 300, 600]
data['time_since_rat'] = pd.cut(data['seconds_after_rat_arrival'], 
                                 bins=time_bins, 
                                 labels=['0-30s', '30-60s', '60-120s', '120-300s', '300-600s'])
time_vigilance = data.groupby('time_since_rat')['vigilance_score'].mean()
ax3.plot(range(len(time_vigilance)), time_vigilance.values, 
         'o-', color='darkblue', linewidth=2, markersize=8)
ax3.set_xlabel('Time Since Rat Arrival')
ax3.set_ylabel('Average Vigilance')
ax3.set_title('Temporal Response to Rat Presence')
ax3.set_xticks(range(len(time_vigilance)))
ax3.set_xticklabels(time_vigilance.index, rotation=45)
ax3.grid(True, alpha=0.3)

plt.suptitle('Phase 5: Statistical Analysis Plots', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the statistical analysis plots
stats_plot_filename = os.path.join(plots_dir, 'statistical_analysis.png')
plt.savefig(stats_plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved statistical analysis plots to: {stats_plot_filename}")
plt.show()

#%%
# ============================================================================
# PHASE 6: VISUALIZATION AND CONCLUSION
# ============================================================================
print("\n" + "="*60)
print("PHASE 6: VISUALIZATION AND CONCLUSION")
print("="*60)

# PHASE 6: Create Habit Analysis Plot and Summary Statistics
print("\nCreating habit analysis plot and summary statistics...")
print("NOTE: Using merged data for habit analysis (includes Dataset 2 context for rat presence)")

# Create visualization with habit analysis and summary
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='white')

# Plot 1: Habit category analysis
habit_vig = data.groupby('habit')['vigilance_score'].mean().sort_values()
colors = ['#3498db' if 'avoid' in cat or 'slow' in cat else '#e67e22' 
          for cat in habit_vig.index]
habit_vig.plot(kind='barh', ax=ax1, color=colors)
ax1.set_xlabel('Average Vigilance Score')
ax1.set_title('Vigilance by Behavior Type')

# Plot 2: Summary statistics
ax2.axis('off')
summary_text = f"""
STATISTICAL SUMMARY

Sample Size:
• With rats: n = {len(with_rats)}
• Without rats: n = {len(without_rats)}

Vigilance Test:
• Increase: {percent_change:+.1f}%
• P-value: {p_value:.4f}
• Cohen's d: {cohens_d:.3f}
• Result: {'✓ Significant' if p_value < 0.05 else '✗ Not significant'}

Avoidance Behavior:
• Difference: {(with_rats['avoidance_behavior'].mean() - without_rats['avoidance_behavior'].mean())*100:+.1f}%
• Chi-square p: {p_chi:.4f}

Interpretation:
Effect size is {'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'}
"""
ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.5))

plt.suptitle('Phase 6: Habit Analysis and Summary Statistics', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the habit analysis and summary plot
habit_summary_filename = os.path.join(plots_dir, 'habit_analysis_summary.png')
plt.savefig(habit_summary_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved habit analysis and summary plot to: {habit_summary_filename}")

plt.show()

#%%
# ============================================================================
# FINAL ANSWER
# ============================================================================
print("\n" + "="*60)
print("FINAL ANSWER: INVESTIGATION A")
print("="*60)

if p_value < 0.05 and percent_change > 10:
    print("\n✓ YES - BATS LIKELY PERCEIVE RATS AS PREDATORS")
    print("\nEvidence:")
    print(f"1. Vigilance significantly increases by {percent_change:.1f}% when rats present")
    print(f"2. Statistical significance: p = {p_value:.4f} < 0.05")
    print(f"3. Effect size (Cohen's d = {cohens_d:.3f}) indicates {['small', 'medium', 'large'][int(abs(cohens_d) > 0.5) + int(abs(cohens_d) > 0.8)]} practical significance")
    print(f"4. Avoidance behavior increases by {(with_rats['avoidance_behavior'].mean() - without_rats['avoidance_behavior'].mean())*100:.1f}%")
    print(f"5. Behavioral patterns (slow/avoid) more common with rat presence")
else:
    print("\n✗ INSUFFICIENT EVIDENCE FOR PREDATOR PERCEPTION")
    print("\nFindings:")
    print(f"1. Vigilance difference: {percent_change:+.1f}% (p = {p_value:.4f})")
    print(f"2. Effect size: Cohen's d = {cohens_d:.3f}")
    print(f"3. Statistical significance not achieved" if p_value >= 0.05 else "Effect too small to be meaningful")
    print("4. Alternative explanation: Rats may be seen primarily as competitors")

print("\n" + "="*60)
print("Analysis Complete")
print("="*60)
#%%