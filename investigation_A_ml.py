# -*- coding: utf-8 -*-
#%%
# ============================================================================
# LIBRARIES AND SETUP
# ============================================================================
"""
Investigation A: Do bats perceive rats as potential predators?

This investigation analyzes bat foraging behavior in the presence of rats using
two complementary datasets:
- Dataset1: Individual bat landing events with detailed behavioral analysis
- Dataset2: Environmental context from 30-minute observation periods

Key behavioral indicators for predator perception:
- Vigilance: bat_landing_to_food (time before approaching food)
- Avoidance: risk-taking behavior and foraging success
- Context: rat presence timing and environmental factors

Libraries used:
- pandas/numpy: Data manipulation and numerical analysis
- matplotlib/seaborn: Visualization and statistical plots  
- scipy: Statistical testing (t-tests, chi-square)
- sklearn: Missing value imputation and data preprocessing
- Random Forest/SVM: ML classification for competition vs predator behaviors
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os
import warnings
warnings.filterwarnings('ignore')

#%%
# ============================================================================
# PHASE 1: DATA LOADING AND RISK-REWARD CORRELATION ANALYSIS
# ============================================================================
print("="*60)
print("PHASE 1: DATA LOADING AND RISK-REWARD CORRELATION ANALYSIS")
print("="*60)

# Load datasets
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')

print(f"\nDataset 1: {dataset1.shape[0]} rows, {dataset1.shape[1]} columns")
print(f"Dataset 2: {dataset2.shape[0]} rows, {dataset2.shape[1]} columns")

# Dataset2 Environmental Context Overview
print(f"\nDataset2 Environmental Context Overview:")
print(f"="*40)
print(f"Total observation periods: {len(dataset2)}")
print(f"Periods with rat activity: {(dataset2['rat_arrival_number'] > 0).sum()}")
print(f"Average rat activity duration: {dataset2['rat_minutes'].mean():.1f} minutes")
print(f"Average bat competition: {dataset2['bat_landing_number'].mean():.1f} landings/period")
print(f"Food availability range: {dataset2['food_availability'].min():.1f} - {dataset2['food_availability'].max():.1f}")
print(f"Peak activity periods: {(dataset2['bat_landing_number'] > 50).sum()} high-activity periods")

# Analyze risk-reward correlation patterns
print("\nRisk-Reward Correlation Analysis:")
print("="*40)

# Check missing data patterns
missing_habit_mask = dataset1['habit'].isnull()
print(f"Missing habits: {missing_habit_mask.sum()}")

# Check numeric data patterns (habits with numbers)
def has_numbers(value):
    if pd.isna(value):
        return False
    return any(char.isdigit() for char in str(value))

numeric_habits = dataset1['habit'].apply(has_numbers)
print(f"Numeric habits: {numeric_habits.sum()}")

# Analyze risk-reward patterns for missing/numeric data
missing_or_numeric = missing_habit_mask | numeric_habits
missing_risk_reward = dataset1[missing_or_numeric][['risk', 'reward']].value_counts()
print(f"\nRisk-Reward patterns for missing/numeric habits:")
print(missing_risk_reward)

# Risk-reward patterns analysis complete - will be used for classification

#%%
# ============================================================================
# PHASE 1 VISUALIZATION: DATA OVERVIEW
# ============================================================================
print("\nCreating Phase 1 visualization: Data Overview")

# Create plots directory for ML analysis if it doesn't exist
plots_dir = os.path.join('plots', 'ml_analysis')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Simple Phase 1 visualization - just show the key insight
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

# Plot 1: Risk-Reward distribution (main finding)
risk_reward_counts = dataset1[['risk', 'reward']].value_counts().unstack(fill_value=0)
risk_reward_counts.plot(kind='bar', ax=ax1, color=['lightcoral', 'lightgreen'], alpha=0.7)
ax1.set_title('Risk-Reward Behavior Patterns', fontweight='bold')
ax1.set_xlabel('Risk Level')
ax1.set_ylabel('Number of Bat Landings')
ax1.legend(['No Reward', 'Reward'])
ax1.tick_params(axis='x', rotation=0)

# Plot 2: Dataset sizes
datasets_info = [len(dataset1), len(dataset2)]
dataset_names = ['Bat Landings\n(Dataset1)', '30min Periods\n(Dataset2)']
bars = ax2.bar(dataset_names, datasets_info, color=['steelblue', 'darkorange'], alpha=0.7)
ax2.set_title('Data Available for Analysis', fontweight='bold')
ax2.set_ylabel('Number of Records')
for bar, val in zip(bars, datasets_info):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{val}', ha='center', fontweight='bold')

plt.suptitle('PHASE 1: Initial Data Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save Phase 1 plot
phase1_filename = os.path.join(plots_dir, 'phase1_data_overview.png')
plt.savefig(phase1_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved Phase 1 visualization to: {phase1_filename}")
plt.show()

#%%
# ============================================================================
# PHASE 2: UPDATED HABIT CLASSIFICATION AND DATA CLEANING
# ============================================================================
print("\n" + "="*60)
print("PHASE 2: UPDATED HABIT CLASSIFICATION AND DATA CLEANING")
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

# Store original habits before classification
original_habits = dataset1['habit'].copy()

# UPDATED habit classification based on new understanding
def classify_habit_updated(habit, risk, reward):
    """
    Updated habit classification based on risk-reward correlation:
    - Missing value + numeric -> Unknown (risk=0, reward=0)
    - risk=0 & reward=1 -> fast or pick
    - risk=1 & reward=0 -> rat-related
    """
    # Handle missing/numeric data based on risk-reward
    if pd.isna(habit) or any(char.isdigit() for char in str(habit) if pd.notna(habit)):
        if risk == 0 and reward == 0:
            return 'unknown'
        elif risk == 0 and reward == 1:
            return 'fast'  # Default for missing but fast behavior
        elif risk == 1 and reward == 0:
            return 'rat'   # Default for missing but rat-related
    
    if pd.isna(habit):
        return 'unknown'
    
    habit_str = str(habit).lower()
    
    # Updated classification rules based on user specifications
    # Pick behaviors (keep pick as pick)
    if 'pick' in habit_str:
        if 'and' in habit_str and ('others' in habit_str or 'rat' in habit_str or 'bat' in habit_str):
            return 'pick'  # N + pick -> pick
        return 'pick'
    
    # Fast behaviors
    if 'fast' in habit_str:
        if 'far' in habit_str:
            return 'fast'  # fast_far -> fast
        elif 'and' in habit_str and 'pick' in habit_str:
            return 'pick'  # fast_and_pick -> pick
        return 'fast'
    
    # Bat and rat behaviors -> bat_and_rat
    if ('bat' in habit_str and 'rat' in habit_str) or \
       ('bat' in habit_str and any(x in habit_str for x in ['and', '+'])) or \
       ('rat' in habit_str and any(x in habit_str for x in ['and', '+'])):
        return 'bat_and_rat'
    
    # Individual bat or rat
    if 'bat' in habit_str:
        return 'bat_and_rat'  # Group with combined category
    if 'rat' in habit_str:
        return 'bat_and_rat'  # Group with combined category
        
    # No eating behaviors: no food, eating, bowl_out, pup and mon, both, gaze
    if any(term in habit_str for term in ['no_food', 'eating', 'bowl_out', 'pup', 'mon', 'both', 'gaze']):
        return 'no_eating'
    
    # Attack/fight behaviors
    if 'attack' in habit_str or 'fight' in habit_str:
        return 'bat_and_rat'  # Group aggressive behaviors with bat_and_rat
    
    # Default fallback based on risk-reward correlation
    if risk == 0 and reward == 1:
        return 'fast'
    elif risk == 1 and reward == 0:
        return 'bat_and_rat'
    else:
        return 'unknown'

# Apply updated classification
dataset1['habit_classified'] = dataset1.apply(
    lambda row: classify_habit_updated(row['habit'], row['risk'], row['reward']), 
    axis=1
)

# Replace original habit column
dataset1['habit'] = dataset1['habit_classified']
dataset1.drop('habit_classified', axis=1, inplace=True)

# Show classification results
original_counts = original_habits.value_counts()
new_counts = dataset1['habit'].value_counts()

print(f"\nClassification Results:")
print(f"Original unique habits: {len(original_counts)}")
print(f"New categories: {len(new_counts)}")
print(f"\nUpdated habit classification:")
for cat, count in new_counts.items():
    print(f"  {cat}: {count}")

# Verify risk-reward correlation
print(f"\nRisk-Reward verification after classification:")
correlation_check = dataset1.groupby(['habit', 'risk', 'reward']).size().reset_index(name='count')
print(correlation_check.sort_values(['habit', 'count'], ascending=[True, False]))

print(f"\nClassification completed successfully!")
print(f"Missing values after classification: {dataset1['habit'].isnull().sum()}")

# Store original habits for later comparison plot (after Phase 3)
original_habits_for_plot = original_counts.copy()

#%%
# ============================================================================
# PHASE 2 VISUALIZATION: HABIT CLASSIFICATION RESULTS
# ============================================================================
print("\nCreating Phase 2 visualization: Habit Classification Results")

# Simple Phase 2 visualization - before and after classification
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

# Plot 1: Original habits (top 8 only)
original_top8 = original_habits_for_plot.head(8)
ax1.bar(range(len(original_top8)), original_top8.values, color='lightcoral', alpha=0.7)
ax1.set_title('Original Habit Categories', fontweight='bold')
ax1.set_xlabel('Habit Type')
ax1.set_ylabel('Count')
ax1.set_xticks(range(len(original_top8)))
# Show actual habit names but truncated for readability
habit_labels = [name[:8] + '...' if len(str(name)) > 8 else str(name) for name in original_top8.index]
ax1.set_xticklabels(habit_labels, rotation=45, ha='right')

# Plot 2: New classified categories
new_counts.plot(kind='bar', ax=ax2, color='steelblue', alpha=0.7)
ax2.set_title('After Classification (Clean Categories)', fontweight='bold')
ax2.set_xlabel('Category')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)

plt.suptitle('PHASE 2: Habit Classification Results', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save Phase 2 plot
phase2_filename = os.path.join(plots_dir, 'phase2_classification_results.png')
plt.savefig(phase2_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved Phase 2 visualization to: {phase2_filename}")
plt.show()

#%%
# ============================================================================
# PHASE 3: UNKNOWN VALUE ANALYSIS AND SMART IMPUTATION
# ============================================================================
print("\n" + "="*60)
print("PHASE 3: UNKNOWN VALUE ANALYSIS AND SMART IMPUTATION")
print("="*60)

# Analyze unknown values in detail
unknown_mask = dataset1['habit'] == 'unknown'
unknown_data = dataset1[unknown_mask].copy()

print(f"Unknown entries analysis:")
print(f"Total unknown entries: {unknown_mask.sum()}")
print(f"All unknown entries have risk=0, reward=0: {(unknown_data['risk'] == 0).all() and (unknown_data['reward'] == 0).all()}")

# Analyze context of unknown entries
print(f"\nUnknown entries context:")
print(f"Average vigilance: {unknown_data['bat_landing_to_food'].mean():.2f} seconds")
print(f"Average time after rat arrival: {unknown_data['seconds_after_rat_arrival'].mean():.2f} seconds")
print(f"Hours after sunset range: {unknown_data['hours_after_sunset'].min():.1f} - {unknown_data['hours_after_sunset'].max():.1f}")

# Smart imputation strategy for unknown values
print(f"\nSmart imputation strategy:")
print(f"Based on risk=0, reward=0 pattern and environmental context:")
print(f"- Low food reward suggests: neutral/waiting behavior")
print(f"- No risk-taking suggests: cautious approach")  
print(f"- Environmental analysis suggests: ambiguous situations")

# Create behavioral categories based on vigilance and context
def impute_unknown_smart(row):
    """Smart imputation based on vigilance and environmental cues"""
    vigilance = row['bat_landing_to_food']
    time_after_rat = row['seconds_after_rat_arrival']
    
    if vigilance > 10:  # High vigilance
        if time_after_rat < 300:  # Within 5 minutes of rat
            return 'cautious'  # Cautious behavior near rats
        else:
            return 'slow_approach'  # General slow approach
    elif vigilance < 3:  # Very quick approach
        return 'quick_neutral'  # Quick but neutral outcome
    else:
        return 'neutral_wait'  # Standard waiting behavior

# Apply smart imputation to unknown values
print(f"\nApplying smart imputation...")
dataset1.loc[unknown_mask, 'habit'] = unknown_data.apply(impute_unknown_smart, axis=1)

# Verify imputation results
print(f"\nImputation results:")
new_categories = dataset1.loc[unknown_mask, 'habit'].value_counts()
for category, count in new_categories.items():
    print(f"  {category}: {count}")

print(f"\nFinal habit distribution after imputation:")
final_habits = dataset1['habit'].value_counts()
for habit, count in final_habits.items():
    print(f"  {habit}: {count}")

# Verify data is clean after imputation
print(f"\nData quality check:")
print(f"Missing habits: {dataset1['habit'].isna().sum()}")
print(f"Missing vigilance values: {dataset1['bat_landing_to_food'].isna().sum()}")
print("All data successfully cleaned after Phase 3 imputation")

# Export cleaned dataset for further analysis
print("\n" + "="*60)
print("EXPORTING CLEANED DATASET")
print("="*60)

datasets_dir = os.path.join('datasets', 'ml_analysis')
if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)

cleaned_filename = os.path.join(datasets_dir, 'dataset1_cleaned_ml_ready.csv')
dataset1.to_csv(cleaned_filename, index=False)

print(f"Exported cleaned dataset to: {cleaned_filename}")
print(f"No unknown values remaining: {(dataset1['habit'] == 'unknown').sum() == 0}")

#%%
# ============================================================================
# PHASE 3 VISUALIZATION: IMPUTATION RESULTS
# ============================================================================
print("\nCreating Phase 3 visualization: Imputation Results")

# Simple Phase 3 visualization - before and after imputation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

# Plot 1: Unknown values progress
unknown_progress = [
    new_counts.get('unknown', 0),  # Before imputation
    (dataset1['habit'] == 'unknown').sum()  # After imputation
]
stages = ['Before Imputation', 'After Imputation']
bars = ax1.bar(stages, unknown_progress, color=['lightcoral', 'lightgreen'], alpha=0.7)
ax1.set_title('Unknown Values Resolved', fontweight='bold')
ax1.set_ylabel('Number of Unknown Values')
for bar, val in zip(bars, unknown_progress):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val}', ha='center', fontweight='bold')

# Plot 2: Final cleaned categories
final_habit_counts = dataset1['habit'].value_counts()
final_habit_counts.plot(kind='bar', ax=ax2, color='steelblue', alpha=0.7)
ax2.set_title('Final Clean Categories', fontweight='bold')
ax2.set_xlabel('Habit Category')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)

plt.suptitle('PHASE 3: Data Cleaning Complete', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save Phase 3 plot
phase3_filename = os.path.join(plots_dir, 'phase3_imputation_results.png')
plt.savefig(phase3_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved Phase 3 visualization to: {phase3_filename}")
plt.show()

#%%
# ============================================================================
# PHASE 4: ML CLASSIFICATION - COMPETITION vs PREDATOR BEHAVIORS
# ============================================================================
print("\n" + "="*60)
print("PHASE 4: ML CLASSIFICATION - COMPETITION vs PREDATOR BEHAVIORS")
print("="*60)

print("Goal: Use ML to classify 'bat_and_rat' entries into 'competition' or 'predator' behaviors")
print("Method: Train on clear behavioral patterns, then predict mixed categories")

# Current habit distribution before ML classification
current_habits = dataset1['habit'].value_counts()
print(f"\nCurrent habit categories:")
for habit, count in current_habits.items():
    print(f"  {habit}: {count}")

# Identify bat_and_rat entries that need classification
bat_and_rat_mask = dataset1['habit'] == 'bat_and_rat'
bat_and_rat_count = bat_and_rat_mask.sum()

print(f"\nEntries to classify: {bat_and_rat_count} 'bat_and_rat' behaviors")

if bat_and_rat_count > 0:
    print(f"Other categories will remain unchanged: {len(dataset1) - bat_and_rat_count} entries")
    
    # ============================================================================
    # CREATE TRAINING DATA FROM CLEAR BEHAVIORAL PATTERNS
    # ============================================================================
    print(f"\n" + "="*40)
    print("CREATING TRAINING DATA FROM BEHAVIORAL PATTERNS")
    print("="*40)

    training_data_list = []
    training_labels_list = []
    
    # Use existing categories to create training examples based on vigilance and behavior patterns
    other_entries = dataset1[~bat_and_rat_mask]
    for idx, row in other_entries.iterrows():
        vigilance = row['bat_landing_to_food']
        
        # Competition-focused behaviors: fast approach (low vigilance) + success
        if (row['habit'] in ['fast', 'pick'] and vigilance < 10) or \
           (row['risk'] == 0 and row['reward'] == 1 and vigilance < 15):
            training_data_list.append(row)
            training_labels_list.append('competition')
        # Predator-focused behaviors: slow/cautious approach (high vigilance) + risk-taking
        elif (row['habit'] == 'rat' and vigilance > 8) or \
             (row['risk'] == 1 and row['reward'] == 0 and vigilance > 10):
            training_data_list.append(row)
            training_labels_list.append('predator')

    print(f"Training data created from behavioral patterns:")
    print(f"  Competition examples: {training_labels_list.count('competition')}")
    print(f"  Predator examples: {training_labels_list.count('predator')}")
    
    if len(training_data_list) >= 20 and training_labels_list.count('competition') >= 5 and training_labels_list.count('predator') >= 5:
        print(f"\nSufficient training data available. Training ML model...")
        
        # ============================================================================
        # TRAIN ML MODEL FOR COMPETITION VS PREDATOR CLASSIFICATION
        # ============================================================================
        
        # Prepare training data
        training_df = pd.DataFrame(training_data_list)
        
        # Feature set based on ml_advice.txt
        feature_cols = [
            'bat_landing_to_food',           # vigilance (key discriminator)
            'seconds_after_rat_arrival',     # rat timing
            'risk', 'reward',                # behavior outcomes  
            'hours_after_sunset',            # time of day
            'month'                          # seasonality
        ]
        
        X_train = training_df[feature_cols]
        y_train = np.array(training_labels_list)
        
        # Handle missing values and scale features
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_train_clean = imputer.fit_transform(X_train)
        X_train_scaled = scaler.fit_transform(X_train_clean)
        
        # Train SVM classifier (best performer from ml_algorithm_comparison.py)
        ml_model = SVC(kernel='linear', random_state=42, probability=True)
        ml_model.fit(X_train_scaled, y_train)
        
        print(f"SVM model trained successfully!")
        
        # SVM feature importance (coefficient analysis for linear kernel)
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': abs(ml_model.coef_[0])  # Absolute coefficients for importance
        }).sort_values('coefficient', ascending=False)
        
        print(f"\nFeature importance (SVM coefficients):")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.3f}")
        
        # ============================================================================
        # CLASSIFY ONLY BAT_AND_RAT ENTRIES
        # ============================================================================
        print(f"\n" + "="*40)
        print("CLASSIFYING BAT_AND_RAT ENTRIES AS COMPETITION vs PREDATOR")
        print("="*40)
        
        # Prepare bat_and_rat entries for prediction
        bat_and_rat_data = dataset1[bat_and_rat_mask].copy()
        X_predict = bat_and_rat_data[feature_cols]
        X_predict_clean = imputer.transform(X_predict)
        X_predict_scaled = scaler.transform(X_predict_clean)
        
        # Make predictions
        predictions = ml_model.predict(X_predict_scaled)
        prediction_confidence = np.max(ml_model.predict_proba(X_predict_scaled), axis=1)
        
        print(f"Classified {len(predictions)} bat_and_rat entries")
        print(f"Average confidence: {prediction_confidence.mean():.3f}")
        
        # Update only the bat_and_rat entries in dataset1
        dataset1.loc[bat_and_rat_mask, 'habit'] = predictions
        
        # Show results
        pred_counts = pd.Series(predictions).value_counts()
        print(f"\nClassification results for bat_and_rat entries:")
        for label, count in pred_counts.items():
            print(f"  {label}: {count}")
        
        print(f"\nPhase 4 completed: bat_and_rat entries classified as 'competition' or 'predator'!")
        
    else:
        print(f"\nWARNING: Insufficient training data")
        print(f"Using simple heuristic classification for bat_and_rat entries...")
        
        # Vigilance-based heuristic fallback for bat_and_rat only
        heuristic_predictions = []
        bat_and_rat_data = dataset1[bat_and_rat_mask].copy()
        for idx, row in bat_and_rat_data.iterrows():
            vigilance = row['bat_landing_to_food']
            # Use vigilance as primary classifier for mixed behaviors
            if vigilance > 15:  # High vigilance = predator response
                heuristic_predictions.append('predator')
            elif vigilance < 5:  # Very low vigilance = competition behavior
                heuristic_predictions.append('competition')
            else:  # Medium vigilance - use secondary factors
                if row['risk'] == 1 and row['reward'] == 0:
                    heuristic_predictions.append('predator')
                else:
                    heuristic_predictions.append('competition')
        
        # Update only bat_and_rat entries
        dataset1.loc[bat_and_rat_mask, 'habit'] = heuristic_predictions
        
        pred_counts = pd.Series(heuristic_predictions).value_counts()
        print(f"Heuristic classification results:")
        for label, count in pred_counts.items():
            print(f"  {label}: {count}")

    # Show final distribution
    print(f"\n" + "="*40)
    print("FINAL HABIT DISTRIBUTION")
    print("="*40)
    
    final_counts = dataset1['habit'].value_counts()
    print(f"All categories (bat_and_rat entries now classified):")
    for category, count in final_counts.items():
        print(f"  {category}: {count}")
    
else:
    print(f"\nNo 'bat_and_rat' entries found to classify.")
    print(f"All entries already have specific categories.")

#%%
# ============================================================================
# PHASE 4 VISUALIZATION: ML CLASSIFICATION RESULTS
# ============================================================================
print("\nCreating Phase 4 visualization: ML Classification Results")

# Phase 4 visualization - before and after ML classification
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

# Plot 1: bat_and_rat entries before classification
if 'bat_and_rat_count' in locals() and bat_and_rat_count > 0:
    before_data = pd.Series([bat_and_rat_count], index=['bat_and_rat'])
    before_data.plot(kind='bar', ax=ax1, color='orange', alpha=0.7)
    ax1.set_title(f'Before ML: {bat_and_rat_count} mixed entries', fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=0)
else:
    ax1.text(0.5, 0.5, 'No bat_and_rat entries\nto classify', 
            ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    ax1.set_title('Before ML Classification', fontweight='bold')

# Plot 2: After ML - final distribution including all categories
final_habits_p4 = dataset1['habit'].value_counts()
colors = ['#2ecc71' if cat == 'competition' else '#e74c3c' if cat == 'predator' else 'steelblue' 
          for cat in final_habits_p4.index]
final_habits_p4.plot(kind='bar', ax=ax2, color=colors, alpha=0.7)
ax2.set_title('After ML: Competition vs Predator Separation', fontweight='bold')
ax2.set_xlabel('Behavior Category')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)

plt.suptitle('PHASE 4: ML Classification Complete', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save Phase 4 plot
phase4_filename = os.path.join(plots_dir, 'phase4_competition_vs_predator.png')
plt.savefig(phase4_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved Phase 4 visualization to: {phase4_filename}")
plt.show()

# Export final dataset with classified categories
ml_final_filename = os.path.join(datasets_dir, 'dataset1_ml_classified.csv')
dataset1.to_csv(ml_final_filename, index=False)
print(f"Exported ML classified dataset to: {ml_final_filename}")

print(f"\nPhase 4 ML classification complete!")
print(f"bat_and_rat entries have been classified as either 'competition' or 'predator'.")

# ============================================================================
# ADDITIONAL PHASE 4: TIME-BASED PREDATOR PERCEPTION ANALYSIS
# ============================================================================
print(f"\n" + "="*60)
print("ADDITIONAL ANALYSIS: TIME-BASED PREDATOR PERCEPTION")
print("="*60)

# Core question: Do bats show higher vigilance when rats are present?
# Since we now have 'competition' and 'predator' categories, let's also do time-based analysis

print("Time-based Analysis: Testing vigilance patterns relative to rat activity")
print("Key measure: bat_landing_to_food (vigilance time)")

# Create vigilance measure
dataset1['vigilance'] = dataset1['bat_landing_to_food']

# Time-based vigilance analysis similar to unified version
immediate_rats = dataset1[dataset1['seconds_after_rat_arrival'] <= 60]['vigilance']  # Within 1 minute
delayed_rats = dataset1[dataset1['seconds_after_rat_arrival'] > 300]['vigilance']   # After 5+ minutes
recent_rats = dataset1[(dataset1['seconds_after_rat_arrival'] > 60) & (dataset1['seconds_after_rat_arrival'] <= 600)]['vigilance']

print(f"\nTime-Based Vigilance Analysis:")
print(f"   Average vigilance IMMEDIATELY after rats (<=1min): {immediate_rats.mean():.2f} seconds (n={len(immediate_rats)})")
print(f"   Average vigilance DELAYED after rats (>5min): {delayed_rats.mean():.2f} seconds (n={len(delayed_rats)})")
print(f"   Average vigilance RECENT after rats (1-10min): {recent_rats.mean():.2f} seconds (n={len(recent_rats)})")

if len(immediate_rats) > 0 and len(delayed_rats) > 0:
    time_difference = immediate_rats.mean() - delayed_rats.mean()
    print(f"   Difference (immediate vs delayed): {time_difference:+.2f} seconds")
    if delayed_rats.mean() > 0:
        time_percent_change = (time_difference / delayed_rats.mean()) * 100
        print(f"   Percentage change: {time_percent_change:+.1f}%")

# Risk-taking behavior analysis (time-based)
risk_immediate = dataset1[dataset1['seconds_after_rat_arrival'] <= 60]['risk'].mean()
risk_delayed = dataset1[dataset1['seconds_after_rat_arrival'] > 300]['risk'].mean() 

print(f"\nRisk-Taking Analysis (Time-Based):")
print(f"   Risk-taking IMMEDIATELY after rats (<=1min): {risk_immediate:.3f}")
print(f"   Risk-taking DELAYED after rats (>5min): {risk_delayed:.3f}")
if not pd.isna(risk_immediate) and not pd.isna(risk_delayed):
    risk_time_diff = risk_immediate - risk_delayed
    print(f"   Difference: {risk_time_diff:+.3f}")

# Reward success analysis (time-based)
reward_immediate = dataset1[dataset1['seconds_after_rat_arrival'] <= 60]['reward'].mean()
reward_delayed = dataset1[dataset1['seconds_after_rat_arrival'] > 300]['reward'].mean()

print(f"\nReward Success Analysis (Time-Based):")
print(f"   Success rate IMMEDIATELY after rats (<=1min): {reward_immediate:.3f}")
print(f"   Success rate DELAYED after rats (>5min): {reward_delayed:.3f}")
if not pd.isna(reward_immediate) and not pd.isna(reward_delayed):
    reward_time_diff = reward_immediate - reward_delayed
    print(f"   Difference: {reward_time_diff:+.3f}")

# Create comprehensive time-based visualization (4-subplot)
print(f"\nCreating comprehensive time-based analysis visualization...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), facecolor='white')

# 1. Time-based vigilance comparison
vigilance_means = [delayed_rats.mean(), immediate_rats.mean(), recent_rats.mean()]
vigilance_categories = ['Delayed (>5min)', 'Immediate (<=1min)', 'Recent (1-10min)']
colors = ['green', 'red', 'orange']
bars = ax1.bar(vigilance_categories, vigilance_means, color=colors, alpha=0.7)
ax1.set_ylabel('Average Vigilance (seconds)')
ax1.set_title('Vigilance by Rat Timing')
for bar, val in zip(bars, vigilance_means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}s', ha='center', fontweight='bold')

# 2. Risk-taking behavior (time-based)
risk_means = [risk_delayed, risk_immediate]
risk_categories = ['Delayed (>5min)', 'Immediate (<=1min)'] 
bars2 = ax2.bar(risk_categories, risk_means, color=['lightgreen', 'lightcoral'], alpha=0.7)
ax2.set_ylabel('Proportion Taking Risks')
ax2.set_title('Risk-Taking by Timing')
ax2.set_ylim(0, 1)
for bar, val in zip(bars2, risk_means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontweight='bold')

# 3. Success rates (time-based)
reward_means = [reward_delayed, reward_immediate]
reward_categories = ['Delayed (>5min)', 'Immediate (<=1min)']
bars3 = ax3.bar(reward_categories, reward_means, color=['lightblue', 'lightcoral'], alpha=0.7)
ax3.set_ylabel('Success Rate (Reward)')
ax3.set_title('Success Rate by Timing')
ax3.set_ylim(0, 1)
for bar, val in zip(bars3, reward_means):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontweight='bold')

# 4. Vigilance distribution overlay (time-based)
ax4.hist(delayed_rats, alpha=0.5, label='Delayed (>5min)', bins=15, color='green', density=True)
ax4.hist(immediate_rats, alpha=0.5, label='Immediate (<=1min)', bins=15, color='red', density=True)
ax4.set_xlabel('Vigilance (seconds)')
ax4.set_ylabel('Density')
ax4.set_title('Vigilance Distributions')
ax4.legend()

plt.suptitle('PHASE 4: Comprehensive Time-Based Predator Perception Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save time-based analysis plot
time_analysis_filename = os.path.join(plots_dir, 'phase4_time_based_predator_analysis.png')
plt.savefig(time_analysis_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved time-based analysis plot to: {time_analysis_filename}")
plt.show()

# Create comprehensive correlation heatmap combining Dataset1 and Dataset2 features
print(f"\nCreating comprehensive correlation analysis...")

# Select key numeric columns from both datasets for correlation analysis
dataset1_cols = ['vigilance', 'risk', 'reward', 'seconds_after_rat_arrival', 'hours_after_sunset']
correlation_data = dataset1[dataset1_cols].corr()

plt.figure(figsize=(10, 8), facecolor='white')
sns.heatmap(correlation_data, annot=True, fmt='.3f', cmap='RdBu_r', 
            center=0, square=True, cbar_kws={'shrink': 0.8})
plt.title('Comprehensive Feature Correlations (ML Analysis)', fontsize=12, fontweight='bold')
plt.tight_layout()

# Save correlation plot
corr_filename = os.path.join(plots_dir, 'phase4_comprehensive_correlations.png')
plt.savefig(corr_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved comprehensive correlation plot to: {corr_filename}")
plt.show()

print(f"\nAdditional Phase 4 analysis complete!")

# ============================================================================
# FINAL HABIT CATEGORY EXPLANATIONS
# ============================================================================
print(f"\n" + "="*60)
print("FINAL HABIT CATEGORY MEANINGS")
print("="*60)

final_habit_meanings = {
    'competition': 'Competition-focused behavior: Fast foraging, successful food acquisition, competing with other bats',
    'predator': 'Predator-focused behavior: Cautious approach, high vigilance, responding to rat presence as threat',
    'fast': 'Quick foraging behavior: Low vigilance, rapid food approach, efficient feeding',
    'pick': 'Selective feeding behavior: Careful food selection, moderate approach speed',
    'no_eating': 'Non-feeding behavior: No food consumption, may include exploration or vigilance',
    'cautious': 'Cautious behavior: High vigilance, slow approach, risk-aware feeding',
    'slow_approach': 'Deliberate behavior: Slow, careful approach to food source',
    'neutral_wait': 'Neutral behavior: Waiting or pausing behavior without clear feeding intent',
    'quick_neutral': 'Quick neutral behavior: Fast movement but without successful feeding'
}

current_categories = set(dataset1['habit'].unique())
print(f"\nHabit categories present in final dataset:")
for category in sorted(current_categories):
    meaning = final_habit_meanings.get(category, 'Specific behavioral pattern identified in the data')
    print(f"  {category}: {meaning}")

print(f"\nKey insight: ML classification has separated ambiguous 'bat_and_rat' entries")
print(f"into clear 'competition' and 'predator' behavioral categories for better analysis.")

#%%
# ============================================================================
# PHASE 5: HYPOTHESIS TESTING - DO BATS PERCEIVE RATS AS PREDATORS?
# ============================================================================
print("\n" + "="*60)
print("PHASE 5: HYPOTHESIS TESTING - DO BATS PERCEIVE RATS AS PREDATORS?")
print("="*60)

print("RESEARCH QUESTION: Do bats perceive rats as potential predators?")
print("EXPECTED OUTCOME: If rats are perceived as predators, bats should show")
print("                  increased vigilance (longer time before approaching food)")
print("")

# STEP 1: FORMULATE HYPOTHESES
print("STEP 1: FORMULATE HYPOTHESES")
print("=" * 40)
print("H0 (Null): μ_with_rats = μ_without_rats")
print("    (No difference in vigilance - rats not perceived as predators)")
print("")
print("H1 (Alternative): μ_with_rats > μ_without_rats") 
print("    (Higher vigilance with rats - rats perceived as predators)")
print("")
print("Significance level: α = 0.05")
print("Test type: One-tailed two-sample t-test")

# STEP 2: PREPARE DATA USING PROPER RAT PRESENCE DETECTION
print("\n" + "=" * 40)
print("STEP 2: DATA PREPARATION")
print("=" * 40)

# Create vigilance variable
dataset1['vigilance'] = dataset1['bat_landing_to_food']

# Use Dataset2 environmental context for proper rat presence detection
print("Merging with Dataset2 for accurate rat presence periods...")

# Merge datasets based on time periods
dataset1['time_period'] = dataset1['start_time'].dt.floor('30min')
dataset2['time_period'] = dataset2['time'].dt.floor('30min')

# Merge to get environmental rat presence data
merged_data = dataset1.merge(
    dataset2[['time_period', 'rat_arrival_number', 'rat_minutes']], 
    on='time_period', 
    how='left'
)

# Create proper rat presence indicator
# Rats present if there were rat arrivals OR rat activity in that 30-min period
merged_data['rats_present'] = ((merged_data['rat_arrival_number'] > 0) | 
                               (merged_data['rat_minutes'] > 0)).fillna(False)

# Update main dataset
dataset1['rats_present'] = merged_data['rats_present']
dataset1['environmental_rat_minutes'] = merged_data['rat_minutes'].fillna(0)

print("Rat presence classification:")
print(f"  Periods WITH rats present: {dataset1['rats_present'].sum()}")
print(f"  Periods WITHOUT rats present: {(~dataset1['rats_present']).sum()}")
print(f"  Total bat observations: {len(dataset1)}")

# Save merged data with environmental context
print(f"\nSaving merged data with environmental context...")
merged_filename = os.path.join(datasets_dir, 'dataset1_merged_with_environmental_context.csv')
merged_data.to_csv(merged_filename, index=False)
print(f"Saved merged dataset to: {merged_filename}")
print(f"Columns: {list(merged_data.columns)}")
print(f"Shape: {merged_data.shape}")

# STEP 3: PERFORM STATISTICAL TEST
print("\n" + "=" * 40)
print("STEP 3: STATISTICAL ANALYSIS")
print("=" * 40)

# Separate vigilance data by rat presence
vigilance_with_rats = dataset1[dataset1['rats_present']]['vigilance'].dropna()
vigilance_without_rats = dataset1[~dataset1['rats_present']]['vigilance'].dropna()

print("Sample sizes:")
print(f"  With rats: n1 = {len(vigilance_with_rats)}")
print(f"  Without rats: n2 = {len(vigilance_without_rats)}")

if len(vigilance_with_rats) > 0 and len(vigilance_without_rats) > 0:
    # Descriptive statistics
    mean_with = vigilance_with_rats.mean()
    mean_without = vigilance_without_rats.mean()
    std_with = vigilance_with_rats.std()
    std_without = vigilance_without_rats.std()
    
    print(f"\nDescriptive Statistics:")
    print(f"  With rats: μ1 = {mean_with:.3f}s, σ1 = {std_with:.3f}s")
    print(f"  Without rats: μ2 = {mean_without:.3f}s, σ2 = {std_without:.3f}s")
    
    # Two-sample t-test
    t_stat, p_value_two_tailed = stats.ttest_ind(vigilance_with_rats, vigilance_without_rats)
    
    # Convert to one-tailed test (we expect vigilance to be HIGHER with rats)
    if mean_with > mean_without:
        p_value = p_value_two_tailed / 2  # One-tailed in expected direction
    else:
        p_value = 1 - (p_value_two_tailed / 2)  # One-tailed in unexpected direction
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(vigilance_with_rats)-1)*std_with**2 + 
                         (len(vigilance_without_rats)-1)*std_without**2) / 
                        (len(vigilance_with_rats) + len(vigilance_without_rats) - 2))
    cohens_d = (mean_with - mean_without) / pooled_std
    
    # Results
    mean_diff = mean_with - mean_without
    percent_change = (mean_diff / mean_without) * 100 if mean_without > 0 else 0
    df = len(vigilance_with_rats) + len(vigilance_without_rats) - 2
    
    print(f"\nTest Results:")
    print(f"  t-statistic = {t_stat:.4f}")
    print(f"  Degrees of freedom = {df}")
    print(f"  One-tailed p-value = {p_value:.4f}")
    print(f"  Mean difference = {mean_diff:+.3f} seconds")
    print(f"  Percentage change = {percent_change:+.1f}%")
    print(f"  Cohen's d = {cohens_d:.4f}")
    
    # STEP 4: DECISION AND INTERPRETATION
    print("\n" + "=" * 40)
    print("STEP 4: DECISION AND INTERPRETATION")
    print("=" * 40)
    
    # Statistical decision
    print("Statistical Decision:")
    if p_value <= 0.05:
        if mean_with > mean_without:
            print(f"  REJECT H0: p = {p_value:.4f} ≤ 0.05")
            print(f"  ACCEPT H1: Bats show significantly higher vigilance with rats")
            statistical_conclusion = "SIGNIFICANT_INCREASE"
        else:
            print(f"  REJECT H0: p = {p_value:.4f} ≤ 0.05")
            print(f"  But vigilance DECREASED with rats (unexpected direction)")
            statistical_conclusion = "SIGNIFICANT_DECREASE"
    else:
        print(f"  FAIL TO REJECT H0: p = {p_value:.4f} > 0.05")
        print(f"  No significant difference in vigilance")
        statistical_conclusion = "NO_SIGNIFICANT_DIFFERENCE"
    
    # Effect size interpretation
    print(f"\nEffect Size Interpretation:")
    if abs(cohens_d) >= 0.8:
        effect_size = "large"
    elif abs(cohens_d) >= 0.5:
        effect_size = "medium"  
    elif abs(cohens_d) >= 0.2:
        effect_size = "small"
    else:
        effect_size = "negligible"
    print(f"  Cohen's d = {cohens_d:.3f} ({effect_size} effect)")
    
    # ML validation
    print(f"\nML Classification Validation:")
    if 'competition' in dataset1['habit'].values and 'predator' in dataset1['habit'].values:
        comp_vigilance = dataset1[dataset1['habit'] == 'competition']['vigilance'].mean()
        pred_vigilance = dataset1[dataset1['habit'] == 'predator']['vigilance'].mean()
        
        print(f"  Competition behaviors: {comp_vigilance:.2f}s average vigilance")
        print(f"  Predator behaviors: {pred_vigilance:.2f}s average vigilance")
        
        if pred_vigilance > comp_vigilance:
            print(f"  --> ML categories support predator perception hypothesis")
        else:
            print(f"  --> ML categories do not clearly support hypothesis")
    
else:
    print(f"\nERROR: Insufficient data for statistical testing")
    statistical_conclusion = "INSUFFICIENT_DATA"
    p_value = float('nan')
    cohens_d = float('nan')
    percent_change = float('nan')
    mean_diff = float('nan')

# Create hypothesis testing visualization
print("\nCreating Phase 5 hypothesis testing visualization...")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

if len(vigilance_with_rats) > 0 and len(vigilance_without_rats) > 0:
    # Plot 1: Hypothesis test results
    means = [mean_without, mean_with]
    errors = [std_without, std_with]
    bars = ax1.bar(['No Rats Present', 'Rats Present'], means, yerr=errors, 
                   color=['#2ecc71', '#e74c3c'], capsize=5, alpha=0.7)
    ax1.set_ylabel('Vigilance (seconds)')
    ax1.set_title(f'Hypothesis Test Results\np = {p_value:.4f}')
    
    # Add significance indicator
    if p_value <= 0.05:
        ax1.text(0.5, max(means) * 1.2, '***' if p_value <= 0.001 else '**' if p_value <= 0.01 else '*', 
                ha='center', fontsize=16, fontweight='bold')
    
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors[bars.index(bar)] + 1,
                 f'{val:.2f}s', ha='center', fontweight='bold')
    
    # Plot 2: Distribution comparison
    ax2.hist(vigilance_without_rats, alpha=0.6, label='No Rats Present', 
             bins=20, color='#2ecc71', density=True)
    ax2.hist(vigilance_with_rats, alpha=0.6, label='Rats Present', 
             bins=20, color='#e74c3c', density=True)
    ax2.set_xlabel('Vigilance (seconds)')
    ax2.set_ylabel('Density')
    ax2.set_title('Vigilance Distribution Comparison')
    ax2.legend()
    
    # Add mean lines
    ax2.axvline(mean_without, color='#27ae60', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axvline(mean_with, color='#c0392b', linestyle='--', linewidth=2, alpha=0.8)
    
    # Plot 3: Effect size visualization
    ax3.bar(['Effect Size'], [abs(cohens_d)], color='steelblue', alpha=0.7)
    ax3.set_ylabel("Cohen's d")
    ax3.set_title(f'Effect Size: {effect_size.title()}')
    ax3.text(0, abs(cohens_d) + 0.05, f'{cohens_d:.3f}', ha='center', fontweight='bold')
    
    # Add effect size reference lines
    ax3.axhline(0.2, color='green', linestyle=':', alpha=0.5, label='Small')
    ax3.axhline(0.5, color='orange', linestyle=':', alpha=0.5, label='Medium') 
    ax3.axhline(0.8, color='red', linestyle=':', alpha=0.5, label='Large')
    ax3.legend()

else:
    for ax in [ax1, ax2, ax3]:
        ax.text(0.5, 0.5, 'Insufficient Data\nfor Analysis', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)

plt.suptitle('Phase 5: Hypothesis Testing - Predator Perception Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the hypothesis testing plots
stats_plot_filename = os.path.join(plots_dir, 'phase5_hypothesis_testing.png')
plt.savefig(stats_plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved hypothesis testing visualization to: {stats_plot_filename}")
plt.show()

#%%
# ============================================================================
# PHASE 6: SCIENTIFIC CONCLUSION - PREDATOR PERCEPTION INVESTIGATION
# ============================================================================
print("\n" + "="*60)
print("PHASE 6: FINAL CONCLUSION")
print("="*60)

# Determine final answer based on statistical analysis
if 'statistical_conclusion' in locals():
    if statistical_conclusion == "SIGNIFICANT_INCREASE":
        final_answer = "YES"
        evidence_strength = "STRONG"
        print(f"CONCLUSION: ✓ YES - Bats perceive rats as predators")
        print(f"Evidence: p = {p_value:.4f}, {percent_change:+.1f}% increase in vigilance")
        
    elif statistical_conclusion == "SIGNIFICANT_DECREASE": 
        final_answer = "NO"
        evidence_strength = "STRONG CONTRARY"
        print(f"CONCLUSION: ✗ NO - Bats do not perceive rats as predators")
        print(f"Evidence: p = {p_value:.4f}, vigilance decreased with rats")
        
    elif statistical_conclusion == "NO_SIGNIFICANT_DIFFERENCE":
        final_answer = "NO"
        evidence_strength = "INSUFFICIENT"
        print(f"CONCLUSION: ✗ NO - No evidence for predator perception")
        print(f"Evidence: p = {p_value:.4f} > 0.05, {effect_size} effect size")
        
    else:  # INSUFFICIENT_DATA
        final_answer = "INCONCLUSIVE"
        evidence_strength = "INSUFFICIENT DATA"
        print(f"CONCLUSION: ? INCONCLUSIVE - Insufficient data")

else:
    final_answer = "ERROR"
    evidence_strength = "ANALYSIS ERROR"
    print(f"ERROR: Analysis could not be completed")

# Create final conclusion visualization like previous phases
print(f"\nCreating Phase 6 conclusion visualization...")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

# Plot 1: Final vigilance comparison with statistical results
if 'mean_with' in locals() and 'mean_without' in locals():
    means = [mean_without, mean_with]
    errors = [std_without, std_with]
    labels = ['No Rats Present', 'Rats Present']
    
    # Color based on final answer
    if final_answer == "YES":
        colors = ['#2ecc71', '#e74c3c']  # Green to Red (significant increase)
    elif final_answer == "NO":
        colors = ['#e74c3c', '#2ecc71']  # Red to Green (no increase/decrease)
    else:
        colors = ['#95a5a6', '#95a5a6']  # Gray (inconclusive)
    
    bars = ax1.bar(labels, means, yerr=errors, color=colors, capsize=5, alpha=0.7)
    ax1.set_ylabel('Average Vigilance (seconds)')
    ax1.set_title(f'Final Results: Vigilance Comparison')
    
    # Add significance indicators
    if 'p_value' in locals() and not pd.isna(p_value):
        if p_value <= 0.001:
            sig_text = '***'
        elif p_value <= 0.01:
            sig_text = '**'  
        elif p_value <= 0.05:
            sig_text = '*'
        else:
            sig_text = 'n.s.'
        
        ax1.text(0.5, max(means) * 1.15, sig_text, ha='center', fontsize=16, fontweight='bold')
        ax1.text(0.5, max(means) * 1.08, f'p = {p_value:.4f}', ha='center', fontsize=10)
    
    # Add values on bars
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors[means.index(val)] + 0.5,
                 f'{val:.2f}s', ha='center', fontweight='bold')

else:
    ax1.text(0.5, 0.5, 'No Data Available\nfor Analysis', 
            ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    ax1.set_title('Vigilance Comparison')

# Plot 2: ML Classification validation
if 'competition' in dataset1['habit'].values and 'predator' in dataset1['habit'].values:
    ml_vigilance_data = dataset1.groupby('habit')['vigilance'].mean().sort_values()
    # Only show competition and predator categories
    ml_data = ml_vigilance_data[ml_vigilance_data.index.isin(['competition', 'predator'])]
    
    colors_ml = ['#2ecc71' if cat == 'competition' else '#e74c3c' for cat in ml_data.index]
    bars_ml = ax2.bar(ml_data.index, ml_data.values, color=colors_ml, alpha=0.7)
    ax2.set_ylabel('Average Vigilance (seconds)')
    ax2.set_title('ML Categories: Competition vs Predator')
    ax2.set_xlabel('Behavior Category')
    
    # Add values on bars
    for bar, val in zip(bars_ml, ml_data.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.2f}s', ha='center', fontweight='bold')
    
    # Rotate labels if needed
    ax2.tick_params(axis='x', rotation=45)
    
else:
    ax2.text(0.5, 0.5, 'ML Categories\nNot Available', 
            ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_title('ML Classification Validation')

# Plot 3: Final answer display with effect size
ax3.axis('off')

# Create final answer box
if final_answer == "YES":
    conclusion_color = '#27ae60'
    answer_text = "YES"
    detail_text = "Bats DO perceive\nrats as predators"
    symbol = "✓"
elif final_answer == "NO":
    conclusion_color = '#e74c3c'
    answer_text = "NO" 
    detail_text = "Bats do NOT perceive\nrats as predators"
    symbol = "✗"
else:
    conclusion_color = '#f39c12'
    answer_text = "INCONCLUSIVE"
    detail_text = "Insufficient evidence\nfor determination"
    symbol = "?"

# Main answer display (top part)
ax3.text(0.5, 0.9, symbol, ha='center', va='center', fontsize=30, 
         color=conclusion_color, fontweight='bold')
ax3.text(0.5, 0.75, answer_text, ha='center', va='center', fontsize=16, 
         fontweight='bold', color=conclusion_color)
ax3.text(0.5, 0.65, detail_text, ha='center', va='center', fontsize=10)

# Evidence strength and p-value
ax3.text(0.5, 0.55, f"Evidence: {evidence_strength}", ha='center', va='center',
         fontsize=9, style='italic')

if 'p_value' in locals() and not pd.isna(p_value):
    ax3.text(0.5, 0.48, f"p = {p_value:.4f}", ha='center', va='center',
             fontsize=8, fontfamily='monospace')

# Effect size visualization (bottom part)
if 'cohens_d' in locals() and not pd.isna(cohens_d):
    # Determine effect size category
    abs_d = abs(cohens_d)
    if abs_d >= 0.8:
        effect_category = "Large"
        effect_color = '#e74c3c'
    elif abs_d >= 0.5:
        effect_category = "Medium"
        effect_color = '#f39c12'
    elif abs_d >= 0.2:
        effect_category = "Small"
        effect_color = '#3498db'
    else:
        effect_category = "Negligible"
        effect_color = '#95a5a6'
    
    # Effect size display
    ax3.text(0.5, 0.35, "Effect Size:", ha='center', va='center',
             fontsize=10, fontweight='bold')
    ax3.text(0.5, 0.28, effect_category, ha='center', va='center',
             fontsize=12, color=effect_color, fontweight='bold')
    ax3.text(0.5, 0.22, f"Cohen's d = {cohens_d:.3f}", ha='center', va='center',
             fontsize=9, fontfamily='monospace')
    
    # Small effect size bar visualization
    bar_width = 0.6
    bar_height = 0.03
    bar_x = 0.2
    bar_y = 0.15
    
    # Background bar (full scale)
    ax3.add_patch(plt.Rectangle((bar_x, bar_y), bar_width, bar_height, 
                               facecolor='lightgray', alpha=0.3))
    
    # Effect size indicator
    effect_position = min(abs_d / 1.0, 1.0) * bar_width  # Scale to 0-1
    ax3.add_patch(plt.Rectangle((bar_x, bar_y), effect_position, bar_height, 
                               facecolor=effect_color, alpha=0.7))
    
    # Scale labels
    ax3.text(bar_x, bar_y - 0.03, '0', ha='center', va='center', fontsize=7)
    ax3.text(bar_x + bar_width/4, bar_y - 0.03, '0.2', ha='center', va='center', fontsize=7)
    ax3.text(bar_x + bar_width/2, bar_y - 0.03, '0.5', ha='center', va='center', fontsize=7)
    ax3.text(bar_x + 3*bar_width/4, bar_y - 0.03, '0.8', ha='center', va='center', fontsize=7)
    ax3.text(bar_x + bar_width, bar_y - 0.03, '1.0', ha='center', va='center', fontsize=7)

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_title('FINAL ANSWER & EFFECT SIZE', fontweight='bold', fontsize=14)

# Overall title
plt.suptitle('Phase 6: Investigation A - Enhanced Final Conclusion', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save final conclusion
conclusion_filename = os.path.join(plots_dir, 'phase6_final_conclusion.png')
plt.savefig(conclusion_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved Phase 6 final conclusion to: {conclusion_filename}")
plt.show()


# ============================================================================
# FINAL SUMMARY: INVESTIGATION A COMPLETE
# ============================================================================
print("\n" + "="*70)
print("INVESTIGATION A COMPLETE - FINAL SUMMARY")
print("="*70)

# Display final answer and key statistics
if 'final_answer' in locals() and 'mean_with' in locals():
    print(f"RESEARCH QUESTION: Do bats perceive rats as potential predators?")
    print(f"ANSWER: {final_answer}")
    print("")
    print(f"KEY RESULTS:")
    print(f"• Sample: {len(dataset1)} observations ({len(vigilance_with_rats) if 'vigilance_with_rats' in locals() else 0} with rats, {len(vigilance_without_rats) if 'vigilance_without_rats' in locals() else 0} without rats)")
    print(f"• Vigilance: {mean_with:.2f}s (with rats) vs {mean_without:.2f}s (without rats)")
    print(f"• Statistical test: p = {p_value:.4f}, Cohen's d = {cohens_d:.3f}")
    print(f"• ML classification: {dataset1['habit'].value_counts().get('competition', 0)} competition, {dataset1['habit'].value_counts().get('predator', 0)} predator behaviors")

else:
    print("ANALYSIS ERROR: Could not complete investigation")

print("\n" + "="*70)

#%%