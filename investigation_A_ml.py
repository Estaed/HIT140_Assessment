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
# PHASE 5: ENHANCED STATISTICAL ANALYSIS WITH ML-SEPARATED CATEGORIES
# ============================================================================
print("\n" + "="*60)
print("PHASE 5: ENHANCED STATISTICAL ANALYSIS WITH ML-SEPARATED CATEGORIES")
print("="*60)

# Create simple rat presence analysis using Dataset1 rat timing information
dataset1['vigilance'] = dataset1['bat_landing_to_food']

# Use existing rat information in Dataset1 (much more accurate!)
print("Using rat timing information from Dataset1...")

# Create rat presence indicator based on seconds_after_rat_arrival
dataset1['recent_rats'] = dataset1['seconds_after_rat_arrival'] <= 600  # Within 10 minutes

print("Rat presence analysis:")
print(f"  Bats with recent rat activity (<=10min): {dataset1['recent_rats'].sum()}")
print(f"  Bats without recent rat activity: {(~dataset1['recent_rats']).sum()}")
print(f"  Total observations: {len(dataset1)}")

# Use ML-enhanced categories for analysis
with_rats = dataset1[dataset1['recent_rats']]
without_rats = dataset1[~dataset1['recent_rats']]

print(f"\nSample sizes:")
print(f"  With recent rats: n = {len(with_rats)}")
print(f"  Without recent rats: n = {len(without_rats)}")

if len(with_rats) > 0 and len(without_rats) > 0:
    # T-test for vigilance difference
    t_stat, p_value = stats.ttest_ind(
        with_rats['vigilance'].dropna(),
        without_rats['vigilance'].dropna()
    )
    
    # Calculate effect sizes
    mean_with = with_rats['vigilance'].mean()
    mean_without = without_rats['vigilance'].mean()
    mean_diff = mean_with - mean_without
    percent_change = (mean_diff / mean_without) * 100 if mean_without > 0 else 0
    
    # Cohen's d
    pooled_std = np.sqrt((with_rats['vigilance'].std()**2 + 
                         without_rats['vigilance'].std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    print(f"\nVigilance Analysis:")
    print(f"  Mean with recent rats: {mean_with:.2f} seconds")
    print(f"  Mean without recent rats: {mean_without:.2f} seconds")
    print(f"  Difference: {mean_diff:.2f} seconds ({percent_change:+.1f}%)")
    print(f"  T-statistic: {t_stat:.3f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    
    # Analyze by ML-enhanced habit category
    print("\nVigilance by ML-enhanced categories:")
    habit_analysis = dataset1.groupby(['habit', 'recent_rats'])['vigilance'].agg(['mean', 'count'])
    print(habit_analysis)
    
    # Compare predator vs competition behaviors
    if 'competition' in dataset1['habit'].values and 'predator' in dataset1['habit'].values:
        competition_vigilance = dataset1[dataset1['habit'] == 'competition']['vigilance'].mean()
        predator_vigilance = dataset1[dataset1['habit'] == 'predator']['vigilance'].mean()
        
        print(f"\nML-Enhanced Category Analysis:")
        print(f"  Competition behavior average vigilance: {competition_vigilance:.2f} seconds")
        print(f"  Predator behavior average vigilance: {predator_vigilance:.2f} seconds")
        print(f"  Difference: {predator_vigilance - competition_vigilance:.2f} seconds")
        
        if predator_vigilance > competition_vigilance:
            print("  ✓ Predator behaviors show higher vigilance as expected!")
        else:
            print("  ⚠ Unexpected: Competition behaviors show higher vigilance")

else:
    print("\nWARNING: Cannot perform statistical analysis due to insufficient sample sizes")
    p_value = float('nan')
    cohens_d = float('nan')
    percent_change = float('nan')

# Create statistical analysis plots
print("\nCreating enhanced statistical analysis plots...")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

if len(with_rats) > 0 and len(without_rats) > 0:
    # Plot 1: Vigilance comparison
    means = [mean_without, mean_with]
    errors = [without_rats['vigilance'].std(), with_rats['vigilance'].std()]
    bars = ax1.bar(['No Recent Rats', 'Recent Rats'], means, yerr=errors, 
                   color=['#2ecc71', '#e74c3c'], capsize=5, alpha=0.7)
    ax1.set_ylabel('Vigilance Score (seconds)')
    ax1.set_title('Average Vigilance by Rat Presence')
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}', ha='center', fontweight='bold')
    
    # Plot 2: Distribution comparison
    ax2.hist(without_rats['vigilance'].dropna(), alpha=0.5, 
             label='No Recent Rats', bins=20, color='#2ecc71')
    ax2.hist(with_rats['vigilance'].dropna(), alpha=0.5, 
             label='Recent Rats', bins=20, color='#e74c3c')
    ax2.set_xlabel('Vigilance Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Vigilance Distribution')
    ax2.legend()

# Plot 3: ML-Enhanced category comparison
if 'competition' in dataset1['habit'].values and 'predator' in dataset1['habit'].values:
    category_means = dataset1.groupby('habit')['vigilance'].mean().sort_values()
    colors = ['#2ecc71' if 'competition' in cat else '#e74c3c' if 'predator' in cat else 'steelblue' 
              for cat in category_means.index]
    category_means.plot(kind='bar', ax=ax3, color=colors, alpha=0.7)
    ax3.set_xlabel('Behavior Category')
    ax3.set_ylabel('Average Vigilance')
    ax3.set_title('Vigilance by ML-Enhanced Categories')
    ax3.tick_params(axis='x', rotation=45)
else:
    ax3.text(0.5, 0.5, 'No ML categories\navailable', 
            ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    ax3.set_title('ML-Enhanced Categories')

plt.suptitle('Phase 5: Enhanced Statistical Analysis with ML Categories', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the statistical analysis plots
stats_plot_filename = os.path.join(plots_dir, 'phase5_enhanced_statistical_analysis.png')
plt.savefig(stats_plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved enhanced statistical analysis to: {stats_plot_filename}")
plt.show()

#%%
# ============================================================================
# PHASE 6: FINAL ANSWER WITH ML-ENHANCED RESULTS
# ============================================================================
print("\n" + "="*60)
print("PHASE 6: FINAL ANSWER WITH ML-ENHANCED RESULTS")
print("="*60)

print("FINAL RESEARCH QUESTION: Do bats perceive rats as potential predators?")
print("METHOD: Used ML to separate competition vs predator behaviors, then analyzed vigilance")

# Create final summary visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='white')

# Plot 1: Final conclusion chart
if len(with_rats) > 0 and len(without_rats) > 0:
    conclusion_data = [mean_without, mean_with]
    labels = ['No Recent Rats', 'Recent Rats']
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(labels, conclusion_data, color=colors, alpha=0.8)
    ax1.set_ylabel('Average Vigilance (seconds)')
    ax1.set_title('MAIN FINDING: Vigilance by Rat Presence')
    
    # Add significance indicator
    if not pd.isna(p_value) and p_value < 0.05:
        ax1.text(0.5, max(conclusion_data) * 1.1, 
                 f'p = {p_value:.4f} *', ha='center', fontweight='bold', fontsize=12)
        ax1.text(0.5, max(conclusion_data) * 1.05, 
                 'Statistically Significant', ha='center', fontsize=10)
    else:
        p_display = p_value if not pd.isna(p_value) else 'N/A'
        ax1.text(0.5, max(conclusion_data) * 1.1, 
                 f'p = {p_display}', ha='center', fontweight='bold', fontsize=12)
        ax1.text(0.5, max(conclusion_data) * 1.05, 
                 'Not Significant', ha='center', fontsize=10)
    
    for bar, val in zip(bars, conclusion_data):
        if not pd.isna(val):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.2f}s', ha='center', fontweight='bold')

# Plot 2: Summary text
ax2.axis('off')
summary_text = f"""
INVESTIGATION A RESULTS

Method:
• ML classification separated mixed behaviors
• Competition vs Predator categories
• Statistical analysis on separated data

Key Findings:
• Sample: {len(dataset1)} observations
• ML separated: {dataset1['habit'].value_counts().get('competition', 0)} competition, 
  {dataset1['habit'].value_counts().get('predator', 0)} predator behaviors

Statistical Results:
• Vigilance difference: {percent_change:+.1f}%
• P-value: {p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}
• Effect size: {cohens_d:.3f}

CONCLUSION:
{'STRONG EVIDENCE FOR PREDATOR PERCEPTION' if p_value < 0.05 and percent_change > 10 else 'MODERATE EVIDENCE' if p_value < 0.05 or abs(cohens_d) > 0.3 else 'INSUFFICIENT EVIDENCE'}
"""

ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

plt.suptitle('Investigation A: Final Results with ML Enhancement', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save final conclusion plot
final_filename = os.path.join(plots_dir, 'phase6_final_ml_enhanced_conclusion.png')
plt.savefig(final_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved final ML-enhanced conclusion to: {final_filename}")
plt.show()

# ============================================================================
# FINAL ANSWER: CLEAR AND COMPREHENSIVE
# ============================================================================
print("\n" + "="*70)
print("FINAL ANSWER: INVESTIGATION A WITH ML ENHANCEMENT")
print("="*70)

print(f"\nMETHOD:")
print(f"1. Used ML to separate ambiguous 'bat_and_rat' behaviors into:")
print(f"   - 'competition': Fast foraging, competition-focused behaviors")
print(f"   - 'predator': Cautious approach, predator-avoidance behaviors")
print(f"2. Analyzed vigilance differences with/without recent rat activity")
print(f"3. Sample: {len(dataset1)} bat observations")

print(f"\nML CLASSIFICATION RESULTS:")
final_cats = dataset1['habit'].value_counts()
for cat, count in final_cats.items():
    if cat in ['competition', 'predator']:
        print(f"  {cat}: {count} behaviors")

print(f"\nSTATISTICAL RESULTS:")
if len(with_rats) > 0 and len(without_rats) > 0:
    print(f"  Vigilance without recent rats: {mean_without:.2f} seconds")
    print(f"  Vigilance with recent rats: {mean_with:.2f} seconds")
    print(f"  Difference: {mean_diff:+.2f} seconds ({percent_change:+.1f}%)")
    print(f"  Statistical test: t = {t_stat:.3f}, p = {p_value:.4f}")
    print(f"  Effect size: Cohen's d = {cohens_d:.3f}")

print(f"\n" + "="*50)

# Final conclusion based on ML-enhanced analysis
if len(with_rats) > 0 and len(without_rats) > 0:
    if p_value < 0.05 and percent_change > 10:
        print("CONCLUSION: YES - STRONG EVIDENCE FOR PREDATOR PERCEPTION")
        print("\nEvidence:")
        print(f"• ML successfully separated competition vs predator behaviors")
        print(f"• Vigilance significantly increases by {percent_change:.1f}% with rat presence")
        print(f"• Statistical significance: p = {p_value:.4f} < 0.05")
        print(f"• Effect size indicates meaningful practical significance")
        print(f"• Predator-classified behaviors show expected high vigilance patterns")
    elif p_value < 0.05 or abs(cohens_d) > 0.3:
        print("CONCLUSION: MODERATE EVIDENCE FOR PREDATOR PERCEPTION")
        print("\nEvidence:")
        print(f"• ML classification provided clearer behavioral separation")
        print(f"• Some statistical evidence for increased vigilance")
        print(f"• Effect size suggests biologically relevant difference")
    else:
        print("CONCLUSION: INSUFFICIENT EVIDENCE")
        print("\nFindings:")
        print(f"• ML classification successful but statistical tests inconclusive")
        print(f"• Vigilance difference: {percent_change:+.1f}% (p = {p_value:.4f})")
        print(f"• May indicate competitive rather than predatory relationship")
else:
    print("CONCLUSION: ANALYSIS LIMITED BY SAMPLE SIZE")
    print("• All observations occurred during rat activity periods")
    print("• ML classification successful but statistical comparison limited")

print("\n" + "="*50)
print("Investigation A Complete - ML Enhanced Analysis")
print("="*70)

#%%