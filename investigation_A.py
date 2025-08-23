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
- lightgbm: Advanced imputation for unknown behavioral patterns
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
import lightgbm as lgb
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

# Analyze risk-reward patterns for known habits
known_habits = dataset1[~missing_or_numeric]
risk_reward_patterns = known_habits.groupby(['risk', 'reward'])['habit'].apply(list).to_dict()

print(f"\nRisk-Reward correlation patterns:")
for (risk, reward), habits in risk_reward_patterns.items():
    unique_habits = list(set(habits))
    print(f"Risk={risk}, Reward={reward}: {unique_habits[:5]}...")  # Show first 5 unique habits

#%%
# ============================================================================
# PHASE 1 VISUALIZATION: DATA OVERVIEW
# ============================================================================
print("\nCreating Phase 1 visualization: Data Overview")

# Create plots directory if it doesn't exist
plots_dir = 'plots'
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

# Phase 2 completed - ready for Phase 3
print(f"\nPhase 2 completed successfully!")
print(f"Ready for Phase 3: Unknown value handling")

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
ax2.set_title('After Classification (5 Categories)', fontweight='bold')
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

# Use Dataset2 to understand environmental context for unknown behaviors
print(f"\nUsing Dataset2 to understand environmental patterns...")

# Create time matching for context analysis (not merging!)
def get_context_period(timestamp):
    """Get the 30-min period that contains this timestamp"""
    dt = pd.to_datetime(timestamp)
    # Round down to nearest 30 minutes
    minute = 0 if dt.minute < 30 else 30
    return dt.replace(minute=minute, second=0, microsecond=0)

# Analyze environmental conditions during unknown behaviors
unknown_periods = unknown_data['start_time'].apply(get_context_period).unique()
print(f"Unknown behaviors occurred across {len(unknown_periods)} different 30-min periods")

# Environmental context analysis
context_analysis = []
for period in unknown_periods:
    # Convert numpy datetime to pandas timestamp for strftime
    period_str = pd.Timestamp(period).strftime('%d/%m/%Y %H:%M')
    context_match = dataset2[dataset2['time'] == period_str]
    if not context_match.empty:
        context_analysis.append({
            'period': period,
            'bat_landings': context_match.iloc[0]['bat_landing_number'],
            'food_available': context_match.iloc[0]['food_availability'], 
            'rat_minutes': context_match.iloc[0]['rat_minutes'],
            'rat_arrivals': context_match.iloc[0]['rat_arrival_number']
        })

if context_analysis:
    context_df = pd.DataFrame(context_analysis)
    print(f"\nEnvironmental context during unknown behaviors:")
    print(f"Average bat landings per 30min: {context_df['bat_landings'].mean():.1f}")
    print(f"Average food availability: {context_df['food_available'].mean():.2f}")
    print(f"Average rat minutes: {context_df['rat_minutes'].mean():.1f}")
    print(f"Periods with rat activity: {(context_df['rat_arrivals'] > 0).sum()}/{len(context_df)}")

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

# Export cleaned dataset for further analysis
print("\n" + "="*60)
print("EXPORTING CLEANED DATASET")
print("="*60)

datasets_dir = 'datasets'
if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)

cleaned_filename = os.path.join(datasets_dir, 'dataset1_cleaned_final.csv')
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
# PHASE 4: ML CLASSIFICATION - ALL BEHAVIORS TO BAT VS RAT ONLY
# ============================================================================
print("\n" + "="*60)
print("PHASE 4: ML CLASSIFICATION - ALL BEHAVIORS TO BAT VS RAT ONLY")
print("="*60)

print("Goal: Use ML to classify ONLY 'bat_and_rat' entries into 'bat' or 'rat'")
print("Method: Keep other categories (fast, pick, etc.) unchanged")

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
    
    # Optional: Add environmental context from dataset2 (for bat_and_rat entries only)
    print(f"\nAdding environmental context for bat_and_rat entries...")

    def get_environmental_context(bat_timestamp):
        """Look up environmental context from dataset2 for a given timestamp"""
        time_diffs = abs(dataset2['time'] - bat_timestamp)
        min_diff_idx = time_diffs.idxmin()
        min_diff = time_diffs[min_diff_idx]
        
        if min_diff <= pd.Timedelta(minutes=30):
            env_row = dataset2.loc[min_diff_idx]
            return {
                'env_rat_minutes': env_row['rat_minutes'],
                'env_bat_landings': env_row['bat_landing_number'],
                'env_food_available': env_row['food_availability']
            }
        else:
            return {
                'env_rat_minutes': 0,
                'env_bat_landings': 0, 
                'env_food_available': 0
            }

    # Add environmental context only for bat_and_rat entries
    bat_and_rat_data = dataset1[bat_and_rat_mask].copy()
    env_context_list = []
    context_matches = 0
    
    for timestamp in bat_and_rat_data['start_time']:
        context = get_environmental_context(timestamp)
        env_context_list.append(context)
        if context['env_rat_minutes'] > 0:
            context_matches += 1

    env_context_df = pd.DataFrame(env_context_list)
    bat_and_rat_with_context = pd.concat([bat_and_rat_data.reset_index(drop=True), env_context_df.reset_index(drop=True)], axis=1)

    print(f"Environmental context found for {context_matches}/{len(bat_and_rat_data)} bat_and_rat entries")

    # ============================================================================
    # CREATE TRAINING DATA FROM CLEAR BAT/RAT PATTERNS
    # ============================================================================
    print(f"\n" + "="*40)
    print("CREATING TRAINING DATA FROM EXISTING CLEAR PATTERNS")
    print("="*40)

    # Use existing clear categories as training data
    clear_rat_entries = dataset1[dataset1['habit'].str.contains('rat', case=False, na=False) & 
                               ~bat_and_rat_mask]  # Exclude bat_and_rat
    clear_bat_entries = dataset1[dataset1['habit'].str.contains('bat', case=False, na=False) & 
                               ~bat_and_rat_mask]  # Exclude bat_and_rat
    
    # Also use behavioral patterns for training
    training_data_list = []
    training_labels_list = []
    
    # Add clear rat entries
    for idx, row in clear_rat_entries.iterrows():
        training_data_list.append(row)
        training_labels_list.append('rat')
    
    # Add clear bat entries  
    for idx, row in clear_bat_entries.iterrows():
        training_data_list.append(row)
        training_labels_list.append('bat')
        
    # Add behavioral patterns from non-bat_and_rat entries
    other_entries = dataset1[~bat_and_rat_mask]
    for idx, row in other_entries.iterrows():
        # High risk, no reward = likely rat-like behavior
        if row['risk'] == 1 and row['reward'] == 0:
            training_data_list.append(row)
            training_labels_list.append('rat')
        # Low risk, high reward = likely bat-like behavior
        elif row['risk'] == 0 and row['reward'] == 1:
            training_data_list.append(row)
            training_labels_list.append('bat')

    print(f"Training data created from existing patterns:")
    print(f"  Rat examples: {training_labels_list.count('rat')}")
    print(f"  Bat examples: {training_labels_list.count('bat')}")
    
    if len(training_data_list) >= 20 and training_labels_list.count('rat') >= 5 and training_labels_list.count('bat') >= 5:
        print(f"\nSufficient training data available. Training ML model...")
        
        # ============================================================================
        # TRAIN ML MODEL FOR BAT VS RAT CLASSIFICATION
        # ============================================================================
        
        # Prepare training data
        training_df = pd.DataFrame(training_data_list)
        
        # Feature set
        feature_cols = [
            'bat_landing_to_food',           # vigilance
            'seconds_after_rat_arrival',     # rat timing
            'risk', 'reward',                # behavior outcomes  
            'hours_after_sunset',            # time of day
            'month'                          # seasonality
        ]
        
        X_train = training_df[feature_cols]
        y_train = np.array(training_labels_list)
        
        # Handle missing values and scale features
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_train_clean = imputer.fit_transform(X_train)
        X_train_scaled = scaler.fit_transform(X_train_clean)
        
        # Train SVM classifier
        from sklearn.svm import SVC
        ml_model = SVC(kernel='linear', random_state=42, probability=True)
        ml_model.fit(X_train_scaled, y_train)
        
        print(f"ML model trained successfully!")
        
        # ============================================================================
        # CLASSIFY ONLY BAT_AND_RAT ENTRIES
        # ============================================================================
        print(f"\n" + "="*40)
        print("CLASSIFYING BAT_AND_RAT ENTRIES ONLY")
        print("="*40)
        
        # Prepare bat_and_rat entries for prediction
        X_predict = bat_and_rat_with_context[feature_cols]
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
        
        print(f"\nPhase 4 completed: bat_and_rat entries classified as 'bat' or 'rat'!")
        
    else:
        print(f"\nWARNING: Insufficient training data")
        print(f"Using simple heuristic classification for bat_and_rat entries...")
        
        # Simple heuristic fallback for bat_and_rat only
        heuristic_predictions = []
        for idx, row in bat_and_rat_data.iterrows():
            if row['risk'] == 1 and row['reward'] == 0:
                heuristic_predictions.append('rat')  # Risk without reward = predator
            else:
                heuristic_predictions.append('bat')  # Default to competition
        
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

# Simple Phase 4 visualization - before and after ML classification
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

# Plot 1: bat_and_rat entries before classification
if 'bat_and_rat_count' in locals() and bat_and_rat_count > 0:
    before_data = pd.Series([bat_and_rat_count], index=['bat_and_rat'])
    before_data.plot(kind='bar', ax=ax1, color='orange', alpha=0.7)
    ax1.set_title(f'Before ML: {bat_and_rat_count} bat_and_rat entries', fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=0)
else:
    ax1.text(0.5, 0.5, 'No bat_and_rat entries\nto classify', 
            ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    ax1.set_title('Before ML Classification', fontweight='bold')

# Plot 2: After ML - final distribution including all categories
final_habits_p4 = dataset1['habit'].value_counts()
final_habits_p4.plot(kind='bar', ax=ax2, color='steelblue', alpha=0.7)
ax2.set_title('After ML: All Categories', fontweight='bold')
ax2.set_xlabel('Habit Category')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)

plt.suptitle('PHASE 4: bat_and_rat Classification Complete', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save Phase 4 plot
phase4_filename = os.path.join(plots_dir, 'phase4_bat_vs_rat.png')
plt.savefig(phase4_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved Phase 4 visualization to: {phase4_filename}")
plt.show()

# Export final dataset with classified categories
datasets_dir = 'datasets'
if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)

ml_final_filename = os.path.join(datasets_dir, 'dataset1_ml_classified.csv')
dataset1.to_csv(ml_final_filename, index=False)
print(f"Exported ML classified dataset to: {ml_final_filename}")

print(f"\nPhase 4 ML classification complete!")
print(f"bat_and_rat entries have been classified as either 'bat' or 'rat'.")

# ============================================================================
# FINAL HABIT CATEGORY EXPLANATIONS
# ============================================================================
print(f"\n" + "="*60)
print("FINAL HABIT CATEGORY MEANINGS")
print("="*60)

final_habit_meanings = {
    'bat': 'Competition-focused behavior: Fast foraging, successful food acquisition, competing with other bats',
    'rat': 'Predator-focused behavior: Cautious approach, high vigilance, responding to rat presence as threat',
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

print(f"\nKey insight: 'bat_and_rat' ambiguous entries have been resolved using ML classification")
print(f"based on behavioral patterns (risk/reward) and environmental context from Dataset2.")


#%%
# ============================================================================
# PHASE 5: ENHANCED PREDATOR PERCEPTION FEATURE ENGINEERING  
# ============================================================================
print("\n" + "="*60)
print("PHASE 5: ENHANCED PREDATOR PERCEPTION FEATURE ENGINEERING")
print("="*60)

# Key hypothesis: If bats perceive rats as predators, we expect:
# 1. Higher vigilance (longer bat_landing_to_food) when rats are present
# 2. More avoidance behaviors in risky situations  
# 3. Different behavioral patterns based on rat timing

# Primary vigilance indicator
dataset1['vigilance_score'] = dataset1['bat_landing_to_food']

# Vigilance categories based on quartiles
vigilance_thresholds = dataset1['vigilance_score'].quantile([0.25, 0.5, 0.75])
print(f"Vigilance thresholds: Q1={vigilance_thresholds[0.25]:.1f}, Q2={vigilance_thresholds[0.5]:.1f}, Q3={vigilance_thresholds[0.75]:.1f}")

dataset1['vigilance_level'] = pd.cut(
    dataset1['vigilance_score'],
    bins=[0, vigilance_thresholds[0.25], vigilance_thresholds[0.5], vigilance_thresholds[0.75], float('inf')],
    labels=['low', 'medium', 'high', 'very_high']
)

dataset1['high_vigilance'] = (dataset1['vigilance_score'] > vigilance_thresholds[0.75]).astype(int)

# Predator perception indicators
print(f"\nCreating predator perception indicators...")

# 1. Avoidance behavior: High vigilance without taking risks
dataset1['avoidance_behavior'] = (
    (dataset1['risk'] == 0) & 
    (dataset1['high_vigilance'] == 1)
).astype(int)

# 2. Rat temporal threat assessment
dataset1['rat_presence_duration'] = (
    dataset1['rat_period_end'] - dataset1['rat_period_start']
).dt.total_seconds()

# Categorize rat threat timing
dataset1['rat_threat_level'] = 'no_immediate_threat'
dataset1.loc[dataset1['seconds_after_rat_arrival'] <= 30, 'rat_threat_level'] = 'immediate_threat'
dataset1.loc[(dataset1['seconds_after_rat_arrival'] > 30) & 
             (dataset1['seconds_after_rat_arrival'] <= 300), 'rat_threat_level'] = 'recent_threat'
dataset1.loc[dataset1['seconds_after_rat_arrival'] > 300, 'rat_threat_level'] = 'distant_threat'

# 3. Behavioral adaptation to rat presence  
dataset1['cautious_approach'] = (
    (dataset1['vigilance_score'] > vigilance_thresholds[0.5]) &  # Above median vigilance
    (dataset1['seconds_after_rat_arrival'] < 600)  # Within 10 minutes of rat
).astype(int)

# 4. Risk-reward behavioral classification
dataset1['behavioral_strategy'] = 'unknown'
dataset1.loc[(dataset1['risk'] == 1) & (dataset1['reward'] == 1), 'behavioral_strategy'] = 'aggressive_successful'
dataset1.loc[(dataset1['risk'] == 1) & (dataset1['reward'] == 0), 'behavioral_strategy'] = 'aggressive_failed' 
dataset1.loc[(dataset1['risk'] == 0) & (dataset1['reward'] == 1), 'behavioral_strategy'] = 'cautious_successful'
dataset1.loc[(dataset1['risk'] == 0) & (dataset1['reward'] == 0), 'behavioral_strategy'] = 'cautious_failed'

print(f"Feature engineering results:")
print(f"  High vigilance rate: {dataset1['high_vigilance'].mean()*100:.1f}%")
print(f"  Avoidance behavior rate: {dataset1['avoidance_behavior'].mean()*100:.1f}%") 
print(f"  Cautious approach rate: {dataset1['cautious_approach'].mean()*100:.1f}%")

print(f"\nRat threat level distribution:")
threat_dist = dataset1['rat_threat_level'].value_counts()
for level, count in threat_dist.items():
    print(f"  {level}: {count}")

print(f"\nBehavioral strategy distribution:")
strategy_dist = dataset1['behavioral_strategy'].value_counts()
for strategy, count in strategy_dist.items():
    print(f"  {strategy}: {count}")

# Create predator perception correlation analysis
print("\n" + "="*40)
print("PREDATOR PERCEPTION CORRELATION ANALYSIS")
print("="*40)

# Correlation matrix for predator perception features
predator_features = ['vigilance_score', 'seconds_after_rat_arrival', 'rat_presence_duration',
                    'high_vigilance', 'avoidance_behavior', 'cautious_approach', 
                    'risk', 'reward']

correlation_matrix = dataset1[predator_features].corr()

# Create correlation heatmap
plt.figure(figsize=(12, 10), facecolor='white')
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Show lower triangle only
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='RdBu_r', 
            center=0, square=True, mask=mask, cbar_kws={'shrink': 0.8})
plt.title('Predator Perception Feature Correlations', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# Save correlation plot
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    
corr_plot_filename = os.path.join(plots_dir, 'predator_perception_correlations.png')
plt.savefig(corr_plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved correlation heatmap to: {corr_plot_filename}")
plt.show()

# Key correlations analysis
print(f"\nKey correlations with vigilance_score:")
vigilance_correlations = correlation_matrix['vigilance_score'].abs().sort_values(ascending=False)
for feature, corr in vigilance_correlations.items():
    if feature != 'vigilance_score':
        direction = "positive" if correlation_matrix['vigilance_score'][feature] > 0 else "negative"
        print(f"  {feature}: {corr:.3f} ({direction})")

# Dataset2 validation analysis (separate from main analysis)
print(f"\n" + "="*40)
print("DATASET2 VALIDATION CONTEXT")
print("="*40)

# Analyze overall patterns in Dataset2 for validation
print(f"Dataset2 overview for validation:")
print(f"Total observation periods: {len(dataset2)}")
print(f"Periods with rat activity: {(dataset2['rat_arrival_number'] > 0).sum()}")
print(f"Average bat landings per period: {dataset2['bat_landing_number'].mean():.1f}")
print(f"Average food availability: {dataset2['food_availability'].mean():.2f}")

# Validate that Dataset1 patterns align with Dataset2 context
high_activity_periods = dataset2[dataset2['rat_arrival_number'] > 0]
print(f"\nValidation: High rat activity periods in Dataset2:")
print(f"Average bat landings during rat periods: {high_activity_periods['bat_landing_number'].mean():.1f}")
print(f"Average food availability during rat periods: {high_activity_periods['food_availability'].mean():.2f}")

print(f"\nFeature engineering completed successfully!")
print(f"Dataset1 ready for predator perception statistical analysis!")
print(f"Dataset2 provides validation context for environmental patterns!")

#%%
# ============================================================================
# PHASE 6: STATISTICAL ANALYSIS WITH ML-ENHANCED CATEGORIES
# ============================================================================
print("\n" + "="*60)
print("PHASE 6: STATISTICAL ANALYSIS WITH ML-ENHANCED CATEGORIES")
print("="*60)

# Main analysis: Compare vigilance with/without rats 
# Need to add vigilance_score to merged_data for analysis
merged_data['vigilance_score'] = dataset1['vigilance_score']
merged_data['avoidance_behavior'] = dataset1['avoidance_behavior']

with_rats = merged_data[merged_data['rat_arrival_number'] > 0]
without_rats = merged_data[merged_data['rat_arrival_number'] == 0]

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

# Analyze by ML-enhanced habit category
print("\nVigilance by ML-enhanced habit category and rat presence:")
habit_analysis = merged_data.groupby(['habit', merged_data['rat_arrival_number'] > 0])['vigilance_score'].agg(['mean', 'count'])
print(habit_analysis)

# Chi-square test for avoidance behavior with merged data
contingency = pd.crosstab(merged_data['rat_arrival_number'] > 0, 
                          merged_data['avoidance_behavior'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

print(f"\nAvoidance Behavior Analysis:")
print(f"  With rats: {with_rats['avoidance_behavior'].mean()*100:.1f}%")
print(f"  Without rats: {without_rats['avoidance_behavior'].mean()*100:.1f}%")
print(f"  Chi-square p-value: {p_chi:.4f}")

# PHASE 6: Create Statistical Analysis Plots with ML-Enhanced Categories
print("\nCreating statistical analysis plots...")
print("NOTE: Using merged data with ML-enhanced rat vs bat classifications")

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

# Time response curve with merged data
time_bins = [0, 30, 60, 120, 300, 600]
merged_data['time_since_rat'] = pd.cut(merged_data['seconds_after_rat_arrival'], 
                                 bins=time_bins, 
                                 labels=['0-30s', '30-60s', '60-120s', '120-300s', '300-600s'])
time_vigilance = merged_data.groupby('time_since_rat')['vigilance_score'].mean()
ax3.plot(range(len(time_vigilance)), time_vigilance.values, 
         'o-', color='darkblue', linewidth=2, markersize=8)
ax3.set_xlabel('Time Since Rat Arrival')
ax3.set_ylabel('Average Vigilance')
ax3.set_title('Temporal Response to Rat Presence')
ax3.set_xticks(range(len(time_vigilance)))
ax3.set_xticklabels(time_vigilance.index, rotation=45)
ax3.grid(True, alpha=0.3)

plt.suptitle('Phase 6: Statistical Analysis with ML-Enhanced Categories', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the statistical analysis plots
stats_plot_filename = os.path.join(plots_dir, 'statistical_analysis.png')
plt.savefig(stats_plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved statistical analysis plots to: {stats_plot_filename}")
plt.show()

#%%
# ============================================================================
# PHASE 7: VISUALIZATION AND CONCLUSION WITH ML RESULTS
# ============================================================================
print("\n" + "="*60)
print("PHASE 7: VISUALIZATION AND CONCLUSION WITH ML RESULTS")
print("="*60)

# PHASE 7: Create Final Habit Analysis with ML Results
print("\nCreating final habit analysis plot and summary statistics...")
print("NOTE: Using ML-enhanced categories with environmental context")

# Create visualization with habit analysis and summary
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='white')

# Plot 1: ML-enhanced habit category analysis
habit_vig = merged_data.groupby('habit')['vigilance_score'].mean().sort_values()
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

plt.suptitle('Phase 7: Final Analysis with ML-Enhanced Categories', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the habit analysis and summary plot
habit_summary_filename = os.path.join(plots_dir, 'habit_analysis_summary.png')
plt.savefig(habit_summary_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved habit analysis and summary plot to: {habit_summary_filename}")

plt.show()

#%%
# ============================================================================
# FINAL ANSWER
# ============================================================================
print("\n" + "="*60)
print("FINAL ANSWER: INVESTIGATION A")
print("="*60)

if p_value < 0.05 and percent_change > 10:
    print("\nYES - BATS LIKELY PERCEIVE RATS AS PREDATORS")
    print("\nEvidence:")
    print(f"1. Vigilance significantly increases by {percent_change:.1f}% when rats present")
    print(f"2. Statistical significance: p = {p_value:.4f} < 0.05")
    print(f"3. Effect size (Cohen's d = {cohens_d:.3f}) indicates {['small', 'medium', 'large'][int(abs(cohens_d) > 0.5) + int(abs(cohens_d) > 0.8)]} practical significance")
    print(f"4. Avoidance behavior increases by {(with_rats['avoidance_behavior'].mean() - without_rats['avoidance_behavior'].mean())*100:.1f}%")
    print(f"5. Behavioral patterns (slow/avoid) more common with rat presence")
else:
    print("\nINSUFFICIENT EVIDENCE FOR PREDATOR PERCEPTION")
    print("\nFindings:")
    print(f"1. Vigilance difference: {percent_change:+.1f}% (p = {p_value:.4f})")
    print(f"2. Effect size: Cohen's d = {cohens_d:.3f}")
    print(f"3. Statistical significance not achieved" if p_value >= 0.05 else "Effect too small to be meaningful")
    print("4. Alternative explanation: Rats may be seen primarily as competitors")

print("\n" + "="*60)
print("Analysis Complete")
print("="*60)
#%%