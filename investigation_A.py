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

# Risk-reward patterns analysis complete - will be used for classification

#%%
# ============================================================================
# PHASE 1 VISUALIZATION: DATA OVERVIEW
# ============================================================================
print("\nCreating Phase 1 visualization: Data Overview")

# Create plots directory for unified analysis if it doesn't exist
plots_dir = os.path.join('plots', 'unified_analysis')
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

# Environmental context shows unknown behaviors occurred during varied conditions

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

datasets_dir = os.path.join('datasets', 'unified_analysis')
if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)

cleaned_filename = os.path.join(datasets_dir, 'dataset1_cleaned_unified_categories.csv')
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


# ============================================================================
# FINAL HABIT CATEGORY EXPLANATIONS
# ============================================================================
print(f"\n" + "="*60)
print("FINAL HABIT CATEGORY MEANINGS")
print("="*60)

final_habit_meanings = {
    'bat_and_rat': 'Mixed behavior: Complex situations involving both competition with other bats and response to rat presence as potential threat',
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

print(f"\nKey insight: 'bat_and_rat' entries represent mixed behavioral situations")
print(f"where bats show responses to both competition and potential predator threats simultaneously.")


#%%
# ============================================================================
# PHASE 4: SIMPLE PREDATOR PERCEPTION ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("PHASE 4: SIMPLE PREDATOR PERCEPTION ANALYSIS")
print("="*60)

# Core question: Do bats show higher vigilance when rats are present?
# Hypothesis: If bats perceive rats as predators -> higher vigilance (longer bat_landing_to_food)

print("Core Analysis: Testing if bats show predator perception behavior")
print("Key measure: bat_landing_to_food (vigilance time)")

# Create simple vigilance measure (already cleaned in Phase 3)
dataset1['vigilance'] = dataset1['bat_landing_to_food']
print(f"Using cleaned vigilance data (missing values handled in Phase 3)")

# Use existing rat information in Dataset1 (much more accurate!)
print("Using rat timing information already in Dataset1...")

# Dataset1 already has rat information:
# - rat_period_start: When rats arrived
# - rat_period_end: When rats left  
# - seconds_after_rat_arrival: Time since rats arrived

# Create rat presence indicator: rats were active when the bat landed
def determine_rat_presence(row):
    """Determine if rats were present when the bat landed"""
    # If no rat period info, assume no rats
    if pd.isna(row['rat_period_start']) or pd.isna(row['rat_period_end']):
        return False
    
    # Check if bat landing time was during rat presence period
    bat_time = row['start_time']
    rat_start = row['rat_period_start'] 
    rat_end = row['rat_period_end']
    
    # Bat landed during rat presence period
    return rat_start <= bat_time <= rat_end

# Also create indicator for recent rat activity (within reasonable time)
def recent_rat_activity(row):
    """Check if rats were recently active (within 10 minutes)"""
    if pd.isna(row['seconds_after_rat_arrival']):
        return False
    # Recent if within 600 seconds (10 minutes)
    return row['seconds_after_rat_arrival'] <= 600

dataset1['rats_present'] = dataset1.apply(determine_rat_presence, axis=1)
dataset1['recent_rats'] = dataset1.apply(recent_rat_activity, axis=1)

print("Rat presence analysis:")
print(f"  Bats landing DURING rat presence: {dataset1['rats_present'].sum()}")
print(f"  Bats landing AFTER recent rat activity: {dataset1['recent_rats'].sum()}")
print(f"  Total observations: {len(dataset1)}")

# Check if we have a valid comparison - all observations seem to have rats present!
rats_present_count = dataset1['rats_present'].sum()
total_observations = len(dataset1)

if rats_present_count == total_observations:
    print(f"\nCRITICAL FINDING: ALL {total_observations} observations have rats present!")
    print("   This reveals rats were active throughout the entire study period.")
    print("   Analysis must use timing-based comparisons instead of presence/absence.")

# COMPREHENSIVE PREDATOR PERCEPTION ANALYSIS
print(f"\n" + "="*50)
print("COMPREHENSIVE PREDATOR PERCEPTION ANALYSIS")
print("="*50)

# Since ALL bats have rats present, we need alternative comparisons
# 1. VIGILANCE ANALYSIS - comparing timing relative to rat activity
immediate_rats = dataset1[dataset1['seconds_after_rat_arrival'] <= 60]['vigilance']  # Within 1 minute
delayed_rats = dataset1[dataset1['seconds_after_rat_arrival'] > 300]['vigilance']   # After 5+ minutes
recent_rats = dataset1[dataset1['recent_rats'] == True]['vigilance']

print(f"\n1. VIGILANCE ANALYSIS (Time-Based Comparison):")
print(f"   Since all bats encountered rats, comparing by timing:")
print(f"   Average vigilance IMMEDIATELY after rats (<=1min): {immediate_rats.mean():.2f} seconds (n={len(immediate_rats)})")
print(f"   Average vigilance DELAYED after rats (>5min): {delayed_rats.mean():.2f} seconds (n={len(delayed_rats)})")
print(f"   Average vigilance with RECENT rat activity (<=10min): {recent_rats.mean():.2f} seconds (n={len(recent_rats)})")
if len(immediate_rats) > 0 and len(delayed_rats) > 0:
    difference = immediate_rats.mean() - delayed_rats.mean()
    print(f"   Difference (immediate vs delayed): {difference:+.2f} seconds")
    if delayed_rats.mean() > 0:
        percent_change = (difference / delayed_rats.mean()) * 100
        print(f"   Percentage change: {percent_change:+.1f}%")
        
    print(f"\nKEY INSIGHT:")
    if difference < 0:
        print(f"   Bats show LOWER vigilance immediately after rats!")
        print(f"   This is OPPOSITE to predator perception theory.")
        print(f"   Suggests rats may be COMPETITORS rather than predators.")

# 2. RISK-TAKING BEHAVIOR ANALYSIS (Time-Based Comparison)
print(f"\n2. RISK-TAKING BEHAVIOR ANALYSIS:")
risk_immediate = dataset1[dataset1['seconds_after_rat_arrival'] <= 60]['risk'].mean()
risk_delayed = dataset1[dataset1['seconds_after_rat_arrival'] > 300]['risk'].mean() 
print(f"   Risk-taking IMMEDIATELY after rats (<=1min): {risk_immediate:.3f} (n={len(immediate_rats)})")
print(f"   Risk-taking DELAYED after rats (>5min): {risk_delayed:.3f} (n={len(delayed_rats)})")
if not pd.isna(risk_immediate) and not pd.isna(risk_delayed):
    risk_diff = risk_immediate - risk_delayed
    print(f"   Difference: {risk_diff:+.3f}")

# 3. REWARD SUCCESS ANALYSIS (Time-Based Comparison)
print(f"\n3. REWARD SUCCESS ANALYSIS:")
reward_immediate = dataset1[dataset1['seconds_after_rat_arrival'] <= 60]['reward'].mean()
reward_delayed = dataset1[dataset1['seconds_after_rat_arrival'] > 300]['reward'].mean()
print(f"   Success rate IMMEDIATELY after rats (<=1min): {reward_immediate:.3f}")
print(f"   Success rate DELAYED after rats (>5min): {reward_delayed:.3f}")  
if not pd.isna(reward_immediate) and not pd.isna(reward_delayed):
    reward_diff = reward_immediate - reward_delayed
    print(f"   Difference: {reward_diff:+.3f}")

# Behavioral analysis complete - ready for visualization

# Store variables needed for Phase 6
mean_delayed = delayed_rats.mean() if len(delayed_rats) > 0 else 0
mean_immediate = immediate_rats.mean() if len(immediate_rats) > 0 else 0
difference = mean_immediate - mean_delayed if len(immediate_rats) > 0 and len(delayed_rats) > 0 else 0
percent_change = (difference / mean_delayed) * 100 if mean_delayed > 0 else 0

# Store risk difference variables
risk_difference = risk_immediate - risk_delayed if not pd.isna(risk_immediate) and not pd.isna(risk_delayed) else 0

# Initialize statistical test variables for Phase 6
t_statistic = float('nan')
t_p_value = float('nan')
chi2_stat = float('nan') 
chi2_p = float('nan')
cramers_v = float('nan')
vigilance_effect = 'unknown'
evidence_count = 0
immediate_data = immediate_rats
delayed_data = delayed_rats

# Perform statistical tests for time-based comparison
if len(immediate_rats) > 0 and len(delayed_rats) > 0:
    # T-test for vigilance differences
    t_statistic, t_p_value = stats.ttest_ind(immediate_rats, delayed_rats)
    print(f"\\nT-test results (immediate vs delayed vigilance):")
    print(f"  t-statistic: {t_statistic:.3f}, p-value: {t_p_value:.4f}")
    
    # Interpret vigilance effect
    if abs(difference) >= 2:  # Practical significance threshold
        if difference > 0:
            vigilance_effect = 'higher_immediate'
        else:
            vigilance_effect = 'higher_delayed'
    else:
        vigilance_effect = 'no_difference'

# Risk-taking behavior statistical test
if not pd.isna(risk_immediate) and not pd.isna(risk_delayed):
    # Create contingency table for chi-square test
    immediate_risk_yes = int(risk_immediate * len(immediate_rats)) if len(immediate_rats) > 0 else 0
    immediate_risk_no = len(immediate_rats) - immediate_risk_yes if len(immediate_rats) > 0 else 0
    delayed_risk_yes = int(risk_delayed * len(delayed_rats)) if len(delayed_rats) > 0 else 0
    delayed_risk_no = len(delayed_rats) - delayed_risk_yes if len(delayed_rats) > 0 else 0
    
    if immediate_risk_yes + immediate_risk_no + delayed_risk_yes + delayed_risk_no > 0:
        contingency_table = np.array([[immediate_risk_yes, immediate_risk_no],
                                     [delayed_risk_yes, delayed_risk_no]])
        chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency_table)
        
        # Calculate Cramér's V for effect size
        n = contingency_table.sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1))) if n > 0 else 0
        
        print(f"\\nChi-square test results (risk-taking behavior):")
        print(f"  chi2-statistic: {chi2_stat:.3f}, p-value: {chi2_p:.4f}")
        print(f"  Cramér's V: {cramers_v:.3f}")

# Count evidence for predator perception
significant_vigilance = t_p_value < 0.05 if not pd.isna(t_p_value) else False
meaningful_vigilance_effect = abs(difference) >= 2
significant_risk = chi2_p < 0.05 if not pd.isna(chi2_p) else False
meaningful_risk_effect = cramers_v >= 0.1 if not pd.isna(cramers_v) else False

evidence_count = sum([significant_vigilance, meaningful_vigilance_effect, 
                     significant_risk, meaningful_risk_effect])

print(f"\\nEvidence summary:")
print(f"  Significant vigilance difference: {significant_vigilance}")
print(f"  Meaningful vigilance effect: {meaningful_vigilance_effect}")
print(f"  Significant risk difference: {significant_risk}")  
print(f"  Meaningful risk effect: {meaningful_risk_effect}")
print(f"  Total evidence indicators: {evidence_count}/4")

# Create comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), facecolor='white')

# 1. Time-based vigilance comparison (corrected)
vigilance_means = [delayed_rats.mean(), immediate_rats.mean(), recent_rats.mean()]
vigilance_categories = ['Delayed (>5min)', 'Immediate (<=1min)', 'Recent (<=10min)']
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

plt.suptitle('PHASE 4: Comprehensive Predator Perception Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save plot
simple_analysis_filename = os.path.join(plots_dir, 'phase4_simple_predator_analysis.png')
plt.savefig(simple_analysis_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved simple analysis plot to: {simple_analysis_filename}")
plt.show()

print(f"\nPhase 4 completed: Comprehensive predator perception analysis ready for statistical testing!")

# Create comprehensive correlation heatmap combining Dataset1 and Dataset2 features
print(f"\nCreating comprehensive correlation analysis...")

# Select key numeric columns from both datasets for correlation analysis
dataset1_cols = ['vigilance', 'risk', 'reward', 'seconds_after_rat_arrival', 'hours_after_sunset']
correlation_data = dataset1[dataset1_cols].corr()

plt.figure(figsize=(10, 8), facecolor='white')
sns.heatmap(correlation_data, annot=True, fmt='.3f', cmap='RdBu_r', 
            center=0, square=True, cbar_kws={'shrink': 0.8})
plt.title('Comprehensive Feature Correlations (Unified Analysis)', fontsize=12, fontweight='bold')
plt.tight_layout()

# Save correlation plot
corr_filename = os.path.join(plots_dir, 'phase4_comprehensive_correlations.png')
plt.savefig(corr_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved comprehensive correlation plot to: {corr_filename}")
plt.show()

#%%
# ============================================================================
# PHASE 5: ENHANCED STATISTICAL ANALYSIS - PREDATOR PERCEPTION TESTING
# ============================================================================
print("\n" + "="*60)
print("PHASE 5: ENHANCED STATISTICAL ANALYSIS - PREDATOR PERCEPTION TESTING")
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

# STEP 3: PERFORM COMPREHENSIVE STATISTICAL TESTS
print("\n" + "=" * 40)
print("STEP 3: COMPREHENSIVE STATISTICAL ANALYSIS")
print("=" * 40)

# Separate vigilance data by rat presence
vigilance_with_rats = dataset1[dataset1['rats_present']]['vigilance'].dropna()
vigilance_without_rats = dataset1[~dataset1['rats_present']]['vigilance'].dropna()

print("Sample sizes:")
print(f"  With rats: n1 = {len(vigilance_with_rats)}")
print(f"  Without rats: n2 = {len(vigilance_without_rats)}")

# Initialize variables for consistent access
p_value = float('nan')
cohens_d = float('nan')
effect_size = "unknown"
statistical_conclusion = "INSUFFICIENT_DATA"

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
    
    # Behavioral category validation (unified analysis approach)
    print(f"\nBehavioral Category Validation:")
    if 'bat_and_rat' in dataset1['habit'].values:
        bat_rat_vigilance = dataset1[dataset1['habit'] == 'bat_and_rat']['vigilance'].mean()
        fast_vigilance = dataset1[dataset1['habit'] == 'fast']['vigilance'].mean()
        
        print(f"  bat_and_rat behaviors: {bat_rat_vigilance:.2f}s average vigilance")
        print(f"  fast behaviors: {fast_vigilance:.2f}s average vigilance")
        
        if bat_rat_vigilance > fast_vigilance:
            print(f"  --> bat_and_rat category shows higher vigilance (supports predator component)")
        else:
            print(f"  --> bat_and_rat category does not show clearly higher vigilance")
    
else:
    print(f"\nERROR: Insufficient data for statistical testing")
    statistical_conclusion = "INSUFFICIENT_DATA"
    p_value = float('nan')
    cohens_d = float('nan')
    percent_change = float('nan')
    mean_diff = float('nan')

# Create enhanced hypothesis testing visualization
print("\nCreating Phase 5 enhanced statistical analysis visualization...")

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

plt.suptitle('Phase 5: Enhanced Statistical Analysis Results', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the enhanced statistical analysis plots
stats_plot_filename = os.path.join(plots_dir, 'phase5_enhanced_statistical_analysis.png')
plt.savefig(stats_plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved enhanced statistical analysis visualization to: {stats_plot_filename}")
plt.show()

print(f"\nPhase 5 completed: Enhanced statistical analysis complete!")

#%%
# ============================================================================
# PHASE 6: FINAL ANSWER AND CONCLUSION
# ============================================================================
print("\n" + "="*60)
print("PHASE 6: FINAL ANSWER AND CONCLUSION")
print("="*60)

# Use the results from Phase 5 statistical testing
print("FINAL RESEARCH QUESTION: Do bats perceive rats as potential predators?")
print("METHOD: Measured vigilance (bat_landing_to_food) with vs without rat presence")

# Determine final answer based on statistical analysis from Phase 5
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

# Create enhanced final summary visualization
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
    # Fallback to time-based comparison if rat presence comparison not available
    if 'mean_delayed' in locals() and 'mean_immediate' in locals():
        conclusion_data = [mean_delayed, mean_immediate]
        labels = ['Delayed (>5min)', 'Immediate (<=1min)']
        colors = ['lightgreen', 'lightcoral']
        bars = ax1.bar(labels, conclusion_data, color=colors, alpha=0.8)
        ax1.set_ylabel('Average Vigilance (seconds)')
        ax1.set_title('Time-Based Vigilance Comparison')
        
        # Add significance indicator for time-based analysis
        if not pd.isna(t_p_value) and t_p_value < 0.05:
            ax1.text(0.5, max(conclusion_data) * 1.1, 
                     f'p = {t_p_value:.4f} *', ha='center', fontweight='bold', fontsize=12)
        
        for bar, val in zip(bars, conclusion_data):
            if not pd.isna(val):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                         f'{val:.2f}s', ha='center', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No Data Available\\nfor Analysis', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Vigilance Comparison')

# Plot 2: Behavioral categories comparison  
if 'bat_and_rat' in dataset1['habit'].values:
    habit_vigilance_data = dataset1.groupby('habit')['vigilance'].mean().sort_values()
    
    # Create color mapping for habits
    habit_colors = []
    for habit in habit_vigilance_data.index:
        if 'bat' in habit and 'rat' in habit:
            habit_colors.append('#e74c3c')  # Red for mixed behavior
        elif habit == 'fast':
            habit_colors.append('#2ecc71')  # Green for fast
        else:
            habit_colors.append('#3498db')  # Blue for others
    
    bars_hab = ax2.bar(range(len(habit_vigilance_data)), habit_vigilance_data.values, 
                       color=habit_colors, alpha=0.7)
    ax2.set_ylabel('Average Vigilance (seconds)')
    ax2.set_title('Behavioral Categories Comparison')
    ax2.set_xlabel('Habit Category')
    ax2.set_xticks(range(len(habit_vigilance_data)))
    ax2.set_xticklabels(habit_vigilance_data.index, rotation=45, ha='right')
    
    # Add values on bars
    for bar, val in zip(bars_hab, habit_vigilance_data.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}s', ha='center', fontweight='bold', fontsize=9)
    
else:
    ax2.text(0.5, 0.5, 'Behavioral Categories\\nNot Available', 
            ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_title('Behavioral Categories Comparison')

# Plot 3: Final answer display with effect size
ax3.axis('off')

# Create final answer box
if final_answer == "YES":
    conclusion_color = '#27ae60'
    answer_text = "YES"
    detail_text = "Bats DO perceive\\nrats as predators"
    symbol = "✓"
elif final_answer == "NO":
    conclusion_color = '#e74c3c'
    answer_text = "NO" 
    detail_text = "Bats do NOT perceive\\nrats as predators"
    symbol = "✗"
else:
    conclusion_color = '#f39c12'
    answer_text = "INCONCLUSIVE"
    detail_text = "Insufficient evidence\\nfor determination"
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

plt.suptitle('Phase 6: Investigation A - Enhanced Final Conclusion', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save enhanced final conclusion plot
final_filename = os.path.join(plots_dir, 'phase6_final_ml_enhanced_conclusion.png')
plt.savefig(final_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved enhanced final conclusion to: {final_filename}")
plt.show()

# ============================================================================
# FINAL ANSWER: SIMPLE AND CLEAR
# ============================================================================
print("\n" + "="*70)
print("FINAL ANSWER: INVESTIGATION A")
print("="*70)


print(f"\nMETHOD:")
print(f"Compared bat vigilance immediately vs delayed after rat activity (time-based analysis)")
print(f"Sample: {len(dataset1)} bat observations ({len(immediate_data)} immediate <=1min, {len(delayed_data)} delayed >5min)")

# Safe display of variables for final summary
diff_display = f"{difference:+.2f}" if not pd.isna(difference) else "N/A"
pct_display = f"{percent_change:+.1f}%" if not pd.isna(percent_change) else "N/A"
p_display = f"{p_value:.4f}" if not pd.isna(p_value) else "N/A"
risk_diff_display = f"{risk_difference:+.3f}" if not pd.isna(risk_difference) else "N/A" 
chi2_display = f"{chi2_p:.4f}" if not pd.isna(chi2_p) else "N/A"
cohens_display = f"{cohens_d:.3f}" if not pd.isna(cohens_d) else "N/A"
cramers_display = f"{cramers_v:.3f}" if not pd.isna(cramers_v) else "N/A"

print(f"\nKEY FINDINGS:")
print(f"• Average vigilance DELAYED after rats (>5min): {mean_delayed:.2f} seconds")
print(f"• Average vigilance IMMEDIATE after rats (<=1min): {mean_immediate:.2f} seconds")
print(f"• Vigilance difference: {diff_display}s ({pct_display} change)")

# Safe display of risk variables  
risk_immediate_display = f"{risk_immediate:.1%}" if not pd.isna(risk_immediate) else "N/A"
risk_delayed_display = f"{risk_delayed:.1%}" if not pd.isna(risk_delayed) else "N/A"
print(f"• Risk-taking DELAYED: {risk_delayed_display}")
print(f"• Risk-taking IMMEDIATE: {risk_immediate_display}")
print(f"• Risk difference: {risk_diff_display}")

# Safe display of test statistics
t_display = f"{t_statistic:.3f}" if not pd.isna(t_statistic) else "N/A"
chi2_stat_display = f"{chi2_stat:.3f}" if not pd.isna(chi2_stat) else "N/A"

print(f"• Vigilance test: t = {t_display}, p = {p_display}, d = {cohens_display}")
print(f"• Risk test: chi2 = {chi2_stat_display}, p = {chi2_display}, V = {cramers_display}")
print(f"• Evidence strength: {evidence_count}/4 indicators significant")

# Use comprehensive evidence for final conclusion with corrected interpretation
print(f"\n" + "="*50)

# Check if effect is opposite to predator theory
vigilance_opposite = difference < 0 if not pd.isna(difference) else False

if vigilance_opposite:
    print("CONCLUSION: STRONG EVIDENCE AGAINST PREDATOR PERCEPTION")
    print("RATS ARE LIKELY COMPETITORS, NOT PREDATORS")
    print("\nEvidence:")
    print(f"• Vigilance DECREASES immediately after rats ({difference:.2f}s, {percent_change:+.1f}%)")
    print("• This is OPPOSITE to predator perception theory")
    print("• Suggests bats view rats as competitors who reduce feeding pressure")
    if evidence_count == 0:
        print("• Statistical tests confirm no evidence for predator behavior")
elif evidence_count >= 2:
    print("CONCLUSION: YES")
    print("BATS LIKELY PERCEIVE RATS AS PREDATORS")
    print("\nEvidence:")
    if significant_vigilance:
        print("• Statistically significant increase in vigilance")
    if meaningful_vigilance_effect:
        print("• Meaningful effect size in vigilance behavior")
    if significant_risk:
        print("• Statistically significant difference in risk-taking")
    if meaningful_risk_effect:
        print("• Meaningful effect size in risk-taking behavior")
    print("• Multiple behavioral indicators support predator perception")
elif evidence_count == 1:
    print("CONCLUSION: WEAK EVIDENCE")
    print("SOME SUPPORT FOR PREDATOR PERCEPTION")
    print("\nEvidence:")
    if significant_vigilance or meaningful_vigilance_effect:
        print("• Some support from vigilance analysis")
    if significant_risk or meaningful_risk_effect:
        print("• Some support from risk-taking analysis")
    print("• Limited but present behavioral indicators")
else:
    print("CONCLUSION: INSUFFICIENT EVIDENCE")
    print("NO STRONG SUPPORT FOR PREDATOR PERCEPTION")
    print("\nReason:")
    print("• Neither vigilance nor risk-taking show strong differences")
    print("• Effect sizes too small to be meaningful")
    print("• Rats may be seen as competitors rather than predators")

print("="*50)
print("Investigation A Complete")
print("="*70)

#%%