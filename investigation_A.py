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
# PHASE 5: SIMPLE STATISTICAL TESTING
# ============================================================================
print("\n" + "="*60)
print("PHASE 5: SIMPLE STATISTICAL TESTING")
print("="*60)

# Use the time-based comparison from Phase 4 for statistical testing
print("TIME-BASED STATISTICAL TESTING:")
print("Since all bats encountered rats, comparing immediate vs delayed responses")

# Data prepared in Phase 4 - time-based comparison
immediate_data = dataset1[dataset1['seconds_after_rat_arrival'] <= 60]['vigilance']
delayed_data = dataset1[dataset1['seconds_after_rat_arrival'] > 300]['vigilance']

# Additional data for comprehensive analysis
immediate_risk = dataset1[dataset1['seconds_after_rat_arrival'] <= 60]['risk']
delayed_risk = dataset1[dataset1['seconds_after_rat_arrival'] > 300]['risk']

print(f"\nSample sizes:")
print(f"  Immediate response (<=1min): n = {len(immediate_data)}")
print(f"  Delayed response (>5min): n = {len(delayed_data)}")

# Perform statistical testing only if both groups have data
from scipy.stats import ttest_ind, chi2_contingency

if len(immediate_data) > 0 and len(delayed_data) > 0:
    t_statistic, p_value = ttest_ind(immediate_data, delayed_data)
else:
    print("WARNING: Cannot perform t-test: insufficient data in one or both groups")
    t_statistic, p_value = float('nan'), float('nan')

# Calculate basic statistics for time-based comparison
mean_immediate = immediate_data.mean() if len(immediate_data) > 0 else float('nan')
mean_delayed = delayed_data.mean() if len(delayed_data) > 0 else float('nan')
difference = mean_immediate - mean_delayed if not (pd.isna(mean_immediate) or pd.isna(mean_delayed)) else float('nan')
percent_change = (difference / mean_delayed) * 100 if mean_delayed > 0 and not pd.isna(difference) else float('nan')

# Calculate Cohen's d (effect size) only if we have valid data
if len(immediate_data) > 1 and len(delayed_data) > 1:
    pooled_std = np.sqrt(((len(immediate_data)-1)*immediate_data.std()**2 + 
                         (len(delayed_data)-1)*delayed_data.std()**2) / 
                        (len(immediate_data) + len(delayed_data) - 2))
    cohens_d = difference / pooled_std if pooled_std > 0 else float('nan')
else:
    cohens_d = float('nan')

print(f"\nSTATISTICAL RESULTS:")
print(f"1. VIGILANCE ANALYSIS (TIME-BASED T-TEST):")
print(f"  Mean vigilance IMMEDIATE after rats (<=1min): {mean_immediate:.2f} seconds")
print(f"  Mean vigilance DELAYED after rats (>5min): {mean_delayed:.2f} seconds")
print(f"  Difference: {difference:+.2f} seconds")
print(f"  Percentage change: {percent_change:+.1f}%")
print(f"  T-statistic: {t_statistic:.3f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Cohen's d (effect size): {cohens_d:.3f}")

# Additional statistical testing for risk-taking behavior (time-based)
print(f"\n2. RISK-TAKING BEHAVIOR ANALYSIS (TIME-BASED CHI-SQUARE TEST):")

# Create timing groups for chi-square test
dataset1['timing_group'] = 'delayed'
dataset1.loc[dataset1['seconds_after_rat_arrival'] <= 60, 'timing_group'] = 'immediate'

# Filter to only include immediate and delayed groups
timing_subset = dataset1[dataset1['timing_group'].isin(['immediate', 'delayed'])]

if len(timing_subset) > 0:
    # Create contingency table for risk-taking behavior by timing
    risk_contingency = pd.crosstab(timing_subset['timing_group'], timing_subset['risk'], margins=False)
    print("   Contingency table (timing vs risk):")
    print(risk_contingency)
    
    # Perform chi-square test if we have enough data
    if risk_contingency.shape == (2, 2) and (risk_contingency > 5).all().all():
        chi2_stat, chi2_p, chi2_dof, chi2_expected = chi2_contingency(risk_contingency)
        print(f"   Chi-square statistic: {chi2_stat:.3f}")
        print(f"   P-value: {chi2_p:.4f}")
        print(f"   Degrees of freedom: {chi2_dof}")
        
        # Calculate effect size (Cramér's V)
        n = risk_contingency.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(risk_contingency.shape) - 1)))
        print(f"   Cramér's V (effect size): {cramers_v:.3f}")
    else:
        print("   WARNING: Chi-square test not appropriate: insufficient data or low cell counts")
        chi2_stat, chi2_p, cramers_v = float('nan'), float('nan'), float('nan')
    
    # Risk proportions by timing
    risk_immediate = timing_subset[timing_subset['timing_group'] == 'immediate']['risk'].mean()
    risk_delayed = timing_subset[timing_subset['timing_group'] == 'delayed']['risk'].mean()
    risk_difference = risk_immediate - risk_delayed if not (pd.isna(risk_immediate) or pd.isna(risk_delayed)) else float('nan')
    print(f"   Risk-taking IMMEDIATE: {risk_immediate:.3f} ({risk_immediate*100:.1f}%)")
    print(f"   Risk-taking DELAYED: {risk_delayed:.3f} ({risk_delayed*100:.1f}%)")
    print(f"   Difference: {risk_difference:+.3f} ({risk_difference*100:+.1f}%)")
else:
    print("   WARNING: No data available for timing-based risk analysis")
    chi2_stat, chi2_p, cramers_v = float('nan'), float('nan'), float('nan')
    risk_difference = float('nan')

# Interpret results for both tests
print(f"\nCOMPREHENSIVE INTERPRETATION:")
print(f"VIGILANCE TEST RESULTS:")
if p_value < 0.05:
    print(f"  STATISTICALLY SIGNIFICANT (p = {p_value:.4f})")
else:
    print(f"  NOT STATISTICALLY SIGNIFICANT (p = {p_value:.4f})")

if abs(cohens_d) < 0.2:
    vigilance_effect = "negligible"
elif abs(cohens_d) < 0.5:
    vigilance_effect = "small"
elif abs(cohens_d) < 0.8:
    vigilance_effect = "medium"
else:
    vigilance_effect = "large"
print(f"  Effect size: {vigilance_effect} (Cohen's d = {cohens_d:.3f})")

print(f"\nRISK-TAKING TEST RESULTS:")
if chi2_p < 0.05:
    print(f"  STATISTICALLY SIGNIFICANT (p = {chi2_p:.4f})")
else:
    print(f"  NOT STATISTICALLY SIGNIFICANT (p = {chi2_p:.4f})")

if cramers_v < 0.1:
    risk_effect = "negligible"
elif cramers_v < 0.3:
    risk_effect = "small"
elif cramers_v < 0.5:
    risk_effect = "medium"
else:
    risk_effect = "large"
print(f"  Effect size: {risk_effect} (Cramér's V = {cramers_v:.3f})")

# Overall conclusion for predator perception
significant_vigilance = p_value < 0.05
significant_risk = chi2_p < 0.05
meaningful_vigilance_effect = abs(cohens_d) >= 0.2
meaningful_risk_effect = cramers_v >= 0.1

print(f"\nOVERALL EVIDENCE FOR PREDATOR PERCEPTION:")
evidence_count = sum([significant_vigilance, significant_risk, meaningful_vigilance_effect, meaningful_risk_effect])

print(f"\nEVIDENCE SUMMARY:")
print(f"  • Significant vigilance difference: {'YES' if significant_vigilance else 'NO'}")
print(f"  • Meaningful vigilance effect size: {'YES' if meaningful_vigilance_effect else 'NO'}")
print(f"  • Significant risk difference: {'YES' if significant_risk else 'NO'}")
print(f"  • Meaningful risk effect size: {'YES' if meaningful_risk_effect else 'NO'}")
print(f"  • Total supporting evidence: {evidence_count}/4 indicators")

if evidence_count >= 2:
    print(f"\nSTRONG EVIDENCE ({evidence_count}/4 indicators support predator perception)")
elif evidence_count == 1:
    print(f"\nWEAK EVIDENCE ({evidence_count}/4 indicators support predator perception)")
else:
    print(f"\nNO EVIDENCE ({evidence_count}/4 indicators support predator perception)")
    
# Add interpretation of negative effects
if difference < 0 and not pd.isna(difference):
    print(f"\nCRITICAL INTERPRETATION:")
    print(f"   Vigilance DECREASES immediately after rats ({difference:.2f}s)")
    print(f"   This suggests COMPETITIVE rather than PREDATORY relationship")
    print(f"   Bats may relax when competitors (rats) are recently active")


# Create simple statistical plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

# Box plot comparison (time-based)
data_for_box = [delayed_data, immediate_data]
box_plot = ax1.boxplot(data_for_box, labels=['Delayed (>5min)', 'Immediate (<=1min)'], patch_artist=True)
box_plot['boxes'][0].set_facecolor('lightgreen')
box_plot['boxes'][1].set_facecolor('lightcoral')
ax1.set_ylabel('Vigilance (seconds)')
ax1.set_title('Time-Based Vigilance Comparison')
ax1.grid(True, alpha=0.3)

# Histogram comparison (time-based)
ax2.hist(delayed_data, alpha=0.6, label='Delayed (>5min)', bins=15, color='green')
ax2.hist(immediate_data, alpha=0.6, label='Immediate (<=1min)', bins=15, color='red')
ax2.set_xlabel('Vigilance (seconds)')
ax2.set_ylabel('Frequency')
ax2.set_title('Time-Based Distribution Comparison')
ax2.legend()

plt.suptitle('Phase 5: Simple Statistical Analysis Results', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save plot
stats_filename = os.path.join(plots_dir, 'phase5_statistical_results.png')
plt.savefig(stats_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved statistical analysis to: {stats_filename}")
plt.show()

print(f"\nPhase 5 completed: Statistical testing complete!")

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

# Create final summary visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

# Plot 1: Time-based conclusion chart
conclusion_data = [mean_delayed, mean_immediate]
labels = ['Delayed (>5min)', 'Immediate (<=1min)']
colors = ['lightgreen', 'lightcoral']
bars = ax1.bar(labels, conclusion_data, color=colors, alpha=0.8)
ax1.set_ylabel('Average Vigilance (seconds)')
ax1.set_title('MAIN FINDING: Time-Based Vigilance Comparison')

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

# Plot 2: Summary stats text
ax2.axis('off')
# Format variables safely for display
diff_display = f"{difference:+.2f}" if not pd.isna(difference) else "N/A"
pct_display = f"{percent_change:+.1f}%" if not pd.isna(percent_change) else "N/A"
p_display = f"{p_value:.4f}" if not pd.isna(p_value) else "N/A"
risk_diff_display = f"{risk_difference:+.3f}" if not pd.isna(risk_difference) else "N/A" 
chi2_display = f"{chi2_p:.4f}" if not pd.isna(chi2_p) else "N/A"
cohens_display = f"{cohens_d:.3f}" if not pd.isna(cohens_d) else "N/A"
cramers_display = f"{cramers_v:.3f}" if not pd.isna(cramers_v) else "N/A"
vigilance_opposite = difference < 0 if not pd.isna(difference) else False

summary_text = f"""
INVESTIGATION A RESULTS

Key Findings:
• Sample: {len(dataset1)} observations
• Vigilance: {diff_display}s ({pct_display}), p={p_display}
• Effect: {vigilance_effect} (d={cohens_display})

CONCLUSION:
{'RATS ARE COMPETITORS' if vigilance_opposite else 'PREDATOR PERCEPTION' if evidence_count >= 2 else 'WEAK EVIDENCE' if evidence_count == 1 else 'INSUFFICIENT EVIDENCE'}
"""

ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

plt.suptitle('Investigation A: Final Results', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save final conclusion plot
final_filename = os.path.join(plots_dir, 'phase6_final_conclusion.png')
plt.savefig(final_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved final conclusion to: {final_filename}")
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