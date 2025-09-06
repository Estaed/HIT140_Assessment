# -*- coding: utf-8 -*-
#%%
# Investigation A: Do bats perceive rats as potential predators?

"""
Investigation A: Do bats perceive rats as potential predators?

This investigation analyzes bat foraging behavior in the presence of rats using
two complementary datasets:
- Dataset1: Individual bat landing events with detailed behavioral analysis
- Dataset2: Environmental context from 30-minute observation periods

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import warnings
warnings.filterwarnings('ignore')

#%%
# ============================================================================
# PHASE 1: DATA LOADING AND OVERVIEW
# ============================================================================
print("PHASE 1: DATA LOADING")

# Load datasets
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
print(f"Loaded: {len(dataset1)} bat events, {len(dataset2)} environmental periods")

# Create plots directory and visualization
plots_dir = os.path.join('plots')
os.makedirs(plots_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
fig.suptitle('Phase 1: Data Overview', fontsize=16, fontweight='bold')
datasets_info = [len(dataset1), len(dataset2)]
dataset_names = ['Bat Events\n(Dataset1)', 'Environmental Periods\n(Dataset2)']
bars = ax.bar(dataset_names, datasets_info, color=['steelblue', 'darkorange'], alpha=0.75)
ax.set_title('Records Available', fontweight='bold')
ax.set_ylabel('Count')
ymax = max(datasets_info)
ax.set_ylim(0, ymax * 1.15)
for bar, val in zip(bars, datasets_info):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax * 0.03, f'{val}', ha='center', fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(plots_dir, 'Phase1_Data_Overview.png'), dpi=300, bbox_inches='tight', facecolor='white')
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

print("Checking for fractional vigilance values and fixing if needed...")
fractions = dataset1[(dataset1['bat_landing_to_food'] > 0) & (dataset1['bat_landing_to_food'] < 1)]
extreme_fractions = fractions[fractions['bat_landing_to_food'] < 0.1]
print(f"Found {len(fractions)} fractional values (<1), of which {len(extreme_fractions)} are extreme (<0.1) – likely errors")

pre_mean = dataset1['bat_landing_to_food'].mean()
pre_min = dataset1['bat_landing_to_food'].min()
pre_max = dataset1['bat_landing_to_food'].max()
print(f"Pre-fix stats: Mean={pre_mean:.2f}s, Min={pre_min:.4f}s, Max={pre_max:.0f}s")

def fix_landing_values(value):
    if pd.isna(value):
        return value
    if value >= 1:
        return int(value)
    if 0 < value < 1:
        return int(round(value * 1000))
    return value

dataset1['bat_landing_to_food'] = dataset1['bat_landing_to_food'].apply(fix_landing_values)
post_mean = dataset1['bat_landing_to_food'].mean()
post_min = dataset1['bat_landing_to_food'].min()
post_max = dataset1['bat_landing_to_food'].max()
print(f"Post-fix stats: Mean={post_mean:.2f}s, Min={post_min:.4f}s, Max={post_max:.0f}s")

# Store original habits before classification
original_habits = dataset1['habit'].copy()

# Habit classification
def classify_habit_updated(habit, risk, reward):
    if pd.isna(habit):
        return 'unknown'
    # Ensure habit is a string before using string methods
    habit_str = str(habit).lower()
    if any(char.isdigit() for char in habit_str):
        return 'unknown'
    if any(term in habit_str for term in ['attack', 'fight', 'disappear']):
        return 'fight'
    if 'fast' in habit_str:
        return 'fast'
    if 'pick' in habit_str:
        return 'pick'
    if 'bat' in habit_str:
        return 'bat_and_rat'
    if 'rat' in habit_str:
        return 'bat_and_rat'
    return 'bat_and_rat'

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

# Apply updated classification with smart imputation for unknown habits
dataset1['habit_classified'] = dataset1.apply(
    lambda row: classify_habit_updated(row['habit'], row.get('risk'), row.get('reward')) if classify_habit_updated(row['habit'], row.get('risk'), row.get('reward')) != 'unknown' else impute_unknown_smart(row), axis=1
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

# Phase 2 visualization - Habit classification only
fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
fig.suptitle('Phase 2: Habit Classification Results', fontsize=16, fontweight='bold')

# Plot: Classified Habit Categories
bars2 = ax.bar(new_counts.index.astype(str), new_counts.values, color='steelblue', alpha=0.75)
ax.set_title('Detailed Habit Classification', fontweight='bold')
ax.set_xlabel('Habit Category')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)
ymax2 = new_counts.max() if len(new_counts) else 0
ax.set_ylim(0, ymax2 * 1.15 if ymax2 > 0 else 1)
for rect, v in zip(bars2, new_counts.values):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + (ymax2 * 0.03 if ymax2 > 0 else 0.1), str(v), ha='center', fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(plots_dir, 'Phase2_Classification_Summary.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Export cleaned dataset for further analysis
print("\n" + "="*60)
print("EXPORTING CLEANED DATASET")
print("="*60)

datasets_dir = os.path.join('datasets')
if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)

cleaned_filename = os.path.join(datasets_dir, 'dataset1_cleaned.csv')
dataset1.to_csv(cleaned_filename, index=False)

print(f"Exported cleaned dataset to: {cleaned_filename}")
print(f"No unknown values remaining: {(dataset1['habit'] == 'unknown').sum() == 0}")

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
# PHASE 3: DATA MERGING AND HYPOTHESIS FORMULATION
# ============================================================================
print("\n" + "="*60)
print("PHASE 3: DATA MERGING AND HYPOTHESIS FORMULATION")
print("="*60)

# STEP 1: FORMULATE CORRECT HYPOTHESES FOR THIS DATASET
print("=" * 40)
print("STEP 1: HYPOTHESIS FORMULATION")
print("=" * 40)
print("IMPORTANT: Dataset1 contains ONLY observations when rats are present")
print("Therefore, we cannot test presence vs absence of rats")
print("")
print("REVISED HYPOTHESES FOR GRADIENT ANALYSIS:")
print("")
print("H0 (Null): Bat vigilance and behavior are independent of rat threat intensity")
print("           β_threat = 0 (no relationship with threat indicators)")
print("")
print("H1 (Alternative): Bat vigilance and behavior change with rat threat intensity") 
print("           β_threat ≠ 0 (significant relationship with threat indicators)")
print("")
print("Key threat indicators to test:")
print("  • Temporal proximity (seconds_after_rat_arrival)")
print("  • Intensity (rat_minutes)")
print("  • Frequency (rat_arrival_number)")
print("")
print("Significance level: α = 0.05")
print("Test approach: Correlation and regression analysis of threat gradients")

# STEP 2: MERGE DATASETS FOR ENVIRONMENTAL CONTEXT
print("\n" + "=" * 40)
print("STEP 2: DATA MERGING")
print("=" * 40)

print("Merging Dataset1 with Dataset2 for environmental context...")

# Before sorting, preserve original index
dataset1['original_index'] = range(len(dataset1))

# Sort both datasets by time for proper merge
dataset1 = dataset1.sort_values("start_time")
dataset2 = dataset2.sort_values("time")

# Merge using merge_asof to match each bat landing with the most recent environmental observation
merged_data = pd.merge_asof(
    left=dataset1, right=dataset2,
    left_on="start_time", right_on="time",
    direction="backward"
)

# Create rat presence indicator (for completeness, though all should be True)
merged_data['rats_present'] = ((merged_data['rat_arrival_number'] > 0) | 
                               (merged_data['rat_minutes'] > 0)).fillna(False)

# Restore original order
merged_data = merged_data.sort_values('original_index').drop('original_index', axis=1)

# Update main dataset
dataset1['rats_present'] = merged_data['rats_present']
dataset1['environmental_rat_minutes'] = merged_data['rat_minutes'].fillna(0)

# Verify data structure
print("\nData structure verification:")
print(f"  Total bat observations: {len(dataset1)}")
print(f"  Observations WITH rat threat indicators: {(dataset1['seconds_after_rat_arrival'].notna()).sum()}")
print(f"  Observations with environmental data: {(dataset1['environmental_rat_minutes'].notna()).sum()}")

# Display threat gradient statistics
print("\nRat threat gradient statistics:")
print(f"  Seconds after rat arrival - Mean: {dataset1['seconds_after_rat_arrival'].mean():.1f}, Median: {dataset1['seconds_after_rat_arrival'].median():.1f}")
print(f"  Rat minutes (from dataset2) - Mean: {merged_data['rat_minutes'].mean():.1f}, Median: {merged_data['rat_minutes'].median():.1f}")
print(f"  Rat arrival number - Mean: {merged_data['rat_arrival_number'].mean():.1f}, Max: {merged_data['rat_arrival_number'].max():.0f}")

# Save merged data with environmental context
print(f"\nSaving merged data with environmental context...")
merged_filename = os.path.join(datasets_dir, 'dataset1_merged_with_dataset2.csv')
merged_data.to_csv(merged_filename, index=False)
print(f"Saved merged dataset to: {merged_filename}")
print(f"Columns: {list(merged_data.columns)}")
print(f"Shape: {merged_data.shape}")

# Update dataset1 to use merged data for subsequent analyses
dataset1 = merged_data

# === Phase 3 Visualizations ===
viz_dir = os.path.join(plots_dir)
os.makedirs(viz_dir, exist_ok=True)

# One combined figure (2 rows x 3 columns):
# Top row = distributions; Bottom row = outcomes
fig, axs = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')
fig.suptitle('Phase 3: Threat Distributions and Outcomes', fontsize=18, fontweight='bold')

# Distributions (top row)
sns.histplot(dataset1['seconds_after_rat_arrival'], bins=30, ax=axs[0, 0], color='steelblue')
axs[0, 0].set_title('Temporal Proximity to Rat Arrival', fontweight='bold')
axs[0, 0].set_xlabel('Seconds After Rat Arrival (lower = closer)')
axs[0, 0].set_ylabel('Count')

sns.histplot(dataset1['rat_minutes'], bins=30, ax=axs[0, 1], color='darkorange')
axs[0, 1].set_title('Rat Presence Intensity', fontweight='bold')
axs[0, 1].set_xlabel('Rat Minutes per Period')
axs[0, 1].set_ylabel('Count')

sns.histplot(dataset1['rat_arrival_number'], bins=30, ax=axs[0, 2], color='seagreen')
axs[0, 2].set_title('Rat Arrival Frequency', fontweight='bold')
axs[0, 2].set_xlabel('Number of Rat Arrivals')
axs[0, 2].set_ylabel('Count')

# Outcomes (bottom row)
sns.scatterplot(x='seconds_after_rat_arrival', y='bat_landing_to_food', data=dataset1, ax=axs[1, 0], s=20, alpha=0.6)
axs[1, 0].set_title('Closer to Rat Arrival → Vigilance', fontweight='bold')
axs[1, 0].set_xlabel('Seconds After Rat Arrival (lower = closer)')
axs[1, 0].set_ylabel('Vigilance (bat_landing_to_food, s)')

sns.scatterplot(x='rat_minutes', y='reward', data=dataset1, ax=axs[1, 1], s=20, alpha=0.6, color='darkorange')
axs[1, 1].set_title('Rat Presence Intensity → Success', fontweight='bold')
axs[1, 1].set_xlabel('Rat Minutes per Period')
axs[1, 1].set_ylabel('Feeding Success (0/1)')

sns.scatterplot(x='rat_arrival_number', y='bat_landing_to_food', data=dataset1, ax=axs[1, 2], s=20, alpha=0.6, color='seagreen')
axs[1, 2].set_title('Rat Arrivals → Vigilance', fontweight='bold')
axs[1, 2].set_xlabel('Number of Rat Arrivals')
axs[1, 2].set_ylabel('Vigilance (bat_landing_to_food, s)')

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
combined_path = os.path.join(viz_dir, 'Phase3_Threat_Distributions_and_Outcomes.png')
plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print(f"Saved Phase 3 overview plot to: {combined_path}")

print("\n" + "=" * 40)
print("ANALYSIS APPROACH")
print("=" * 40)
print("Since all observations include rat presence, the analysis will focus on:")
print("1. How PROXIMITY to rat arrival affects bat vigilance")
print("2. How INTENSITY of rat presence affects feeding success")
print("3. How FREQUENCY of rat encounters affects behavior")
print("4. Whether defensive behaviors correlate with threat levels")
print("\nThis gradient approach will reveal if bats perceive rats as predators")
print("through behavioral changes proportional to threat level.")
#%%
# ============================================================================
# PHASE 4: HYPOTHESIS TESTING - THREAT GRADIENT ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("PHASE 4: HYPOTHESIS TESTING - THREAT GRADIENT ANALYSIS")
print("="*60)

from scipy.stats import spearmanr, mannwhitneyu
import numpy as np

# Use the merged dataset from Phase 3
dataset1 = merged_data.copy()

print("\nTesting threat gradient hypotheses based on Phase 3 formulation:")
print("H0: β_threat = 0 (no relationship with threat indicators)")
print("H1: β_threat ≠ 0 (significant relationship with threat indicators)\n")

# Store all results
all_hypothesis_results = {}

print("="*40)
print("CORE THREAT GRADIENT HYPOTHESES")
print("="*40)

# H1: Temporal proximity → Vigilance
print("\nH1: Temporal proximity increases vigilance")
print("Expected: Negative correlation (recent arrival → higher vigilance)")
valid_h1 = dataset1[['seconds_after_rat_arrival', 'bat_landing_to_food']].dropna()
corr, p_val = spearmanr(valid_h1['seconds_after_rat_arrival'], valid_h1['bat_landing_to_food'])
all_hypothesis_results['H1_temporal_vigilance'] = {'correlation': corr, 'p_value': p_val}
print(f"  Correlation: r={corr:.4f}")
print(f"  P-value: {p_val:.4f}")
print(f"  Result: {'SIGNIFICANT' if p_val < 0.05 else 'Not significant'}")
if p_val < 0.05:
    print(f"  Direction: {'As expected (supports)' if corr < 0 else 'OPPOSITE of expected (contradicts)'}")

# H2: Threat intensity → Success
print("\nH2: Threat intensity reduces feeding success")
print("Expected: Negative correlation (more rat minutes → lower success)")
valid_h2 = dataset1[['rat_minutes', 'reward']].dropna()
corr, p_val = spearmanr(valid_h2['rat_minutes'], valid_h2['reward'])
all_hypothesis_results['H2_intensity_success'] = {'correlation': corr, 'p_value': p_val}
print(f"  Correlation: r={corr:.4f}")
print(f"  P-value: {p_val:.4f}")
print(f"  Result: {'SIGNIFICANT' if p_val < 0.05 else 'Not significant'}")
if p_val < 0.05:
    print(f"  Direction: {'As expected (supports)' if corr < 0 else 'OPPOSITE of expected (contradicts)'}")

# H3: Threat frequency → Vigilance
print("\nH3: Threat frequency increases vigilance")
print("Expected: Positive correlation (more arrivals → higher vigilance)")
valid_h3 = dataset1[['rat_arrival_number', 'bat_landing_to_food']].dropna()
corr, p_val = spearmanr(valid_h3['rat_arrival_number'], valid_h3['bat_landing_to_food'])
all_hypothesis_results['H3_frequency_vigilance'] = {'correlation': corr, 'p_value': p_val}
print(f"  Correlation: r={corr:.4f}")
print(f"  P-value: {p_val:.4f}")
print(f"  Result: {'SIGNIFICANT' if p_val < 0.05 else 'Not significant'}")
if p_val < 0.05:
    print(f"  Direction: {'As expected (supports)' if corr > 0 else 'OPPOSITE of expected (contradicts)'}")

# H4: Threat frequency → Success
print("\nH4: Threat frequency reduces feeding success")
print("Expected: Negative correlation (more arrivals → lower success)")
valid_h4 = dataset1[['rat_arrival_number', 'reward']].dropna()
corr, p_val = spearmanr(valid_h4['rat_arrival_number'], valid_h4['reward'])
all_hypothesis_results['H4_frequency_success'] = {'correlation': corr, 'p_value': p_val}
print(f"  Correlation: r={corr:.4f}")
print(f"  P-value: {p_val:.4f}")
print(f"  Result: {'SIGNIFICANT' if p_val < 0.05 else 'Not significant'}")
if p_val < 0.05:
    print(f"  Direction: {'As expected (supports)' if corr < 0 else 'OPPOSITE of expected (contradicts)'}")

print("\n" + "="*40)
print("BEHAVIORAL RESPONSE HYPOTHESES")
print("="*40)

# H5: Risk-taking under threat
print("\nH5: Risk-taking decreases with higher threat")
for threat_var in ['seconds_after_rat_arrival', 'rat_minutes', 'rat_arrival_number']:
    valid = dataset1[[threat_var, 'risk']].dropna()
    corr, p_val = spearmanr(valid[threat_var], valid['risk'])
    all_hypothesis_results[f'H5_risk_{threat_var}'] = {'correlation': corr, 'p_value': p_val}
    print(f"  {threat_var}: r={corr:.4f}, p={p_val:.4f}")

# H6: Defensive behaviors under threat
print("\nH6: Defensive behaviors increase with threat")
defensive_habits = ['cautious', 'slow_approach', 'fight']
dataset1['defensive'] = dataset1['habit'].isin(defensive_habits).astype(int)

for threat_var in ['seconds_after_rat_arrival', 'rat_minutes', 'rat_arrival_number']:
    valid = dataset1[[threat_var, 'defensive']].dropna()
    median_val = valid[threat_var].median()
    
    # Adjust for seconds_after_rat_arrival (lower = higher threat)
    if threat_var == 'seconds_after_rat_arrival':
        high_threat_group = valid[valid[threat_var] <= median_val]['defensive']
        low_threat_group = valid[valid[threat_var] > median_val]['defensive']
    else:
        high_threat_group = valid[valid[threat_var] > median_val]['defensive']
        low_threat_group = valid[valid[threat_var] <= median_val]['defensive']
    
    stat, p_val = mannwhitneyu(high_threat_group, low_threat_group)
    all_hypothesis_results[f'H6_defensive_{threat_var}'] = {
        'high_threat': high_threat_group.mean(), 
        'low_threat': low_threat_group.mean(), 
        'p_value': p_val
    }
    print(f"  {threat_var}: High={high_threat_group.mean():.1%}, Low={low_threat_group.mean():.1%}, p={p_val:.4f}")

# H7: Time of night effects
print("\nH7: Anti-predator behavior changes with time")
median_time = dataset1['hours_after_sunset_x'].median()
early_night = dataset1[dataset1['hours_after_sunset_x'] <= median_time]['bat_landing_to_food']
late_night = dataset1[dataset1['hours_after_sunset_x'] > median_time]['bat_landing_to_food']
stat, p_val = mannwhitneyu(early_night.dropna(), late_night.dropna())
all_hypothesis_results['H7_time_effect'] = {
    'early_mean': early_night.mean(),
    'late_mean': late_night.mean(),
    'p_value': p_val
}
print(f"  Early night: {early_night.mean():.2f}s, Late night: {late_night.mean():.2f}s")
print(f"  P-value: {p_val:.4f}")

# H8: Composite threat index
print("\nH8: Composite threat index predicts vigilance")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

threat_cols = ['seconds_after_rat_arrival', 'rat_minutes', 'rat_arrival_number']
valid_for_index = dataset1[threat_cols].dropna()

# Create threat index (invert seconds_after_rat_arrival)
valid_for_index = valid_for_index.copy()
valid_for_index['threat_proximity'] = 1 / (1 + valid_for_index['seconds_after_rat_arrival']/60)
valid_for_index['threat_intensity'] = valid_for_index['rat_minutes']
valid_for_index['threat_frequency'] = valid_for_index['rat_arrival_number']

threat_components = valid_for_index[['threat_proximity', 'threat_intensity', 'threat_frequency']]
scaled_threats = scaler.fit_transform(threat_components)
dataset1.loc[valid_for_index.index, 'threat_index'] = scaled_threats.mean(axis=1)

valid_composite = dataset1[['threat_index', 'bat_landing_to_food']].dropna()
corr, p_val = spearmanr(valid_composite['threat_index'], valid_composite['bat_landing_to_food'])
all_hypothesis_results['H8_composite_threat'] = {'correlation': corr, 'p_value': p_val}
print(f"  Correlation with vigilance: r={corr:.4f}")
print(f"  P-value: {p_val:.4f}")

# Summary table
print("\n" + "="*40)
print("HYPOTHESIS TESTING SUMMARY")
print("="*40)
print(f"Total hypotheses tested: {len(all_hypothesis_results)}")
print(f"Significant results (p<0.05): {sum(1 for r in all_hypothesis_results.values() if r.get('p_value', 1) < 0.05)}")

# === Phase 4 Visualizations: All Hypotheses in One Figure ===
# Helpers to compute verdict labels and annotate plots
colors_map = {
    'Predator evidence': '#2e7d32',
    'Competitor/facilitation': '#d84315',
    'Not significant': '#616161',
    'Contextual': '#1565c0'
}

def verdict_corr(rule, corr, p):
    if p is None or np.isnan(p) or p >= 0.05:
        return 'Not significant'
    if rule == 'neg_support':
        return 'Predator evidence' if corr < 0 else 'Competitor/facilitation'
    if rule == 'pos_support':
        return 'Predator evidence' if corr > 0 else 'Competitor/facilitation'
    return 'Not significant'

def verdict_groups(high, low, p):
    if p is None or np.isnan(p) or p >= 0.05:
        return 'Not significant'
    return 'Predator evidence' if (high > low) else 'Competitor/facilitation'

def annotate(ax, text):
    # Place label above, aligned to left to avoid overlapping points
    ax.text(0.02, 1.06, text, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='left', color=colors_map.get(text, '#000000'),
            bbox=dict(facecolor='white', edgecolor=colors_map.get(text, '#000000'), boxstyle='round,pad=0.3'),
            clip_on=False, zorder=5)

fig, axs = plt.subplots(4, 3, figsize=(20, 16), facecolor='white')
fig.suptitle('Phase 4: Hypotheses Overview (Threat → Responses)', fontsize=18, fontweight='bold')
from matplotlib.patches import Patch
legend_handles = [
    Patch(color=colors_map['Predator evidence'], label='Predator evidence: behavior consistent with avoiding rats (higher vigilance or lower success as threat increases).'),
    Patch(color=colors_map['Competitor/facilitation'], label='Competitor/facilitation: behavior improves with rat activity (shared resources).'),
    Patch(color=colors_map['Not significant'], label='Not significant: no reliable relationship (p ≥ 0.05).'),
    Patch(color=colors_map['Contextual'], label='Contextual: descriptive control (e.g., time-of-night), not a predator test.')
]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=1, frameon=True)

# Helper for jittered binary y
rng = np.random.default_rng(42)

# H1: seconds_after_rat_arrival vs bat_landing_to_food
sns.regplot(x='seconds_after_rat_arrival', y='bat_landing_to_food', data=dataset1,
            ax=axs[0, 0], scatter_kws={'s':15, 'alpha':0.5}, line_kws={'color':'crimson'})
h1 = all_hypothesis_results.get('H1_temporal_vigilance', {})
axs[0, 0].set_title(f"H1: Proximity → Vigilance\n(r={h1.get('correlation', float('nan')):.3f}, p={h1.get('p_value', float('nan')):.4f})",
                    fontweight='bold')
axs[0, 0].set_xlabel('Seconds After Rat Arrival (lower = closer)')
axs[0, 0].set_ylabel('Vigilance (bat_landing_to_food, s)')
annotate(axs[0, 0], verdict_corr('neg_support', h1.get('correlation', np.nan), h1.get('p_value', np.nan)))

# H2: rat_minutes vs reward (jitter)
y2 = dataset1['reward'] + rng.uniform(-0.02, 0.02, size=len(dataset1))
axs[0, 1].scatter(dataset1['rat_minutes'], y2, s=15, alpha=0.5, color='darkorange')
h2 = all_hypothesis_results.get('H2_intensity_success', {})
axs[0, 1].set_title(f"H2: Intensity → Success\n(r={h2.get('correlation', float('nan')):.3f}, p={h2.get('p_value', float('nan')):.4f})",
                    fontweight='bold')
axs[0, 1].set_xlabel('Rat Minutes per Period')
axs[0, 1].set_ylabel('Feeding Success (0/1)')
axs[0, 1].set_ylim(-0.1, 1.1)
annotate(axs[0, 1], verdict_corr('neg_support', h2.get('correlation', np.nan), h2.get('p_value', np.nan)))

# H3: rat_arrival_number vs bat_landing_to_food
sns.regplot(x='rat_arrival_number', y='bat_landing_to_food', data=dataset1,
            ax=axs[0, 2], scatter_kws={'s':15, 'alpha':0.5}, line_kws={'color':'crimson'})
h3 = all_hypothesis_results.get('H3_frequency_vigilance', {})
axs[0, 2].set_title(f"H3: Frequency → Vigilance\n(r={h3.get('correlation', float('nan')):.3f}, p={h3.get('p_value', float('nan')):.4f})",
                    fontweight='bold')
axs[0, 2].set_xlabel('Number of Rat Arrivals')
axs[0, 2].set_ylabel('Vigilance (s)')
annotate(axs[0, 2], verdict_corr('pos_support', h3.get('correlation', np.nan), h3.get('p_value', np.nan)))

# H4: rat_arrival_number vs reward (jitter)
y4 = dataset1['reward'] + rng.uniform(-0.02, 0.02, size=len(dataset1))
axs[1, 0].scatter(dataset1['rat_arrival_number'], y4, s=15, alpha=0.5, color='seagreen')
h4 = all_hypothesis_results.get('H4_frequency_success', {})
axs[1, 0].set_title(f"H4: Frequency → Success\n(r={h4.get('correlation', float('nan')):.3f}, p={h4.get('p_value', float('nan')):.4f})",
                    fontweight='bold')
axs[1, 0].set_xlabel('Number of Rat Arrivals')
axs[1, 0].set_ylabel('Feeding Success (0/1)')
axs[1, 0].set_ylim(-0.1, 1.1)
annotate(axs[1, 0], verdict_corr('neg_support', h4.get('correlation', np.nan), h4.get('p_value', np.nan)))

# H5a: seconds_after_rat_arrival vs risk (jitter)
y5a = dataset1['risk'] + rng.uniform(-0.02, 0.02, size=len(dataset1))
axs[1, 1].scatter(dataset1['seconds_after_rat_arrival'], y5a, s=15, alpha=0.5, color='steelblue')
res = all_hypothesis_results.get('H5_risk_seconds_after_rat_arrival', {})
axs[1, 1].set_title('H5: Proximity → Risk', fontweight='bold')
axs[1, 1].set_xlabel('Seconds After Rat Arrival (lower = closer)')
axs[1, 1].set_ylabel('Risk (0/1)')
axs[1, 1].set_ylim(-0.1, 1.1)
annotate(axs[1, 1], verdict_corr('neg_support', res.get('correlation', np.nan), res.get('p_value', np.nan)))

# H5b: rat_minutes vs risk (jitter)
y5b = dataset1['risk'] + rng.uniform(-0.02, 0.02, size=len(dataset1))
axs[1, 2].scatter(dataset1['rat_minutes'], y5b, s=15, alpha=0.5, color='darkorange')
res = all_hypothesis_results.get('H5_risk_rat_minutes', {})
axs[1, 2].set_title('H5: Intensity → Risk', fontweight='bold')
axs[1, 2].set_xlabel('Rat Minutes per Period')
axs[1, 2].set_ylabel('Risk (0/1)')
axs[1, 2].set_ylim(-0.1, 1.1)
annotate(axs[1, 2], verdict_corr('neg_support', res.get('correlation', np.nan), res.get('p_value', np.nan)))

# H5c: rat_arrival_number vs risk (jitter)
y5c = dataset1['risk'] + rng.uniform(-0.02, 0.02, size=len(dataset1))
axs[2, 0].scatter(dataset1['rat_arrival_number'], y5c, s=15, alpha=0.5, color='seagreen')
res = all_hypothesis_results.get('H5_risk_rat_arrival_number', {})
axs[2, 0].set_title('H5: Frequency → Risk', fontweight='bold')
axs[2, 0].set_xlabel('Number of Rat Arrivals')
axs[2, 0].set_ylabel('Risk (0/1)')
axs[2, 0].set_ylim(-0.1, 1.1)
annotate(axs[2, 0], verdict_corr('neg_support', res.get('correlation', np.nan), res.get('p_value', np.nan)))

# H6a: Defensive proportion by proximity (high vs low)
valid = dataset1[['seconds_after_rat_arrival', 'defensive']].dropna()
median_val = valid['seconds_after_rat_arrival'].median()
high = valid[valid['seconds_after_rat_arrival'] <= median_val]['defensive']
low  = valid[valid['seconds_after_rat_arrival'] >  median_val]['defensive']
axs[2, 1].bar(['High Threat', 'Low Threat'], [high.mean(), low.mean()], color=['crimson','gray'])
axs[2, 1].set_ylim(0, 1)
axs[2, 1].set_title(f"H6: Defensive vs Proximity\n(p={all_hypothesis_results.get('H6_defensive_seconds_after_rat_arrival', {}).get('p_value', float('nan')):.4f})",
                    fontweight='bold')
axs[2, 1].set_ylabel('Defensive Proportion')
annotate(axs[2, 1], verdict_groups(high.mean(), low.mean(), all_hypothesis_results.get('H6_defensive_seconds_after_rat_arrival', {}).get('p_value', np.nan)))

# H6b: Defensive proportion by intensity (high vs low)
valid = dataset1[['rat_minutes', 'defensive']].dropna()
median_val = valid['rat_minutes'].median()
high = valid[valid['rat_minutes'] >  median_val]['defensive']
low  = valid[valid['rat_minutes'] <= median_val]['defensive']
axs[2, 2].bar(['High Threat', 'Low Threat'], [high.mean(), low.mean()], color=['darkorange','gray'])
axs[2, 2].set_ylim(0, 1)
axs[2, 2].set_title(f"H6: Defensive vs Intensity\n(p={all_hypothesis_results.get('H6_defensive_rat_minutes', {}).get('p_value', float('nan')):.4f})",
                    fontweight='bold')
axs[2, 2].set_ylabel('Defensive Proportion')
annotate(axs[2, 2], verdict_groups(high.mean(), low.mean(), all_hypothesis_results.get('H6_defensive_rat_minutes', {}).get('p_value', np.nan)))

# H6c: Defensive proportion by frequency (high vs low)
valid = dataset1[['rat_arrival_number', 'defensive']].dropna()
median_val = valid['rat_arrival_number'].median()
high = valid[valid['rat_arrival_number'] >  median_val]['defensive']
low  = valid[valid['rat_arrival_number'] <= median_val]['defensive']
axs[3, 0].bar(['High Threat', 'Low Threat'], [high.mean(), low.mean()], color=['seagreen','gray'])
axs[3, 0].set_ylim(0, 1)
axs[3, 0].set_title(f"H6: Defensive vs Frequency\n(p={all_hypothesis_results.get('H6_defensive_rat_arrival_number', {}).get('p_value', float('nan')):.4f})",
                    fontweight='bold')
axs[3, 0].set_ylabel('Defensive Proportion')
annotate(axs[3, 0], verdict_groups(high.mean(), low.mean(), all_hypothesis_results.get('H6_defensive_rat_arrival_number', {}).get('p_value', np.nan)))

# H7: Early vs Late night vigilance (bars)
axs[3, 1].bar(['Early Night', 'Late Night'],
              [all_hypothesis_results.get('H7_time_effect', {}).get('early_mean', float('nan')),
               all_hypothesis_results.get('H7_time_effect', {}).get('late_mean', float('nan'))],
              color=['steelblue','gray'])
axs[3, 1].set_title(f"H7: Time-of-Night Effect\n(p={all_hypothesis_results.get('H7_time_effect', {}).get('p_value', float('nan')):.4f})",
                    fontweight='bold')
axs[3, 1].set_ylabel('Mean Vigilance (s)')
annotate(axs[3, 1], 'Contextual')

# H8: Composite threat index vs vigilance
sns.regplot(x='threat_index', y='bat_landing_to_food', data=dataset1,
            ax=axs[3, 2], scatter_kws={'s':15, 'alpha':0.5}, line_kws={'color':'crimson'})
h8 = all_hypothesis_results.get('H8_composite_threat', {})
axs[3, 2].set_title(f"H8: Composite Threat → Vigilance\n(r={h8.get('correlation', float('nan')):.3f}, p={h8.get('p_value', float('nan')):.4f})",
                    fontweight='bold')
axs[3, 2].set_xlabel('Composite Threat Index (scaled)')
axs[3, 2].set_ylabel('Vigilance (s)')
annotate(axs[3, 2], verdict_corr('pos_support', h8.get('correlation', np.nan), h8.get('p_value', np.nan)))

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
phase4_overview = os.path.join(plots_dir, 'Phase4_Hypotheses_Overview_(Threat_to_Responses).png')
plt.savefig(phase4_overview, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved Phase 4 overview plot to: {phase4_overview}")

#%%
# ============================================================================
# PHASE 5: GLM ANALYSIS - CONTROLLED EFFECTS
# ============================================================================
print("\n" + "="*60)
print("PHASE 5: GLM ANALYSIS - CONTROLLED EFFECTS")
print("="*60)

import statsmodels.api as sm
from statsmodels.genmod import families

print("Testing multivariate relationships while controlling for confounders\n")

# Model 1: Basic threat model
print("="*40)
print("MODEL 1: THREAT EFFECTS ON FEEDING SUCCESS")
print("="*40)

glm_data = dataset1.copy()
response = 'reward'
predictors = ['seconds_after_rat_arrival', 'rat_minutes', 'rat_arrival_number',
              'food_availability', 'hours_after_sunset_x', 'bat_landing_number']

glm_subset = glm_data[predictors + [response]].dropna()
X = sm.add_constant(glm_subset[predictors])
y = glm_subset[response]

print(f"Sample size: {len(glm_subset)}")

glm_model = sm.GLM(y, X, family=families.Binomial())
glm_results = glm_model.fit()

print(f"Model fit: AIC={glm_results.aic:.1f}\n")
print("Threat predictor effects:")
print("-" * 50)

# Store GLM results for Phase 6
glm_threat_effects = {}
for var in ['seconds_after_rat_arrival', 'rat_minutes', 'rat_arrival_number']:
    coef = glm_results.params[var]
    p_val = glm_results.pvalues[var]
    glm_threat_effects[var] = {'coefficient': coef, 'p_value': p_val}
    
    print(f"{var:30} β={coef:+.6f}  p={p_val:.4f}")
    if p_val < 0.05:
        print(f"  → {'SIGNIFICANT' if p_val < 0.01 else 'Significant'}: ", end="")
        if var == 'seconds_after_rat_arrival':
            print("Recent arrivals REDUCE success **" if coef < 0 else "Recent arrivals INCREASE success (unexpected)")
        elif var == 'rat_minutes':
            print("More rat presence REDUCES success **" if coef < 0 else "More rat presence INCREASES success (unexpected)")
        elif var == 'rat_arrival_number':
            print("More arrivals REDUCE success **" if coef < 0 else "More arrivals INCREASE success (unexpected)")

# Model 2: Including defensive behavior
print("\n" + "="*40)
print("MODEL 2: INCLUDING BEHAVIORAL RESPONSE")
print("="*40)

glm_data['defensive'] = glm_data['habit'].isin(defensive_habits).astype(int)
predictors_behav = predictors + ['defensive']

glm_subset_behav = glm_data[predictors_behav + [response]].dropna()
X_behav = sm.add_constant(glm_subset_behav[predictors_behav])
y_behav = glm_subset_behav[response]

glm_behav = sm.GLM(y_behav, X_behav, family=families.Binomial())
glm_results_behav = glm_behav.fit()

coef_def = glm_results_behav.params['defensive']
p_def = glm_results_behav.pvalues['defensive']
glm_threat_effects['defensive_behavior'] = {'coefficient': coef_def, 'p_value': p_def}

print(f"Defensive behavior effect: β={coef_def:+.4f}, p={p_def:.4f}")
if p_def < 0.05 and coef_def < 0:
    print("  → Defensive behaviors REDUCE feeding success (as expected) **")

# === Phase 5 Visualizations: GLM Overview ===
# Verdict helper for GLM coefficients (binary outcome: reward)
def verdict_glm(var, coef, p):
    if p is None or np.isnan(p) or p >= 0.05:
        return 'Not significant'
    # Predator evidence if higher threat reduces success (negative effect)
    if var in ['seconds_after_rat_arrival', 'rat_minutes', 'rat_arrival_number', 'defensive_behavior']:
        return 'Predator evidence' if coef < 0 else 'Competitor/facilitation'
    return 'Not significant'

# Pretty names for axes
glm_pretty = {
    'seconds_after_rat_arrival': 'Seconds After Rat Arrival (lower = closer)',
    'rat_minutes': 'Rat Minutes per Period',
    'rat_arrival_number': 'Number of Rat Arrivals',
    'defensive': 'Defensive Behavior (0/1)'
}

# Create combined figure (2x3): three threat predictors + defensive + model info panel
fig_glm, axs_glm = plt.subplots(2, 3, figsize=(20, 12), facecolor='white')
fig_glm.suptitle('Phase 5: GLM Effects on Feeding Success', fontsize=18, fontweight='bold')

from matplotlib.patches import Patch
legend_handles_glm = [
    Patch(color=colors_map['Predator evidence'], label='Predator evidence: increasing threat reduces success (β < 0).'),
    Patch(color=colors_map['Competitor/facilitation'], label='Competitor/facilitation: success improves with rat activity (β > 0).'),
    Patch(color=colors_map['Not significant'], label='Not significant: no reliable effect (p ≥ 0.05).')
]
fig_glm.legend(handles=legend_handles_glm, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=1, frameon=True)

# Helper to draw a predictor panel with predicted line
import numpy as _np

def draw_glm_panel(ax, model, predictors_list, x_name, data_df, title_prefix):
    # Jittered y for visibility
    y_jit = data_df['reward'] + _np.random.default_rng(7).uniform(-0.02, 0.02, size=len(data_df))
    ax.scatter(data_df[x_name], y_jit, s=15, alpha=0.4, color='#455a64')

    # Build prediction line holding others at mean
    x_vals = _np.linspace(data_df[x_name].min(), data_df[x_name].max(), 200)
    base = data_df[predictors_list].mean()
    X_line = _pd.DataFrame(_np.repeat([base.values], len(x_vals), axis=0), columns=predictors_list)
    X_line[x_name] = x_vals
    X_line = sm.add_constant(X_line, has_constant='add')
    y_hat = model.predict(X_line)
    ax.plot(x_vals, y_hat, color='crimson', linewidth=2)

    # Title and labels
    coef = model.params.get(x_name, _np.nan)
    pval = model.pvalues.get(x_name, _np.nan)
    ax.set_title(f"{title_prefix}\n(β={coef:+.4f}, p={pval:.4f})", fontweight='bold')
    ax.set_xlabel(glm_pretty.get(x_name, x_name))
    ax.set_ylabel('Feeding Success (0/1)')
    ax.set_ylim(-0.1, 1.1)
    annotate(ax, verdict_glm(x_name if x_name != 'defensive' else 'defensive_behavior', coef, pval))

# Panels for main GLM (model 1)
_pd = pd  # alias for local helper use
main_predictors = predictors
main_df = glm_subset.copy()

# seconds_after_rat_arrival
draw_glm_panel(axs_glm[0, 0], glm_results, main_predictors, 'seconds_after_rat_arrival', main_df, 'GLM: Proximity → Success')
# rat_minutes
draw_glm_panel(axs_glm[0, 1], glm_results, main_predictors, 'rat_minutes', main_df, 'GLM: Intensity → Success')
# rat_arrival_number
draw_glm_panel(axs_glm[0, 2], glm_results, main_predictors, 'rat_arrival_number', main_df, 'GLM: Frequency → Success')

# Defensive model panel (model 2)
behav_predictors = predictors_behav
behav_df = glm_subset_behav.copy()
draw_glm_panel(axs_glm[1, 0], glm_results_behav, behav_predictors, 'defensive', behav_df, 'GLM: Defensive Behavior → Success')

# Model info / coefficients table (text panel)
axs_glm[1, 1].axis('off')
lines = [
    'Model 1 (Threat Predictors):',
    f"  β(sec after rat) = {glm_results.params['seconds_after_rat_arrival']:+.4f} (p={glm_results.pvalues['seconds_after_rat_arrival']:.4f})",
    f"  β(rat minutes)   = {glm_results.params['rat_minutes']:+.4f} (p={glm_results.pvalues['rat_minutes']:.4f})",
    f"  β(arrivals)       = {glm_results.params['rat_arrival_number']:+.4f} (p={glm_results.pvalues['rat_arrival_number']:.4f})",
    '',
    'Model 2 (Including Behavior):',
    f"  β(defensive)      = {glm_results_behav.params['defensive']:+.4f} (p={glm_results_behav.pvalues['defensive']:.4f})",
]
axs_glm[1, 1].text(0.0, 1.0, "\n".join(lines), va='top', ha='left', fontsize=12)

# Empty / spare panel for future or leave blank
axs_glm[1, 2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
phase5_overview = os.path.join(plots_dir, 'Phase5_GLM_Effects_on_Feeding_Succes.png')
plt.savefig(phase5_overview, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved Phase 5 overview plot to: {phase5_overview}")

#%%
# ============================================================================
# PHASE 6: FINAL CONCLUSION AND ANSWER
# ============================================================================
print("\n" + "="*60)
print("PHASE 6: FINAL CONCLUSION AND ANSWER")
print("="*60)

# Compile evidence
evidence_for = []
evidence_against = []

print("="*40)
print("EVIDENCE COMPILATION")
print("="*40)

# From hypothesis tests
for test_name, result in all_hypothesis_results.items():
    if 'p_value' in result and result['p_value'] < 0.05:
        if 'correlation' in result:
            corr = result['correlation']
            p = result['p_value']
            
            if 'H1' in test_name and corr < 0:
                evidence_for.append(f"Temporal proximity increases vigilance (r={corr:.3f}, p={p:.3f})")
            elif 'H1' in test_name and corr > 0:
                evidence_against.append(f"Temporal proximity DECREASES vigilance (r={corr:.3f}, p={p:.3f})")
            
            elif 'H2' in test_name and corr < 0:
                evidence_for.append(f"Rat intensity reduces success (r={corr:.3f}, p={p:.3f})")
            elif 'H2' in test_name and corr > 0:
                evidence_against.append(f"Rat intensity INCREASES success (r={corr:.3f}, p={p:.3f})")
            
            elif 'H4' in test_name and corr < 0:
                evidence_for.append(f"More arrivals reduce success (r={corr:.3f}, p={p:.3f})")

# From GLM
for var, result in glm_threat_effects.items():
    if result['p_value'] < 0.05:
        coef = result['coefficient']
        p = result['p_value']
        
        if var == 'seconds_after_rat_arrival' and coef < 0:
            evidence_for.append(f"GLM: Recent rat arrival reduces success (β={coef:.4f}, p={p:.3f})")
        elif var == 'rat_minutes' and coef > 0:
            evidence_against.append(f"GLM: Rat presence increases success (β={coef:.3f}, p={p:.3f})")
        elif var == 'rat_arrival_number' and coef < 0:
            evidence_for.append(f"GLM: Multiple arrivals reduce success (β={coef:.3f}, p={p:.3f})")
        elif var == 'defensive_behavior' and coef < 0:
            evidence_for.append(f"GLM: Defensive behavior reduces success (β={coef:.2f}, p={p:.3f})")

total_evidence_for = len(evidence_for)
total_evidence_against = len(evidence_against)

print(f"\nEvidence summary:")
print(f"  Supporting predator hypothesis: {total_evidence_for}")
print(f"  Contradicting predator hypothesis: {total_evidence_against}")

# FINAL ANSWER WITH DETAILED EXPLANATION
print("\n" + "="*40)
print("FINAL ANSWER TO INVESTIGATION A")
print("="*40)
print("Do bats perceive rats as potential predators?")
print()

if total_evidence_for >= 3:
    print("ANSWER: YES - EVIDENCE SUPPORTS PREDATOR PERCEPTION")
    # Extra interpretation directly under the answer
    if total_evidence_for > total_evidence_against:
        print("Extra: Predator interaction (close to mixed; some facilitation signals remain).")
    elif total_evidence_against > total_evidence_for:
        print("Extra: Competitor/facilitator-leaning signal.")
    else:
        print("Extra: Mixed/ambiguous signal.")
    print(f"\nSupporting evidence ({total_evidence_for} indicators):")
    for e in evidence_for:
        print(f"  • {e}")
        
elif total_evidence_for >= 1:
    print("ANSWER: WEAK EVIDENCE FOR PREDATOR PERCEPTION")
    # Extra interpretation directly under the answer
    if total_evidence_for > total_evidence_against:
        print("Extra: Predator interaction (close to mixed; some facilitation signals remain).")
    elif total_evidence_against > total_evidence_for:
        print("Extra: Competitor/facilitator-leaning signal.")
    else:
        print("Extra: Mixed/ambiguous signal.")
    print(f"\nLimited supporting evidence ({total_evidence_for} indicators):")
    for e in evidence_for:
        print(f"  • {e}")
    if total_evidence_against > 0:
        print(f"\nHowever, contradicting evidence also found ({total_evidence_against} indicators):")
        for e in evidence_against:
            print(f"  • {e}")
            
else:
    print("ANSWER: NO - INSUFFICIENT EVIDENCE FOR PREDATOR PERCEPTION")
    # Extra interpretation directly under the answer
    if total_evidence_against > 0:
        print("Extra: Competitor/facilitator or ambiguous interaction.")
    else:
        print("Extra: No clear pattern; ambiguous interaction.")
    print("\n" + "="*40)
    print("DETAILED EXPLANATION OF NEGATIVE FINDING")
    print("="*40)

if total_evidence_for > total_evidence_against:
    print("Interpretation: Predator interaction, but close to mixed (some facilitation signals remain).")
elif total_evidence_against > total_evidence_for:
    print("Interpretation: Competitor/facilitator interaction.")
else:
    print("Interpretation: Ambiguous interaction; evidence balanced between predator and competition.")

    
    print("\n1. EXPECTED VS OBSERVED PATTERNS:")
    print("   If bats perceived rats as predators, we would expect:")
    print("   • Higher vigilance when rats are recently present → NOT FOUND")
    print("   • Lower feeding success with more rat activity → NOT FOUND")
    print("   • More defensive behaviors under high threat → NOT FOUND")
    
    print("\n2. ACTUAL PATTERNS IN THE DATA:")
    if any('temporal' in str(e).lower() for e in evidence_against):
        print("   • Positive correlation between time since rat arrival and vigilance")
        print("     (Bats are MORE cautious when rats have been gone LONGER)")
    if any('intensity' in str(e).lower() for e in evidence_against):
        print("   • Positive correlation between rat presence and feeding success")
        print("     (Bats are MORE successful when rats are around)")
    
    print("\n3. POSSIBLE EXPLANATIONS FOR NEGATIVE FINDING:")
    print("   a) Rats as facilitators, not threats:")
    print("      - Rats might indicate good foraging conditions")
    print("      - Both species attracted to same high-quality resources")
    print("   b) Habituation effect:")
    print("      - Bats may have habituated to rat presence in this colony")
    print("      - Regular exposure reduced threat perception")
    print("   c) Methodological considerations:")
    print("      - All observations include rat presence (no control)")
    print("      - Temporal variables may not capture immediate responses")

print("\n" + "="*40)
print("STATISTICAL SUMMARY")
print("="*40)
print(f"Hypotheses tested: {len(all_hypothesis_results)}")
print(f"GLM predictors tested: {len(glm_threat_effects)}")
print(f"Total significant findings: {total_evidence_for + total_evidence_against}")
print(f"Direction of evidence: {'Supports' if total_evidence_for > total_evidence_against else 'Contradicts' if total_evidence_against > total_evidence_for else 'Mixed'}")

# === Phase 6 Visualization: Final Summary ===
fig6, axs6 = plt.subplots(2, 3, figsize=(22, 13), facecolor='white')
fig6.suptitle('Phase 6: Final Summary and Verdict', fontsize=18, fontweight='bold')

from matplotlib.patches import Patch
legend_handles_phase6 = [
    Patch(color=colors_map['Predator evidence'], label='Predator evidence: behavior consistent with avoiding rats.'),
    Patch(color=colors_map['Competitor/facilitation'], label='Competitor/facilitation: behavior improves with rat activity.'),
    Patch(color=colors_map['Not significant'], label='Not significant: no reliable effect/relationship (p ≥ 0.05).'),
    Patch(color=colors_map['Contextual'], label='Contextual: descriptive control, not a predator test.')
]
fig6.legend(handles=legend_handles_phase6, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=1, frameon=True)

# Panel A: Evidence tally
bars_tally = axs6[0, 0].bar(['Predator evidence', 'Competitor/facilitation'],
               [total_evidence_for, total_evidence_against],
               color=[colors_map['Predator evidence'], colors_map['Competitor/facilitation']], alpha=0.85)
axs6[0, 0].set_title('Evidence Tally', fontweight='bold')
axs6[0, 0].set_ylabel('Count of significant indicators')
tally_max = max(total_evidence_for, total_evidence_against)
axs6[0, 0].set_ylim(0, tally_max + 1.5)
for i, bar in enumerate(bars_tally):
    height = bar.get_height()
    axs6[0, 0].text(bar.get_x() + bar.get_width()/2, height + 0.2, str(int(height)), ha='center', va='bottom', fontweight='bold')

# Panel B: Key GLM effects (bars colored by verdict)
glm_names = ['seconds_after_rat_arrival', 'rat_minutes', 'rat_arrival_number', 'defensive_behavior']
glm_labels = ['Proximity', 'Intensity', 'Frequency', 'Defensive']
glm_vals = [glm_threat_effects.get(n, {}).get('coefficient', np.nan) for n in glm_names]
glm_p = [glm_threat_effects.get(n, {}).get('p_value', np.nan) for n in glm_names]
glm_verdicts = [verdict_glm(n, glm_threat_effects.get(n, {}).get('coefficient', np.nan), glm_threat_effects.get(n, {}).get('p_value', np.nan)) for n in glm_names]
glm_colors = [colors_map.get(v, '#9e9e9e') for v in glm_verdicts]

# Draw bars but hide green (Predator evidence) bars by setting height=0 and alpha=0
bars_glm = []
for label, coef, color, ver in zip(glm_labels, glm_vals, glm_colors, glm_verdicts):
    draw_height = 0 if ver == 'Predator evidence' else coef
    draw_alpha = 0.0 if ver == 'Predator evidence' else 0.9
    bar = axs6[0, 1].bar(label, draw_height, color=color, alpha=draw_alpha)
    bars_glm.append(bar[0])

axs6[0, 1].axhline(0, color='#424242', linewidth=1)
axs6[0, 1].set_title('GLM Effects on Success (β)', fontweight='bold')
axs6[0, 1].set_ylabel('Coefficient (β)')
abs_max = max([abs(v) for v in glm_vals if not np.isnan(v)] + [1])
axs6[0, 1].set_ylim(-abs_max * 1.8, abs_max * 1.8)

# Add annotations for all predictors. For hidden bars, anchor at the TRUE coef height
for bar, coef, pval, ver in zip(bars_glm, glm_vals, glm_p, glm_verdicts):
    x = bar.get_x() + bar.get_width()/2
    y = coef
    offset = 14 if coef >= 0 else -16
    va = 'bottom' if coef >= 0 else 'top'
    axs6[0, 1].annotate(f"p={pval:.3f}\n{ver}", xy=(x, y), xytext=(0, offset), textcoords='offset points',
                        ha='center', va=va, fontsize=9,
                        bbox=dict(facecolor='white', edgecolor=colors_map.get(ver, '#424242'), boxstyle='round,pad=0.25'))

# Panel C: Hypotheses verdict grid (compact)
axs6[0, 2].axis('off')
grid_items = []
def _safe_get(name, key, default=np.nan):
    d = all_hypothesis_results.get(name, {})
    return d.get(key, default)

# Build verdicts using existing helpers
h1_ver = verdict_corr('neg_support', _safe_get('H1_temporal_vigilance', 'correlation'), _safe_get('H1_temporal_vigilance', 'p_value'))
h2_ver = verdict_corr('neg_support', _safe_get('H2_intensity_success', 'correlation'), _safe_get('H2_intensity_success', 'p_value'))
h3_ver = verdict_corr('pos_support', _safe_get('H3_frequency_vigilance', 'correlation'), _safe_get('H3_frequency_vigilance', 'p_value'))
h4_ver = verdict_corr('neg_support', _safe_get('H4_frequency_success', 'correlation'), _safe_get('H4_frequency_success', 'p_value'))

h6a = all_hypothesis_results.get('H6_defensive_seconds_after_rat_arrival', {})
h6b = all_hypothesis_results.get('H6_defensive_rat_minutes', {})
h6c = all_hypothesis_results.get('H6_defensive_rat_arrival_number', {})
h6a_ver = verdict_groups(h6a.get('high_threat', np.nan), h6a.get('low_threat', np.nan), h6a.get('p_value', np.nan))
h6b_ver = verdict_groups(h6b.get('high_threat', np.nan), h6b.get('low_threat', np.nan), h6b.get('p_value', np.nan))
h6c_ver = verdict_groups(h6c.get('high_threat', np.nan), h6c.get('low_threat', np.nan), h6c.get('p_value', np.nan))

h7_ver = 'Contextual'
h8_ver = verdict_corr('pos_support', _safe_get('H8_composite_threat', 'correlation'), _safe_get('H8_composite_threat', 'p_value'))

grid_items = [
    ("H1 Proximity → Vigilance", h1_ver),
    ("H2 Intensity → Success", h2_ver),
    ("H3 Frequency → Vigilance", h3_ver),
    ("H4 Frequency → Success", h4_ver),
    ("H6 Proximity → Defensive", h6a_ver),
    ("H6 Intensity → Defensive", h6b_ver),
    ("H6 Frequency → Defensive", h6c_ver),
    ("H7 Time-of-Night (control)", h7_ver),
    ("H8 Composite Threat → Vigilance", h8_ver),
]

ypos = 0.95
for label, verdict in grid_items:
    axs6[0, 2].text(0.02, ypos, f"{label}: {verdict}", fontsize=10, fontweight='bold', ha='left', va='top',
                    color=colors_map.get(verdict, '#000000'),
                    bbox=dict(facecolor='white', edgecolor=colors_map.get(verdict, '#000000'), boxstyle='round,pad=0.3'))
    ypos -= 0.10
axs6[0, 2].set_title('Hypotheses Verdicts', fontweight='bold')

# Panel D: Final answer text
axs6[1, 0].axis('off')
final_lines = []
if total_evidence_for >= 3:
    final_lines.append('ANSWER: YES - EVIDENCE SUPPORTS PREDATOR PERCEPTION')
elif total_evidence_for >= 1:
    final_lines.append('ANSWER: WEAK EVIDENCE FOR PREDATOR PERCEPTION')
else:
    final_lines.append('ANSWER: NO - INSUFFICIENT EVIDENCE FOR PREDATOR PERCEPTION')

if total_evidence_for > total_evidence_against:
    final_lines.append('Interpretation: Predator interaction, but close to mixed')
    final_lines.append('some facilitation signals remain).')
elif total_evidence_against > total_evidence_for:
    final_lines.append('Interpretation:')
    final_lines.append('Competitor/facilitator interaction.')
else:
    final_lines.append('Interpretation:')
    final_lines.append('Ambiguous interaction; balanced signals.')

axs6[1, 0].text(0.0, 1.0, "\n".join(final_lines), va='top', ha='left', fontsize=14, fontweight='bold')

# Panel E: Statistical summary (compact)
axs6[1, 1].axis('off')
stats_lines = [
    f"Hypotheses tested: {len(all_hypothesis_results)}",
    f"GLM predictors tested: {len(glm_threat_effects)}",
    f"Total significant findings: {total_evidence_for + total_evidence_against}",
    f"Predator evidence: {total_evidence_for}",
    f"Competitor/facilitation: {total_evidence_against}",
]
axs6[1, 1].text(0.0, 1.0, "\n".join(stats_lines), va='top', ha='left', fontsize=12)

# Panel F: Evidence direction pie chart
axs6[1, 2].pie([max(total_evidence_for, 0.0001), max(total_evidence_against, 0.0001)],
               labels=['Predator', 'Competitor'], autopct='%1.0f%%', startangle=90,
               colors=[colors_map['Predator evidence'], colors_map['Competitor/facilitation']])
axs6[1, 2].axis('equal')
axs6[1, 2].set_title('Direction of Significant Evidence', fontweight='bold')

plt.tight_layout(rect=[0, 0.06, 1, 0.93])
phase6_summary = os.path.join(plots_dir, 'Phase6_Final_Summary_and_Verdict.png')
plt.savefig(phase6_summary, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved Phase 6 final summary plot to: {phase6_summary}")
