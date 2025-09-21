# -*- coding: utf-8 -*-
#%%
# Investigation B: Do behaviours change with seasonal conditions?

"""
Investigation B: Do the behaviours change across seasons?

Focus: Compare bat foraging/anti-predator behaviours between Winter (food scarce, fewer rat encounters)
and Spring (food abundant, more rat encounters).

Data sources:
- Dataset1: Individual bat landing events with detailed behavioural measures
- Dataset2: 30-minute environmental periods with rat activity context

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
fig.suptitle('Phase 1: Data Overview (Investigation B)', fontsize=16, fontweight='bold')
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
print("Context: Standardizing behaviours to enable Winter vs Spring comparisons.")

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
print("STEP 1: HYPOTHESIS FORMULATION (Investigation B)")
print("=" * 40)
print("Question: Do behaviours differ between seasons (Winter vs Spring)?")
print("")
print("Context: Winter = scarce food, fewer rat encounters; Spring = abundant food, more encounters.")
print("")
print("H0 (Null): Behavioural measures do not differ by season.")
print("H1 (Alternative): Behavioural measures differ by season.")
print("")
print("Key comparisons to test:")
print("  • Rat encounter frequency (rat_arrival_number)")
print("  • Rat presence intensity (rat_minutes)")
print("  • Vigilance (bat_landing_to_food)")
print("  • Foraging success (reward)")
print("")
print("Significance level: α = 0.05")
print("Approach: Seasonal EDA + Mann–Whitney tests; GLMs with season and interactions in Phase 5")

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

# === Season derivation for Investigation B (Winter vs Spring) ===
def _season_label_from_month(dt):
    try:
        m = int(pd.to_datetime(dt).month)
    except Exception:
        return 'Other'
    # Northern Hemisphere season mapping
    if m in [12, 1, 2]:
        return 'Winter'
    if m in [3, 4, 5]:
        return 'Spring'
    if m in [6, 7, 8]:
        return 'Summer'
    if m in [9, 10, 11]:
        return 'Autumn'
    return 'Other'

dataset1['start_time'] = pd.to_datetime(dataset1['start_time'])
dataset1['season_label'] = dataset1['start_time'].apply(_season_label_from_month)
season_counts = dataset1['season_label'].value_counts().to_dict()
print(f"Seasons present (Phase 3): {season_counts}")

# Overwrite the original merged file to include season_label (no new file)
merged_filename = os.path.join(datasets_dir, 'dataset1_merged_with_dataset2.csv')
dataset1.to_csv(merged_filename, index=False)
print(f"Updated merged dataset (overwritten) with season: {merged_filename}")

# Choose analysis pair based on availability: prefer Winter vs Spring, else Summer vs Autumn, else top 2
available = set(dataset1['season_label'].dropna().unique())
analysis_pair = None
for pair in [('Winter', 'Spring'), ('Summer', 'Autumn')]:
    if set(pair).issubset(available):
        analysis_pair = pair
        break
if analysis_pair is None:
    top_two = list(dataset1['season_label'].value_counts().head(2).index)
    analysis_pair = (top_two[0], top_two[1]) if len(top_two) == 2 else (top_two[0], top_two[0])

# Build EDA frame and safe hue column
_eda_df = dataset1[dataset1['season_label'].isin(analysis_pair)].copy()
if len(_eda_df) == 0:
    _eda_df = dataset1.copy()
_eda_df['season_label_plot'] = _eda_df['season_label']
if _eda_df['season_label_plot'].nunique() == 0:
    _eda_df['season_label_plot'] = 'All'
elif _eda_df['season_label_plot'].nunique() == 1:
    # keep single label but still usable as hue
    pass

# === Phase 3 Visualizations ===
viz_dir = os.path.join(plots_dir)
os.makedirs(viz_dir, exist_ok=True)

# One combined figure (2 rows x 3 columns):
# Top row = distributions; Bottom row = outcomes
fig, axs = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')
fig.suptitle('Phase 3: Seasonal Distributions and Outcomes (Winter vs Spring)', fontsize=18, fontweight='bold')

# Distributions (top row)
sns.histplot(data=_eda_df, x='seconds_after_rat_arrival', bins=30, ax=axs[0, 0], hue='season_label_plot', multiple='stack')
axs[0, 0].set_title('Temporal Proximity to Rat Arrival', fontweight='bold')
axs[0, 0].set_xlabel('Seconds After Rat Arrival (lower = closer)')
axs[0, 0].set_ylabel('Count')

sns.histplot(data=_eda_df, x='rat_minutes', bins=30, ax=axs[0, 1], hue='season_label_plot', multiple='stack')
axs[0, 1].set_title('Rat Presence Intensity', fontweight='bold')
axs[0, 1].set_xlabel('Rat Minutes per Period')
axs[0, 1].set_ylabel('Count')

sns.histplot(data=_eda_df, x='rat_arrival_number', bins=30, ax=axs[0, 2], hue='season_label_plot', multiple='stack')
axs[0, 2].set_title('Rat Arrival Frequency', fontweight='bold')
axs[0, 2].set_xlabel('Number of Rat Arrivals')
axs[0, 2].set_ylabel('Count')

# Outcomes (bottom row)
sns.scatterplot(x='seconds_after_rat_arrival', y='bat_landing_to_food', data=_eda_df, ax=axs[1, 0], s=20, alpha=0.6, hue='season_label_plot')
axs[1, 0].set_title('Closer to Rat Arrival → Vigilance', fontweight='bold')
axs[1, 0].set_xlabel('Seconds After Rat Arrival (lower = closer)')
axs[1, 0].set_ylabel('Vigilance (bat_landing_to_food, s)')

sns.scatterplot(x='rat_minutes', y='reward', data=_eda_df, ax=axs[1, 1], s=20, alpha=0.6, hue='season_label_plot')
axs[1, 1].set_title('Rat Presence Intensity → Success', fontweight='bold')
axs[1, 1].set_xlabel('Rat Minutes per Period')
axs[1, 1].set_ylabel('Feeding Success (0/1)')

sns.scatterplot(x='rat_arrival_number', y='bat_landing_to_food', data=_eda_df, ax=axs[1, 2], s=20, alpha=0.6, hue='season_label_plot')
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
print("Seasonal analysis focus (Investigation B):")
print("1. Do distributions of threat indicators differ by season (Winter vs Spring)?")
print("2. Do outcomes (vigilance, success) differ by season?")
print("3. How do threat–outcome relationships vary across seasons?")
print("\nWe use plots and tests to compare Winter (scarce food, fewer rats) vs Spring (abundant food, more rats).")

#%%
# ============================================================================
# PHASE 3.1: IQR ANALYSIS AND CORRELATION MAPPING
# ============================================================================
print("\n" + "="*60)
print("PHASE 3.1: IQR ANALYSIS AND CORRELATION MAPPING")
print("="*60)

# STEP 1: IQR ANALYSIS FOR KEY VARIABLES
print("=" * 40)
print("STEP 1: IQR ANALYSIS FOR KEY VARIABLES")
print("=" * 40)

key_variables = {
    'bat_landing_to_food': 'Vigilance (seconds)',
    'seconds_after_rat_arrival': 'Temporal Proximity (seconds)',
    'rat_minutes': 'Threat Intensity (minutes)', 
    'rat_arrival_number': 'Threat Frequency (count)'
}

iqr_results = {}
print("IQR Analysis for key threat and response variables:\n")

for var, description in key_variables.items():
    if var in dataset1.columns:
        data = dataset1[var].dropna()
        
        # Calculate quartiles and IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        # Store results
        iqr_results[var] = {
            'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
            'lower_bound': lower_bound, 'upper_bound': upper_bound,
            'outliers': outliers, 'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data)) * 100
        }
        
        print(f"{description}:")
        print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"  Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  Outliers: {len(outliers)} ({(len(outliers) / len(data) * 100):.1f}%)")
        print(f"  Range: {data.min():.2f} to {data.max():.2f}")
        print()

# Seasonal summary (medians) for key variables
print("Seasonal medians for key variables (Winter vs Spring if available):")
if 'season_label' in dataset1.columns:
    for season in ['Winter', 'Spring']:
        if season in set(dataset1['season_label']):
            subset = dataset1[dataset1['season_label'] == season]
            vals = []
            for var, description in key_variables.items():
                if var in subset.columns and subset[var].notna().any():
                    vals.append(f"{description.split(' (')[0]}={subset[var].median():.2f}")
            if vals:
                print(f"  {season}: " + ", ".join(vals))
else:
    print("  season_label not found (run Phase 3 season derivation first)")

# STEP 2: CORRELATION MATRIX ANALYSIS
print("=" * 40)
print("STEP 2: CORRELATION MATRIX ANALYSIS")
print("=" * 40)

# Select numeric columns for correlation analysis
numeric_cols = dataset1.select_dtypes(include=[np.number]).columns
correlation_data = dataset1[numeric_cols].dropna()

# Calculate correlation matrix
correlation_matrix = correlation_data.corr()

print(f"Correlation matrix calculated for {len(numeric_cols)} numeric variables")
print(f"Sample size for correlation analysis: {len(correlation_data)}\n")

# Seasonal correlations for selected pairs
if 'season_label' in dataset1.columns:
    print("Seasonal correlations (selected pairs):")
    _pairs = [
        ('seconds_after_rat_arrival', 'bat_landing_to_food'),
        ('rat_minutes', 'reward'),
        ('rat_arrival_number', 'bat_landing_to_food'),
        ('rat_arrival_number', 'reward')
    ]
    for season in ['Winter', 'Spring']:
        subset = dataset1[dataset1['season_label'] == season]
        if len(subset) > 10:
            print(f"  {season}:")
            for a, b in _pairs:
                if a in subset.columns and b in subset.columns and subset[[a, b]].dropna().shape[0] > 5:
                    r = subset[[a, b]].dropna().corr().iloc[0, 1]
                    print(f"    {a} → {b}: r={r:.3f}")
        

# Identify strong correlations
print("Strong correlations identified:")
strong_correlations = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        var1 = correlation_matrix.columns[i]
        var2 = correlation_matrix.columns[j]
        corr_value = correlation_matrix.iloc[i, j]
        
        if abs(corr_value) >= 0.7:
            strength = "Very strong"
            strong_correlations.append((var1, var2, corr_value, strength))
        elif abs(corr_value) >= 0.5:
            strength = "Strong"
            strong_correlations.append((var1, var2, corr_value, strength))
        elif abs(corr_value) >= 0.3:
            strength = "Moderate"
            strong_correlations.append((var1, var2, corr_value, strength))

# Sort by absolute correlation strength
strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

if strong_correlations:
    for var1, var2, corr, strength in strong_correlations[:15]:  # Show top 15
        direction = "positive" if corr > 0 else "negative"
        print(f"  {var1} ↔ {var2}: r={corr:.3f} ({strength} {direction})")
else:
    print("  No correlations stronger than |r| = 0.3 found")

# Key correlations for hypothesis testing
print(f"\nKey correlations for threat hypothesis testing:")
threat_response_pairs = [
    ('seconds_after_rat_arrival', 'bat_landing_to_food'),
    ('rat_minutes', 'reward'),
    ('rat_arrival_number', 'bat_landing_to_food'),
    ('rat_arrival_number', 'reward')
]

for var1, var2 in threat_response_pairs:
    if var1 in correlation_matrix.columns and var2 in correlation_matrix.columns:
        corr_val = correlation_matrix.loc[var1, var2]
        print(f"  {var1} → {var2}: r={corr_val:.3f}")

# STEP 3: COMBINED VISUALIZATION
print(f"\n" + "=" * 40)
print("STEP 3: CREATING VISUALIZATIONS")
print("=" * 40)

# Create combined figure with IQR box plots and correlation heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor='white')
fig.suptitle('Phase 3.1: IQR Analysis and Correlation Mapping', fontsize=16, fontweight='bold')

# Left panel: Box plots for IQR analysis
box_data = []
box_labels = []
for var, description in key_variables.items():
    if var in dataset1.columns:
        box_data.append(dataset1[var].dropna())
        box_labels.append(description.replace(' (', '\n('))

bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True, notch=True)
ax1.set_title('IQR Distributions of Key Variables', fontweight='bold')
ax1.set_ylabel('Values')
ax1.tick_params(axis='x', rotation=45, labelsize=10)

# Color the box plots
colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add outlier count annotations
for i, (var, description) in enumerate(key_variables.items()):
    if var in iqr_results:
        outlier_count = iqr_results[var]['outlier_count']
        outlier_pct = iqr_results[var]['outlier_percentage']
        ax1.text(i+1, ax1.get_ylim()[1] * 0.95, f'{outlier_count} outliers\n({outlier_pct:.1f}%)', 
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Right panel: Correlation heatmap
# Focus on key variables for cleaner visualization
key_vars_for_heatmap = ['bat_landing_to_food', 'seconds_after_rat_arrival', 'rat_minutes', 
                       'rat_arrival_number', 'reward', 'risk', 'food_availability', 
                       'hours_after_sunset_x', 'bat_landing_number']
available_vars = [var for var in key_vars_for_heatmap if var in correlation_matrix.columns]
heatmap_corr = correlation_matrix.loc[available_vars, available_vars]

im = ax2.imshow(heatmap_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax2.set_title('Correlation Matrix (Key Variables)', fontweight='bold')

# Add correlation values as text
for i in range(len(available_vars)):
    for j in range(len(available_vars)):
        text = ax2.text(j, i, f'{heatmap_corr.iloc[i, j]:.2f}', 
                       ha='center', va='center', color='white' if abs(heatmap_corr.iloc[i, j]) > 0.5 else 'black',
                       fontsize=9, fontweight='bold')

ax2.set_xticks(range(len(available_vars)))
ax2.set_yticks(range(len(available_vars))) 
ax2.set_xticklabels([var.replace('_', '\n') for var in available_vars], rotation=45, ha='right')
ax2.set_yticklabels([var.replace('_', '\n') for var in available_vars])

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
phase31_path = os.path.join(plots_dir, 'Phase3.1_IQR_and_Correlation_Analysis.png')
plt.savefig(phase31_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print(f"Saved Phase 3.1 analysis plot to: {phase31_path}")

# STEP 4: KEY INSIGHTS SUMMARY
print(f"\n" + "=" * 40)
print("STEP 4: KEY STATISTICAL INSIGHTS")
print("=" * 40)

print("IQR Analysis Summary:")
total_outliers = sum(result['outlier_count'] for result in iqr_results.values())
print(f"  Total outliers identified: {total_outliers}")
print(f"  Variables with highest outlier rates:")
sorted_by_outliers = sorted(iqr_results.items(), key=lambda x: x[1]['outlier_percentage'], reverse=True)
for var, result in sorted_by_outliers[:3]:
    print(f"    {key_variables[var]}: {result['outlier_percentage']:.1f}% outliers")

print(f"\nCorrelation Analysis Summary:")
print(f"  {len(strong_correlations)} correlations found (|r| ≥ 0.3)")
if strong_correlations:
    print(f"  Strongest correlation: {strong_correlations[0][0]} ↔ {strong_correlations[0][1]} (r={strong_correlations[0][2]:.3f})")

print(f"\nImplications for Hypothesis Testing:")
print("  • IQR analysis will guide outlier treatment in statistical tests")
print("  • Strong correlations identified may require multicollinearity checks")
print("  • Distribution patterns inform choice of parametric vs non-parametric tests")
print("  • Outlier patterns may reveal important behavioral extremes")

#%%
# ============================================================================
# PHASE 4: HYPOTHESIS TESTING - SEASONAL COMPARISON (WINTER vs SPRING)
# ============================================================================
print("\n" + "="*60)
print("PHASE 4: HYPOTHESIS TESTING - SEASONAL COMPARISON")
print("="*60)

from scipy.stats import mannwhitneyu, fisher_exact
import numpy as np
from statsmodels.stats.multitest import multipletests
try:
    from statsmodels.stats.contingency_tables import Table2x2
except Exception:
    Table2x2 = None

# Use the merged dataset from Phase 3 and ensure season_label exists
dataset1 = merged_data.copy()
dataset1['start_time'] = pd.to_datetime(dataset1['start_time'])
try:
    dataset1['season_label']
except KeyError:
    # derive using the same function
    dataset1['season_label'] = dataset1['start_time'].apply(_season_label_from_month)

# Pick seasons to compare (prefer Winter vs Spring)
available = set(dataset1['season_label'].dropna().unique())
season_pair = None
for pair in [('Winter', 'Spring'), ('Summer', 'Autumn')]:
    if set(pair).issubset(available):
        season_pair = pair
        break
if season_pair is None and len(available) >= 2:
    top_two = list(dataset1['season_label'].value_counts().head(2).index)
    season_pair = (top_two[0], top_two[1])
elif season_pair is None:
    season_pair = (list(available)[0], list(available)[0])

print(f"Comparing seasons: {season_pair[0]} vs {season_pair[1]}")
w_df = dataset1[dataset1['season_label'] == season_pair[0]].copy()
s_df = dataset1[dataset1['season_label'] == season_pair[1]].copy()

def mw_test(x_w, x_s):
    x_w = pd.Series(x_w).dropna()
    x_s = pd.Series(x_s).dropna()
    if len(x_w) == 0 or len(x_s) == 0:
        return np.nan, np.nan, np.nan
    u, p = mannwhitneyu(x_w, x_s, alternative='two-sided')
    # rank-biserial effect size from U
    n1, n2 = len(x_w), len(x_s)
    rbes = 1 - (2*u)/(n1*n2)
    return p, rbes, u

results = {}

# H1 rat_arrival_number
if 'rat_arrival_number' in dataset1.columns:
    p, eff, _ = mw_test(w_df['rat_arrival_number'], s_df['rat_arrival_number'])
    results['H1_arrivals'] = {'p': p, 'effect': eff,
        'w_med': np.nanmedian(w_df['rat_arrival_number']), 's_med': np.nanmedian(s_df['rat_arrival_number'])}

# H2 rat_minutes
if 'rat_minutes' in dataset1.columns:
    p, eff, _ = mw_test(w_df['rat_minutes'], s_df['rat_minutes'])
    results['H2_minutes'] = {'p': p, 'effect': eff,
        'w_med': np.nanmedian(w_df['rat_minutes']), 's_med': np.nanmedian(s_df['rat_minutes'])}

# H3 vigilance
if 'bat_landing_to_food' in dataset1.columns:
    p, eff, _ = mw_test(w_df['bat_landing_to_food'], s_df['bat_landing_to_food'])
    results['H3_vigilance'] = {'p': p, 'effect': eff,
        'w_med': np.nanmedian(w_df['bat_landing_to_food']), 's_med': np.nanmedian(s_df['bat_landing_to_food'])}

# H4 success (reward) — Fisher exact
if 'reward' in dataset1.columns:
    w_succ = int(w_df['reward'].dropna().sum()); w_fail = int((w_df['reward']==0).sum())
    s_succ = int(s_df['reward'].dropna().sum()); s_fail = int((s_df['reward']==0).sum())
    table = np.array([[w_succ, w_fail],[s_succ, s_fail]])
    try:
        _, p = fisher_exact(table)
    except Exception:
        p = np.nan
    or_val, ci_low, ci_high = (np.nan, np.nan, np.nan)
    if Table2x2 is not None and table.min() > 0:
        t2 = Table2x2(table)
        or_val = t2.oddsratio
        ci_low, ci_high = t2.oddsratio_confint()
    results['H4_success'] = {'p': p, 'effect': or_val,
        'w_prop': w_succ/max(1, (w_succ+w_fail)), 's_prop': s_succ/max(1, (s_succ+s_fail)),
        'ci': (ci_low, ci_high)}

# H5 risk
if 'risk' in dataset1.columns:
    w_pos = int(w_df['risk'].dropna().sum()); w_neg = int((w_df['risk']==0).sum())
    s_pos = int(s_df['risk'].dropna().sum()); s_neg = int((s_df['risk']==0).sum())
    table = np.array([[w_pos, w_neg],[s_pos, s_neg]])
    try:
        _, p = fisher_exact(table)
    except Exception:
        p = np.nan
    results['H5_risk'] = {'p': p, 'w_prop': w_pos/max(1,(w_pos+w_neg)), 's_prop': s_pos/max(1,(s_pos+s_neg))}

# H6 defensive
defensive_habits = ['cautious', 'slow_approach', 'fight']
dataset1['defensive'] = dataset1['habit'].isin(defensive_habits).astype(int)
w_def = int(w_df['habit'].isin(defensive_habits).sum()); w_nodef = len(w_df) - w_def
s_def = int(s_df['habit'].isin(defensive_habits).sum()); s_nodef = len(s_df) - s_def
table = np.array([[w_def, w_nodef],[s_def, s_nodef]])
try:
    _, p = fisher_exact(table)
except Exception:
    p = np.nan
results['H6_defensive'] = {'p': p, 'w_prop': w_def/max(1,len(w_df)), 's_prop': s_def/max(1,len(s_df))}

# H7 time-of-night within season (vigilance)
if 'hours_after_sunset_x' in dataset1.columns and 'bat_landing_to_food' in dataset1.columns:
    def early_late_p(df):
        if len(df)==0:
            return np.nan
        m = df['hours_after_sunset_x'].median()
        e = df[df['hours_after_sunset_x']<=m]['bat_landing_to_food']
        l = df[df['hours_after_sunset_x']>m]['bat_landing_to_food']
        if e.notna().sum()>0 and l.notna().sum()>0:
            return mannwhitneyu(e.dropna(), l.dropna()).pvalue
        return np.nan
    results['H7_time_within_winter'] = {'p': early_late_p(w_df)}
    results['H7_time_within_spring'] = {'p': early_late_p(s_df)}

# FDR across primary four tests (H1–H4)
primary_keys = [k for k in ['H1_arrivals','H2_minutes','H3_vigilance','H4_success'] if k in results]
primary_p = [results[k]['p'] for k in primary_keys if not pd.isna(results[k]['p'])]
adj_map = {}
if len(primary_p):
    rej, p_adj, _, _ = multipletests(primary_p, method='fdr_bh')
    for k, pa, rj in zip(primary_keys, p_adj, rej):
        adj_map[k] = {'p_adj': pa, 'reject': bool(rj)}

print("\nSeasonal hypothesis results (Winter vs Spring):")
for k, v in results.items():
    line = f"  {k}: p={v.get('p', np.nan):.4f}"
    if k in adj_map:
        line += f", p_fdr={adj_map[k]['p_adj']:.4f}"
    if 'w_med' in v:
        line += f", Winter={v['w_med']:.2f}, Spring={v['s_med']:.2f}, effect={v.get('effect', np.nan):.3f}"
    if 'w_prop' in v:
        line += f", Winter={v['w_prop']:.2f}, Spring={v['s_prop']:.2f}"
    print(line)

# Removed seasonal 2x2 summary figure (kept classic overview only)

# Compute composite threat index for downstream A-style plots if needed
try:
    from sklearn.preprocessing import StandardScaler as _Phase4Scaler
    _threat_cols = ['seconds_after_rat_arrival', 'rat_minutes', 'rat_arrival_number']
    if all(col in dataset1.columns for col in _threat_cols):
        _valid_idx = dataset1[_threat_cols].dropna().index
        _tmp = dataset1.loc[_valid_idx, _threat_cols].copy()
        _tmp['threat_proximity'] = 1 / (1 + _tmp['seconds_after_rat_arrival']/60)
        _tmp['threat_intensity'] = _tmp['rat_minutes']
        _tmp['threat_frequency'] = _tmp['rat_arrival_number']
        _components = _tmp[['threat_proximity', 'threat_intensity', 'threat_frequency']]
        _scaled = _Phase4Scaler().fit_transform(_components)
        dataset1.loc[_valid_idx, 'threat_index'] = _scaled.mean(axis=1)
except Exception:
    pass

# Backward compatibility for downstream figures ported from Investigation A
# (Phase 5/6 reference all_hypothesis_results). We'll populate season-specific
# results separately; keep this defined to avoid NameErrors.
all_hypothesis_results = {}

# Summary table
print("\n" + "="*40)
print("HYPOTHESIS TESTING SUMMARY (Seasonal)")
print("="*40)
_total_tests = len(results)
_sig_count = 0
for k, v in results.items():
    if k in adj_map:
        if adj_map[k]['p_adj'] < 0.05:
            _sig_count += 1
    else:
        if v.get('p', 1.0) < 0.05:
            _sig_count += 1
print(f"Total hypotheses tested: {_total_tests}")
print(f"Significant results (FDR where applicable): {_sig_count}")

# === Phase 4 Visualizations: All Hypotheses in One Figure ===
# Helpers to compute verdict labels and annotate plots
colors_map = {
    # Season-aware palette and contextual colors
    'Higher in Spring': '#2e7d32',
    'Higher in Winter': '#c62828',
    'No seasonal difference': '#616161',
    'Contextual': '#1565c0',
    # Generic significance labels (used in GLM and some annotations)
    'Significant': '#2e7d32',
    'Not significant': '#616161',
    # Phase 6 categories
    'Predator evidence': '#2e7d32',
    'Competitor/facilitation': '#c62828'
}

def verdict_corr(rule, corr, p):
    if p is None or np.isnan(p) or p >= 0.05:
        return 'No seasonal difference'
    return 'Higher in Spring'  # Placeholder; not used for seasonal group diff in Phase 4

def verdict_groups(high, low, p):
    if p is None or np.isnan(p) or p >= 0.05:
        return 'No seasonal difference'
    return 'Higher in Spring' if (low is not None and high is not None and low < high) else 'Higher in Winter'

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
    Patch(color=colors_map['Higher in Spring'], label='Higher in Spring (p < 0.05)'),
    Patch(color=colors_map['Higher in Winter'], label='Higher in Winter (p < 0.05)'),
    Patch(color=colors_map['No seasonal difference'], label='No seasonal difference (p ≥ 0.05)'),
    Patch(color=colors_map['Contextual'], label='Contextual (descriptive/control panel)')
]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=1, frameon=True)

def _pv(k):
    if k in adj_map:
        return adj_map[k]['p_adj']
    return results.get(k, {}).get('p', np.nan)

def _vtext(k, w_val, s_val):
    p = _pv(k)
    if np.isnan(p):
        return 'No seasonal difference'
    if p >= 0.05:
        return 'No seasonal difference'
    return 'Higher in Spring' if s_val > w_val else 'Higher in Winter'

# H1: seconds_after_rat_arrival vs bat_landing_to_food (keep original style/colors)
sns.regplot(x='seconds_after_rat_arrival', y='bat_landing_to_food', data=dataset1,
            ax=axs[0, 0], scatter_kws={'s':15, 'alpha':0.5}, line_kws={'color':'crimson'})
axs[0, 0].set_title(f"H1: Proximity → Vigilance\n(seasonal p={_pv('H3_vigilance'):.4f})", fontweight='bold')
axs[0, 0].set_xlabel('Seconds After Rat Arrival (lower = closer)')
axs[0, 0].set_ylabel('Vigilance (bat_landing_to_food, s)')
annotate(
    axs[0, 0],
    _vtext('H3_vigilance', results.get('H3_vigilance', {}).get('w_med', np.nan), results.get('H3_vigilance', {}).get('s_med', np.nan))
)

# H2: rat_minutes vs reward (jitter) original style
_rng = np.random.default_rng(42)
y2 = dataset1['reward'] + _rng.uniform(-0.02, 0.02, size=len(dataset1))
axs[0, 1].scatter(dataset1['rat_minutes'], y2, s=15, alpha=0.5, color='darkorange')
axs[0, 1].set_title(f"H2: Intensity → Success\n(seasonal p={_pv('H4_success'):.4f})", fontweight='bold')
axs[0, 1].set_xlabel('Rat Minutes per Period')
axs[0, 1].set_ylabel('Feeding Success (0/1)')
axs[0, 1].set_ylim(-0.1, 1.1)
annotate(
    axs[0, 1],
    _vtext('H4_success', results.get('H4_success', {}).get('w_prop', np.nan), results.get('H4_success', {}).get('s_prop', np.nan))
)

# H3: rat_arrival_number vs bat_landing_to_food original style
sns.regplot(x='rat_arrival_number', y='bat_landing_to_food', data=dataset1,
            ax=axs[0, 2], scatter_kws={'s':15, 'alpha':0.5}, line_kws={'color':'crimson'})
axs[0, 2].set_title(f"H3: Frequency → Vigilance\n(seasonal p={_pv('H1_arrivals'):.4f})", fontweight='bold')
axs[0, 2].set_xlabel('Number of Rat Arrivals')
axs[0, 2].set_ylabel('Vigilance (s)')
annotate(
    axs[0, 2],
    _vtext('H1_arrivals', results.get('H1_arrivals', {}).get('w_med', np.nan), results.get('H1_arrivals', {}).get('s_med', np.nan))
)

# H4: rat_arrival_number vs reward (jitter) original style
y4 = dataset1['reward'] + _rng.uniform(-0.02, 0.02, size=len(dataset1))
axs[1, 0].scatter(dataset1['rat_arrival_number'], y4, s=15, alpha=0.5, color='seagreen')
axs[1, 0].set_title(f"H4: Frequency → Success\n(seasonal p={_pv('H4_success'):.4f})", fontweight='bold')
axs[1, 0].set_xlabel('Number of Rat Arrivals')
axs[1, 0].set_ylabel('Feeding Success (0/1)')
axs[1, 0].set_ylim(-0.1, 1.1)
annotate(
    axs[1, 0],
    _vtext('H4_success', results.get('H4_success', {}).get('w_prop', np.nan), results.get('H4_success', {}).get('s_prop', np.nan))
)

# H5a: seconds_after_rat_arrival vs risk (jitter) original style
y5a = dataset1['risk'] + _rng.uniform(-0.02, 0.02, size=len(dataset1))
axs[1, 1].scatter(dataset1['seconds_after_rat_arrival'], y5a, s=15, alpha=0.5, color='steelblue')
axs[1, 1].set_title(f"H5: Proximity → Risk\n(seasonal p={_pv('H5_risk'):.4f})", fontweight='bold')
axs[1, 1].set_xlabel('Seconds After Rat Arrival (lower = closer)')
axs[1, 1].set_ylabel('Risk (0/1)')
axs[1, 1].set_ylim(-0.1, 1.1)
annotate(
    axs[1, 1],
    _vtext('H5_risk', results.get('H5_risk', {}).get('w_prop', np.nan), results.get('H5_risk', {}).get('s_prop', np.nan))
)

# H5b: rat_minutes vs risk (jitter)
y5b = dataset1['risk'] + _rng.uniform(-0.02, 0.02, size=len(dataset1))
axs[1, 2].scatter(dataset1['rat_minutes'], y5b, s=15, alpha=0.5, color='darkorange')
axs[1, 2].set_title(f"H5: Intensity → Risk\n(seasonal p={_pv('H5_risk'):.4f})", fontweight='bold')
axs[1, 2].set_xlabel('Rat Minutes per Period')
axs[1, 2].set_ylabel('Risk (0/1)')
axs[1, 2].set_ylim(-0.1, 1.1)
annotate(
    axs[1, 2],
    _vtext('H5_risk', results.get('H5_risk', {}).get('w_prop', np.nan), results.get('H5_risk', {}).get('s_prop', np.nan))
)

# H5c: rat_arrival_number vs risk (jitter)
y5c = dataset1['risk'] + _rng.uniform(-0.02, 0.02, size=len(dataset1))
axs[2, 0].scatter(dataset1['rat_arrival_number'], y5c, s=15, alpha=0.5, color='seagreen')
axs[2, 0].set_title('H5: Frequency → Risk', fontweight='bold')
axs[2, 0].set_xlabel('Number of Rat Arrivals')
axs[2, 0].set_ylabel('Risk (0/1)')
axs[2, 0].set_ylim(-0.1, 1.1)
annotate(
    axs[2, 0],
    _vtext('H5_risk', results.get('H5_risk', {}).get('w_prop', np.nan), results.get('H5_risk', {}).get('s_prop', np.nan))
)

# H6a: Defensive proportion by proximity (keep original style)
valid = dataset1[['seconds_after_rat_arrival', 'defensive']].dropna()
median_val = valid['seconds_after_rat_arrival'].median()
high = valid[valid['seconds_after_rat_arrival'] <= median_val]['defensive']
low  = valid[valid['seconds_after_rat_arrival'] >  median_val]['defensive']
axs[2, 1].bar(['High Threat', 'Low Threat'], [high.mean(), low.mean()], color=['crimson','gray'])
axs[2, 1].set_ylim(0, 1)
axs[2, 1].set_title(f"H6: Defensive vs Proximity\n(p={_pv('H6_defensive'):.4f})", fontweight='bold')
axs[2, 1].set_ylabel('Defensive Proportion')
annotate(
    axs[2, 1],
    _vtext('H6_defensive', results.get('H6_defensive', {}).get('w_prop', np.nan), results.get('H6_defensive', {}).get('s_prop', np.nan))
)

# H6b: Defensive proportion by intensity (keep original style)
valid = dataset1[['rat_minutes', 'defensive']].dropna()
median_val = valid['rat_minutes'].median()
high = valid[valid['rat_minutes'] >  median_val]['defensive']
low  = valid[valid['rat_minutes'] <= median_val]['defensive']
axs[2, 2].bar(['High Threat', 'Low Threat'], [high.mean(), low.mean()], color=['darkorange','gray'])
axs[2, 2].set_ylim(0, 1)
axs[2, 2].set_title(f"H6: Defensive vs Intensity\n(p={_pv('H6_defensive'):.4f})", fontweight='bold')
axs[2, 2].set_ylabel('Defensive Proportion')
annotate(
    axs[2, 2],
    _vtext('H6_defensive', results.get('H6_defensive', {}).get('w_prop', np.nan), results.get('H6_defensive', {}).get('s_prop', np.nan))
)

# H6c: Defensive proportion by frequency (keep original style)
valid = dataset1[['rat_arrival_number', 'defensive']].dropna()
median_val = valid['rat_arrival_number'].median()
high = valid[valid['rat_arrival_number'] >  median_val]['defensive']
low  = valid[valid['rat_arrival_number'] <= median_val]['defensive']
axs[3, 0].bar(['High Threat', 'Low Threat'], [high.mean(), low.mean()], color=['seagreen','gray'])
axs[3, 0].set_ylim(0, 1)
axs[3, 0].set_title(f"H6: Defensive vs Frequency\n(p={_pv('H6_defensive'):.4f})", fontweight='bold')
axs[3, 0].set_ylabel('Defensive Proportion')
annotate(axs[3, 0], 'Significant' if _pv('H6_defensive') < 0.05 else 'Not significant')

# H7: Early vs Late night vigilance (bars)
axs[3, 1].bar(['Early Night', 'Late Night'], [
    dataset1[dataset1['hours_after_sunset_x'] <= dataset1['hours_after_sunset_x'].median()]['bat_landing_to_food'].mean(),
    dataset1[dataset1['hours_after_sunset_x'] >  dataset1['hours_after_sunset_x'].median()]['bat_landing_to_food'].mean()],
    color=['steelblue','gray'])
axs[3, 1].set_title('H7: Time-of-Night Effect (contextual)', fontweight='bold')
axs[3, 1].set_ylabel('Mean Vigilance (s)')
annotate(axs[3, 1], 'Contextual')

# H8: Composite threat index vs vigilance
sns.regplot(x='threat_index', y='bat_landing_to_food', data=dataset1,
            ax=axs[3, 2], scatter_kws={'s':15, 'alpha':0.5}, line_kws={'color':'crimson'})
axs[3, 2].set_title('H8: Composite Threat → Vigilance (contextual)', fontweight='bold')
axs[3, 2].set_xlabel('Composite Threat Index (scaled)')
axs[3, 2].set_ylabel('Vigilance (s)')
annotate(axs[3, 2], 'Not significant' if dataset1['threat_index'].isna().all() else 'Contextual')

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
print("PHASE 5: GLM ANALYSIS - SEASONAL EFFECTS (CONTROLLED)")
print("="*60)

import statsmodels.api as sm
from statsmodels.genmod import families
import statsmodels.formula.api as smf

print("Testing multivariate relationships with season and controls\n")

# Model 1: Basic threat model
print("="*40)
print("MODEL 1: FEEDING SUCCESS ~ THREATS + CONTROLS + SEASON")
print("="*40)

glm_data = dataset1.copy()
if 'season_label' not in glm_data.columns:
    glm_data['season_label'] = glm_data['start_time'].apply(_season_label_from_month)
glm_data['is_spring'] = (glm_data['season_label'] == 'Spring').astype(int)
glm_data['rat_minutes_is_spring'] = glm_data['rat_minutes'] * glm_data['is_spring']
glm_data['rat_arrival_number_is_spring'] = glm_data['rat_arrival_number'] * glm_data['is_spring']

response = 'reward'
predictors = ['seconds_after_rat_arrival', 'rat_minutes', 'rat_arrival_number',
              'food_availability', 'hours_after_sunset_x', 'bat_landing_number',
              'is_spring', 'rat_minutes_is_spring', 'rat_arrival_number_is_spring']

glm_subset = glm_data[predictors + [response]].dropna()
X = sm.add_constant(glm_subset[predictors])
y = glm_subset[response]

print(f"Sample size: {len(glm_subset)}")

glm_model = sm.GLM(y, X, family=families.Binomial())
glm_results = glm_model.fit()

print(f"Model fit: AIC={glm_results.aic:.1f}\n")
print("Key effects (seasonal controls):")
print("-" * 50)

# Store GLM results for Phase 6 (core threats only)
glm_threat_effects = {}
for var in ['seconds_after_rat_arrival', 'rat_minutes', 'rat_arrival_number']:
    coef = glm_results.params.get(var, float('nan'))
    p_val = glm_results.pvalues.get(var, float('nan'))
    glm_threat_effects[var] = {'coefficient': coef, 'p_value': p_val}
    print(f"{var:30} β={coef:+.6f}  p={p_val:.4f}")
for var in ['is_spring', 'rat_minutes_is_spring', 'rat_arrival_number_is_spring']:
    if var in glm_results.params.index:
        print(f"{var:30} β={glm_results.params[var]:+.6f}  p={glm_results.pvalues[var]:.4f}")

# Identify significant GLM terms among threat+season predictors
threat_and_season_terms = [
    'seconds_after_rat_arrival', 'rat_minutes', 'rat_arrival_number',
    'is_spring', 'rat_minutes_is_spring', 'rat_arrival_number_is_spring'
]
glm_significant_terms = [
    t for t in threat_and_season_terms
    if t in glm_results.params.index and glm_results.pvalues.get(t, 1.0) < 0.05
]

# Model 2: Including defensive behavior
print("\n" + "="*40)
print("MODEL 2: RAT ARRIVALS ~ CONTROLS + SEASON + SEASON×HOURS")
print("="*40)

pois_df = dataset1.copy()
if 'season_label' not in pois_df.columns:
    pois_df['season_label'] = pois_df['start_time'].apply(_season_label_from_month)
pois_df['is_spring'] = (pois_df['season_label'] == 'Spring').astype(int)
pois_df['hours_is_spring'] = pois_df['hours_after_sunset_x'] * pois_df['is_spring']

predictors_pois = ['food_availability', 'bat_landing_number', 'hours_after_sunset_x', 'is_spring', 'hours_is_spring']
pois_subset = pois_df[predictors_pois + ['rat_arrival_number']].dropna()
X_pois = sm.add_constant(pois_subset[predictors_pois])
y_pois = pois_subset['rat_arrival_number']

pois_model = sm.GLM(y_pois, X_pois, family=families.Poisson())
pois_results = pois_model.fit()
print(f"Sample size: {len(pois_subset)}")
print(f"Model fit: AIC={pois_results.aic:.1f}\n")
for var in predictors_pois:
    print(f"{var:30} β={pois_results.params[var]:+.6f}  p={pois_results.pvalues[var]:.4f}")

# === Phase 5 Visualizations: GLM Overview ===
# Verdict helper for GLM coefficients (binary outcome: reward)
def verdict_glm(var, coef, p):
    if p is None or np.isnan(p) or p >= 0.05:
        return 'Not significant'
    return 'Significant'

# Pretty names for axes
glm_pretty = {
    'seconds_after_rat_arrival': 'Seconds After Rat Arrival (lower = closer)',
    'rat_minutes': 'Rat Minutes per Period',
    'rat_arrival_number': 'Number of Rat Arrivals',
    'defensive': 'Defensive Behavior (0/1)',
    'is_spring': 'Season (Spring=1, Winter=0)',
    'rat_minutes_is_spring': 'Interaction: Rat Minutes × Spring',
    'rat_arrival_number_is_spring': 'Interaction: Arrivals × Spring'
}

# Create combined figure (2x3): three threat predictors + defensive + model info panel
fig_glm, axs_glm = plt.subplots(2, 3, figsize=(20, 12), facecolor='white')
fig_glm.suptitle('Phase 5: GLM Effects on Feeding Success', fontsize=18, fontweight='bold')

from matplotlib.patches import Patch
legend_handles_glm = [
    Patch(color=colors_map['Significant'], label='Significant (p < 0.05).'),
    Patch(color=colors_map['Not significant'], label='Not significant (p ≥ 0.05).')
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

def draw_glm_binary_panel(ax, model, predictors_list, bin_name, data_df, title_prefix):
    base = data_df[predictors_list].mean()
    X0 = base.copy(); X0[bin_name] = 0
    X1 = base.copy(); X1[bin_name] = 1
    Xp = sm.add_constant(pd.DataFrame([X0, X1]), has_constant='add')
    y_hat = model.predict(Xp)
    ax.bar(['Winter (0)', 'Spring (1)'], y_hat, color=['#607d8b','#c62828'], alpha=0.9)
    coef = model.params.get(bin_name, _np.nan)
    pval = model.pvalues.get(bin_name, _np.nan)
    ax.set_title(f"{title_prefix}\n(β={coef:+.4f}, p={pval:.4f})", fontweight='bold')
    ax.set_ylabel('Predicted Feeding Success')
    ax.set_ylim(0, 1)
    annotate(ax, verdict_glm(bin_name, coef, pval))

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

# Fourth panel: plot the first significant season-related term if present
axs_glm[1, 0].cla()
fourth_term = None
for cand in ['is_spring', 'rat_minutes_is_spring', 'rat_arrival_number_is_spring']:
    if cand in glm_significant_terms:
        fourth_term = cand
        break
if fourth_term == 'is_spring':
    draw_glm_binary_panel(axs_glm[1, 0], glm_results, main_predictors, 'is_spring', main_df, 'GLM: Season (Spring vs Winter)')
elif fourth_term in ['rat_minutes_is_spring', 'rat_arrival_number_is_spring']:
    # Need this term present in predictors for plotting line; build an augmented predictors list
    aug_predictors = main_predictors.copy()
    if fourth_term not in aug_predictors:
        aug_predictors.append(fourth_term)
    draw_glm_panel(axs_glm[1, 0], glm_results, aug_predictors, fourth_term, main_df.assign(**{fourth_term: main_df['rat_minutes'] if 'minutes' in fourth_term else main_df['rat_arrival_number']}), 'GLM: Season Interaction')
else:
    axs_glm[1, 0].axis('off')

# Model info / coefficients table (text panel)
axs_glm[1, 1].axis('off')
lines = [
    'Model 1 (Threat Predictors):',
    f"  β(sec after rat) = {glm_results.params['seconds_after_rat_arrival']:+.4f} (p={glm_results.pvalues['seconds_after_rat_arrival']:.4f})",
    f"  β(rat minutes)   = {glm_results.params['rat_minutes']:+.4f} (p={glm_results.pvalues['rat_minutes']:.4f})",
    f"  β(arrivals)       = {glm_results.params['rat_arrival_number']:+.4f} (p={glm_results.pvalues['rat_arrival_number']:.4f})",
    '',
    'Model 2 (Including Behavior):',
    '  (season and interactions included in Model 1)',
    '',
    f"Significant terms (p<0.05): {', '.join(glm_significant_terms) if len(glm_significant_terms) else 'None'}"
]
axs_glm[1, 1].text(0.0, 1.0, "\n".join(lines), va='top', ha='left', fontsize=12)

# Empty / spare panel for future or leave blank
axs_glm[1, 2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
phase5_overview = os.path.join(plots_dir, 'Phase5_GLM_Effects_on_Feeding_Success.png')
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
print("EVIDENCE COMPILATION (Seasonal)")
print("="*40)

# Determine seasonal direction for each hypothesis from Phase 4 results
spring_higher = []
winter_higher = []
no_diff = []

def _season_dir_for(k):
    v = results.get(k, {})
    if not v:
        return ('No seasonal difference', np.nan)
    p = adj_map[k]['p_adj'] if k in adj_map else v.get('p', np.nan)
    w_val = v.get('w_med', v.get('w_prop', np.nan))
    s_val = v.get('s_med', v.get('s_prop', np.nan))
    if pd.isna(p) or pd.isna(w_val) or pd.isna(s_val):
        return ('No seasonal difference', p)
    if p < 0.05:
        return ('Higher in Spring' if s_val > w_val else 'Higher in Winter', p)
    return ('No seasonal difference', p)

_keys_interest = [k for k in ['H1_arrivals','H2_minutes','H3_vigilance','H4_success','H5_risk','H6_defensive'] if k in results]
season_dir_map = {}
for k in _keys_interest:
    d, p = _season_dir_for(k)
    season_dir_map[k] = {'verdict': d, 'p': p}
    if d == 'Higher in Spring':
        spring_higher.append(k)
    elif d == 'Higher in Winter':
        winter_higher.append(k)
    else:
        no_diff.append(k)

total_sig = len(spring_higher) + len(winter_higher)

print(f"\nSeasonal evidence summary:")
print(f"  Higher in Spring (significant): {len(spring_higher)}")
print(f"  Higher in Winter (significant): {len(winter_higher)}")
print(f"  No seasonal difference: {len(no_diff)}")

# FINAL ANSWER WITH DETAILED EXPLANATION
print("\n" + "="*40)
print("FINAL ANSWER TO INVESTIGATION B")
print("="*40)
print("Do behaviours change with seasonal conditions (Winter vs Spring)?")
print()

if total_sig >= 2:
    print("ANSWER: YES - EVIDENCE OF SEASONAL DIFFERENCES")
    print(f"Details: {len(spring_higher)} higher in Spring, {len(winter_higher)} higher in Winter; {len(no_diff)} no difference.")
elif total_sig == 1:
    print("ANSWER: LIMITED EVIDENCE OF SEASONAL DIFFERENCE")
    print(f"Details: {len(spring_higher)} higher in Spring, {len(winter_higher)} higher in Winter; {len(no_diff)} no difference.")
else:
    print("ANSWER: NO - NO CLEAR SEASONAL DIFFERENCE")
    print("Details: All primary measures show no significant seasonal difference.")

print("Interpretation:")
if total_sig == 0:
    print("Seasonal effects are not evident; behaviours appear stable across Winter and Spring.")
else:
    if len(spring_higher) >= len(winter_higher):
        print("On balance, behaviours tend to be higher in Spring for the significant measures.")
    else:
        print("On balance, behaviours tend to be higher in Winter for the significant measures.")

print("\n" + "="*40)
print("STATISTICAL SUMMARY (Seasonal)")
print("="*40)
print(f"Seasonal hypotheses evaluated: {len(_keys_interest)}")
print(f"Significant seasonal differences: {total_sig}")
print(f"Higher in Spring: {len(spring_higher)} | Higher in Winter: {len(winter_higher)} | No difference: {len(no_diff)}")

# === Phase 6 Visualization: Final Summary ===
fig6, axs6 = plt.subplots(2, 3, figsize=(22, 13), facecolor='white')
fig6.suptitle('Phase 6: Final Summary and Verdict', fontsize=18, fontweight='bold')

from matplotlib.patches import Patch
legend_handles_phase6 = [
    Patch(color=colors_map['Higher in Spring'], label='Higher in Spring (p < 0.05)'),
    Patch(color=colors_map['Higher in Winter'], label='Higher in Winter (p < 0.05)'),
    Patch(color=colors_map['No seasonal difference'], label='No seasonal difference (p ≥ 0.05)')
]
fig6.legend(handles=legend_handles_phase6, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=1, frameon=True)

# Panel A: Evidence tally
bars_tally = axs6[0, 0].bar(['Higher in Spring', 'Higher in Winter'],
               [len(spring_higher), len(winter_higher)],
               color=[colors_map['Higher in Spring'], colors_map['Higher in Winter']], alpha=0.85)
axs6[0, 0].set_title('Seasonal Evidence Tally', fontweight='bold')
axs6[0, 0].set_ylabel('Count of significant indicators')
tally_max = max(len(spring_higher), len(winter_higher))
axs6[0, 0].set_ylim(0, tally_max + 1.5)
for i, bar in enumerate(bars_tally):
    height = bar.get_height()
    axs6[0, 0].text(bar.get_x() + bar.get_width()/2, height + 0.2, str(int(height)), ha='center', va='bottom', fontweight='bold')

# Panel B: Key GLM effects (bars colored by verdict)
glm_names = ['seconds_after_rat_arrival', 'rat_minutes', 'rat_arrival_number']
glm_labels = ['Proximity', 'Intensity', 'Frequency']
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

# Panel C: Hypotheses verdict grid (seasonal)
axs6[0, 2].axis('off')
labels_map = {
    'H1_arrivals': 'H1 Arrivals (rat_arrival_number)',
    'H2_minutes': 'H2 Intensity (rat_minutes)',
    'H3_vigilance': 'H3 Vigilance (landing→food)',
    'H4_success': 'H4 Success (reward)',
    'H5_risk': 'H5 Risk (0/1)',
    'H6_defensive': 'H6 Defensive (0/1)'
}
grid_items = [(labels_map[k], season_dir_map.get(k, {}).get('verdict', 'No seasonal difference')) for k in _keys_interest]

ypos = 0.95
for label, verdict in grid_items:
    axs6[0, 2].text(0.02, ypos, f"{label}: {verdict}", fontsize=10, fontweight='bold', ha='left', va='top',
                    color=colors_map.get(verdict, '#000000'),
                    bbox=dict(facecolor='white', edgecolor=colors_map.get(verdict, '#000000'), boxstyle='round,pad=0.3'))
    ypos -= 0.10
axs6[0, 2].set_title('Hypotheses Verdicts', fontweight='bold')

# Panel D: Final answer text (seasonal)
axs6[1, 0].axis('off')
final_lines = []
if total_sig >= 2:
    final_lines.append('ANSWER: YES - EVIDENCE OF SEASONAL DIFFERENCES')
elif total_sig == 1:
    final_lines.append('ANSWER: LIMITED EVIDENCE OF SEASONAL DIFFERENCE')
else:
    final_lines.append('ANSWER: NO - NO CLEAR SEASONAL DIFFERENCE')

final_lines.append('Interpretation:')
if total_sig == 0:
    final_lines.append('Behaviours appear stable across Winter and Spring.')
elif len(spring_higher) >= len(winter_higher):
    final_lines.append('On balance, higher in Spring among significant measures.')
else:
    final_lines.append('On balance, higher in Winter among significant measures.')

axs6[1, 0].text(0.0, 1.0, "\n".join(final_lines), va='top', ha='left', fontsize=14, fontweight='bold')

# Panel E: Statistical summary (compact)
axs6[1, 1].axis('off')
stats_lines = [
    f"Seasonal hypotheses evaluated: {len(_keys_interest)}",
    f"Significant seasonal differences: {total_sig}",
    f"Higher in Spring: {len(spring_higher)}",
    f"Higher in Winter: {len(winter_higher)}",
    f"No seasonal difference: {len(no_diff)}",
]
axs6[1, 1].text(0.0, 1.0, "\n".join(stats_lines), va='top', ha='left', fontsize=12)

# Panel F: Evidence direction pie chart
axs6[1, 2].pie([max(len(spring_higher), 0.0001), max(len(winter_higher), 0.0001)],
               labels=['Higher in Spring', 'Higher in Winter'], autopct='%1.0f%%', startangle=90,
               colors=[colors_map['Higher in Spring'], colors_map['Higher in Winter']])
axs6[1, 2].axis('equal')
axs6[1, 2].set_title('Direction of Significant Seasonal Differences', fontweight='bold')

plt.tight_layout(rect=[0, 0.06, 1, 0.93])
phase6_summary = os.path.join(plots_dir, 'Phase6_Final_Summary_and_Verdict.png')
plt.savefig(phase6_summary, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved Phase 6 final summary plot to: {phase6_summary}")
# %%
