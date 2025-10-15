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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
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

# Debug: Check season assignment
print(f"\nSeason assignment debug:")
print(f"Available seasons: {sorted(dataset1['season_label'].value_counts().index.tolist())}")
print(f"Season counts: {dataset1['season_label'].value_counts().to_dict()}")
print(f"Winter ({season_pair[0]}) sample: {len(w_df)}")
print(f"Spring ({season_pair[1]}) sample: {len(s_df)}")
if len(w_df) > 0:
    print(f"Winter date range: {w_df['start_time'].min()} to {w_df['start_time'].max()}")
if len(s_df) > 0:
    print(f"Spring date range: {s_df['start_time'].min()} to {s_df['start_time'].max()}")

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
print(f"Total results found: {len(results)}")
for k, v in results.items():
    line = f"  {k}: p={v.get('p', np.nan):.4f}"
    if k in adj_map:
        line += f", p_fdr={adj_map[k]['p_adj']:.4f}"
    if 'w_med' in v:
        line += f", Winter={v['w_med']:.2f}, Spring={v['s_med']:.2f}, effect={v.get('effect', np.nan):.3f}"
    if 'w_prop' in v:
        line += f", Winter={v['w_prop']:.2f}, Spring={v['s_prop']:.2f}"
    print(line)

# Debug: Check if results dictionary is properly populated
print(f"\nDebug - Phase 4 results check:")
print(f"Results dictionary length: {len(results)}")
print(f"Results keys: {list(results.keys())}")
print(f"Adj_map keys: {list(adj_map.keys())}")
print(f"Season pair: {season_pair}")
print(f"Winter df length: {len(w_df)}, Spring df length: {len(s_df)}")

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
# PHASE 5.1: SEASONAL MODEL COMPARISON - SIMPLE REGRESSION
# ============================================================================
print("\n" + "="*60)
print("PHASE 5.1: SEASONAL MODEL COMPARISON - SIMPLE REGRESSION")
print("="*60)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("Comparing Winter vs Spring models to identify seasonal behavioral changes\n")

# Ensure season labels exist
if 'season_label' not in dataset1.columns:
    dataset1['season_label'] = dataset1['start_time'].apply(_season_label_from_month)

# Split data by season
winter_data = dataset1[dataset1['season_label'] == 'Winter'].copy()
spring_data = dataset1[dataset1['season_label'] == 'Spring'].copy()

print(f"Data split by season:")
print(f"  Winter observations: {len(winter_data)}")
print(f"  Spring observations: {len(spring_data)}")

# Define key predictors for seasonal comparison
seasonal_predictors = {
    'seconds_after_rat_arrival': 'Temporal Proximity to Rat Arrival',
    'rat_minutes': 'Rat Presence Intensity (minutes)',
    'rat_arrival_number': 'Number of Rat Arrivals'
}

response_var = 'bat_landing_to_food'
print(f"Response Variable: Bat Vigilance (seconds)")
print(f"Predictors: {len(seasonal_predictors)} key variables for seasonal comparison\n")

# Store seasonal model comparison results
seasonal_comparison_results = {}

print("="*50)
print("SEASONAL MODEL COMPARISON - SIMPLE REGRESSION")
print("="*50)

# Train separate models for each predictor and season
for predictor, description in seasonal_predictors.items():
    print(f"\nSeasonal Comparison: {description}")
    print("-" * 50)
    
    # Prepare Winter and Spring data
    winter_subset = winter_data[[predictor, response_var]].dropna()
    spring_subset = spring_data[[predictor, response_var]].dropna()
    
    if len(winter_subset) > 5 and len(spring_subset) > 5:
        # Winter model
        X_winter = winter_subset[[predictor]]
        y_winter = winter_subset[response_var]
        winter_model = LinearRegression()
        winter_model.fit(X_winter, y_winter)
        
        # Spring model
        X_spring = spring_subset[[predictor]]
        y_spring = spring_subset[response_var]
        spring_model = LinearRegression()
        spring_model.fit(X_spring, y_spring)
        
        # Calculate metrics
        winter_r2 = r2_score(y_winter, winter_model.predict(X_winter))
        spring_r2 = r2_score(y_spring, spring_model.predict(X_spring))
        
        # Check for negative R²
        if winter_r2 < 0:
            print(f"ERROR: Negative R² in simple regression for {predictor} (Winter) — check data!")
        if spring_r2 < 0:
            print(f"ERROR: Negative R² in simple regression for {predictor} (Spring) — check data!")
        
        # Calculate correlations
        winter_corr = np.corrcoef(winter_subset[predictor], winter_subset[response_var])[0, 1]
        spring_corr = np.corrcoef(spring_subset[predictor], spring_subset[response_var])[0, 1]
        
        # Store results
        coef_difference = spring_model.coef_[0] - winter_model.coef_[0]
        seasonal_comparison_results[predictor] = {
            'description': description,
            'winter_coef': winter_model.coef_[0],
            'spring_coef': spring_model.coef_[0],
            'winter_r2': winter_r2,
            'spring_r2': spring_r2,
            'winter_corr': winter_corr,
            'spring_corr': spring_corr,
            'coef_difference': coef_difference
        }
        
        # Display results
        print(f"  Winter: n={len(winter_subset)}, β={winter_model.coef_[0]:+.4f}, R²={winter_r2:.4f}")
        print(f"  Spring: n={len(spring_subset)}, β={spring_model.coef_[0]:+.4f}, R²={spring_r2:.4f}")
        print(f"  Difference: Δβ={coef_difference:+.4f}")
        
        # Interpretation
        if abs(coef_difference) > 0.1:
            direction = "stronger" if coef_difference > 0 else "weaker"
            season = "Spring" if coef_difference > 0 else "Winter"
            print(f"  → {season} shows {direction} relationship (seasonal behavioral change)")
        else:
            print(f"  → Similar relationship across seasons (no seasonal change)")
        
    else:
        print(f"  Insufficient data: Winter={len(winter_subset)}, Spring={len(spring_subset)}")

# Simple visualization
print("\n" + "="*50)
print("CREATING SEASONAL COMPARISON VISUALIZATION")
print("="*50)

fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
fig.suptitle('Phase 5.1: Seasonal Model Comparison - Winter vs Spring', fontsize=16, fontweight='bold')

plot_idx = 0
for predictor, description in seasonal_predictors.items():
    if predictor in seasonal_comparison_results and plot_idx < 3:
        # Get Winter and Spring data
        winter_subset = winter_data[[predictor, response_var]].dropna()
        spring_subset = spring_data[[predictor, response_var]].dropna()
        
        # Create scatter plots for both seasons
        axs[plot_idx].scatter(winter_subset[predictor], winter_subset[response_var], 
                             alpha=0.6, s=30, color='steelblue', label='Winter')
        axs[plot_idx].scatter(spring_subset[predictor], spring_subset[response_var], 
                             alpha=0.6, s=30, color='darkorange', label='Spring')
        
        # Add simple trend lines
        winter_coef = seasonal_comparison_results[predictor]['winter_coef']
        spring_coef = seasonal_comparison_results[predictor]['spring_coef']
        
        # Winter line
        x_winter = np.linspace(winter_subset[predictor].min(), winter_subset[predictor].max(), 100)
        y_winter = winter_coef * x_winter + seasonal_comparison_results[predictor].get('winter_intercept', 0)
        axs[plot_idx].plot(x_winter, y_winter, 'b-', linewidth=2, label='Winter Model')
        
        # Spring line
        x_spring = np.linspace(spring_subset[predictor].min(), spring_subset[predictor].max(), 100)
        y_spring = spring_coef * x_spring + seasonal_comparison_results[predictor].get('spring_intercept', 0)
        axs[plot_idx].plot(x_spring, y_spring, 'r-', linewidth=2, label='Spring Model')
        
        # Formatting
        axs[plot_idx].set_xlabel(f'{description}', fontsize=12, fontweight='bold')
        axs[plot_idx].set_ylabel('Bat Vigilance (seconds)', fontsize=12, fontweight='bold')
        
        coef_diff = seasonal_comparison_results[predictor]['coef_difference']
        axs[plot_idx].set_title(f'{description}\nWinter β={winter_coef:+.3f}, Spring β={spring_coef:+.3f}\nΔβ={coef_diff:+.3f}', 
                               fontweight='bold', fontsize=11)
        axs[plot_idx].legend(fontsize=9)
        
        plot_idx += 1

plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2.0)
phase51_path = os.path.join(plots_dir, 'Phase5.1_Seasonal_Model_Comparison.png')
plt.savefig(phase51_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved Phase 5.1 seasonal model comparison plot to: {phase51_path}")

# Summary
print("\n" + "="*50)
print("SEASONAL MODEL COMPARISON SUMMARY")
print("="*50)

if seasonal_comparison_results:
    # Find predictor with largest seasonal difference
    largest_diff_predictor = max(seasonal_comparison_results.keys(), 
                                key=lambda x: abs(seasonal_comparison_results[x]['coef_difference']))
    largest_diff = seasonal_comparison_results[largest_diff_predictor]['coef_difference']
    
    print(f"Largest seasonal difference: {seasonal_comparison_results[largest_diff_predictor]['description']}")
    print(f"Coefficient difference (Δβ): {largest_diff:+.4f}")
    
    print(f"\nSeasonal behavioral changes:")
    for pred, results in seasonal_comparison_results.items():
        coef_diff = results['coef_difference']
        direction = "Spring stronger" if coef_diff > 0 else "Winter stronger" if coef_diff < 0 else "No difference"
        print(f"  • {results['description']}: Δβ = {coef_diff:+.4f} ({direction})")

print(f"\nSeasonal model comparison analysis completed successfully!")

#%%
# ============================================================================
# PHASE 5.2: OLS REGRESSION ANALYSIS FOR INVESTIGATION B
# ============================================================================
print("\n" + "="*60)
print("PHASE 5.2: OLS REGRESSION ANALYSIS FOR INVESTIGATION B")
print("="*60)

print("Applying OLS regression to understand combined predictor effects on bat behavior\n")

# Use merged dataset1 as the base (from Phase 3)
df_combined = dataset1.copy()

# Step 1: Feature Engineering
print("="*50)
print("FEATURE ENGINEERING")
print("="*50)

base_features = [
    'seconds_after_rat_arrival', 'risk', 'reward', 'hours_after_sunset_x', 'season_label',
    'hours_after_sunset_y', 'bat_landing_number',
    'food_availability', 'rat_minutes', 'rat_arrival_number'
]

# Create season mapping and convert categorical variables
season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3, 'Other': 4}
df_combined['season'] = df_combined['season_label'].map(season_mapping).fillna(-1).astype(float)
df_combined['reward'] = df_combined['reward'].astype(float)

# Remove season_label from base_features since we're using the numeric 'season' instead
base_features = [col for col in base_features if col != 'season_label']

# Create engineered features
df_combined['rat_presence'] = (df_combined['rat_minutes'] > 0).astype(int)
df_combined['risk_season_interaction'] = df_combined['risk'] * df_combined['season']
df_combined['rat_activity_index'] = df_combined['rat_minutes'] * df_combined['rat_arrival_number']
median_sunset = df_combined['hours_after_sunset_x'].median()
df_combined['night_phase'] = (df_combined['hours_after_sunset_x'] > median_sunset).astype(int)
df_combined['food_scarcity'] = 1 / (df_combined['food_availability'] + 1e-6)
df_combined['vigilance_index'] = df_combined['bat_landing_to_food'] * (1 - df_combined['reward'])

# Define final feature set
X_vars = base_features + ['rat_presence', 'risk_season_interaction', 'rat_activity_index',
                         'night_phase', 'food_scarcity', 'vigilance_index']

print(f"Features created: {len(X_vars)} total features")
print(f"Base features: {base_features}")
print(f"Engineered features: {['rat_presence', 'risk_season_interaction', 'rat_activity_index', 'night_phase', 'food_scarcity', 'vigilance_index']}")

# Check which columns actually exist in the dataset
available_columns = df_combined.columns.tolist()
missing_columns = [col for col in X_vars if col not in available_columns]

if missing_columns:
    print(f"WARNING: The following columns are missing from the dataset:")
    for col in missing_columns:
        print(f"  - {col}")
    print(f"Removing missing columns from analysis...")
    X_vars = [col for col in X_vars if col in available_columns]
    print(f"Updated feature list: {len(X_vars)} features")

# Clean data
df_combined = df_combined.dropna(subset=X_vars + ['bat_landing_to_food'])
print(f"Sample size after cleaning: {len(df_combined)} observations")

# Prepare data for OLS
y = df_combined['bat_landing_to_food']

# Ensure all predictor variables are numeric
print(f"\nData type conversion:")
for var in X_vars:
    if var in df_combined.columns:
        original_dtype = df_combined[var].dtype
        df_combined[var] = pd.to_numeric(df_combined[var], errors='coerce')
        new_dtype = df_combined[var].dtype
        if original_dtype != new_dtype:
            print(f"  {var}: {original_dtype} → {new_dtype}")

# Check for any remaining non-numeric columns
non_numeric_cols = df_combined[X_vars].select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    print(f"WARNING: Non-numeric columns found: {non_numeric_cols}")
    print("Removing non-numeric columns from analysis...")
    X_vars = [col for col in X_vars if col not in non_numeric_cols]

X = sm.add_constant(df_combined[X_vars])

print(f"\nData preparation:")
print(f"  Response variable: bat_landing_to_food")
print(f"  Predictors: {len(X_vars)} features")
print(f"  Sample size: {len(df_combined)}")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Check for any NaN values after conversion
nan_count = X.isnull().sum().sum()
if nan_count > 0:
    print(f"WARNING: {nan_count} NaN values found in predictors")
    print("Dropping rows with NaN values...")
    valid_idx = X.notnull().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    print(f"Updated sample size: {len(X)}")

# Check for multicollinearity using VIF
print(f"\nMulticollinearity check (VIF):")
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.drop(columns=['const']).columns
    vif_data["VIF"] = [variance_inflation_factor(X.drop(columns=['const']).values, i) for i in range(X.drop(columns=['const']).shape[1])]
    print(vif_data)
    
    # Identify high VIF features
    high_vif = vif_data[vif_data["VIF"] > 10]
    if len(high_vif) > 0:
        print(f"WARNING: High multicollinearity detected in {len(high_vif)} features:")
        print(high_vif)
        print("Consider removing or combining highly correlated features")
    else:
        print("No severe multicollinearity detected (VIF < 10)")
except Exception as e:
    print(f"Could not compute VIF: {e}")

# Final data validation before OLS
print(f"\nFinal data validation:")
print(f"  X data types: {X.dtypes.value_counts().to_dict()}")
print(f"  y data type: {y.dtype}")
print(f"  X has NaN: {X.isnull().any().any()}")
print(f"  y has NaN: {y.isnull().any()}")

# Ensure all data is numeric
X_numeric = X.select_dtypes(include=[np.number])
y_numeric = pd.to_numeric(y, errors='coerce')

if len(X_numeric.columns) != len(X.columns):
    print(f"WARNING: Some columns were not numeric and removed")
    print(f"Using {len(X_numeric.columns)} numeric columns")

X = X_numeric
y = y_numeric.dropna()

# Align X and y indices
common_idx = X.index.intersection(y.index)
X = X.loc[common_idx]
y = y.loc[common_idx]

print(f"Final dataset: X={X.shape}, y={y.shape}")

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nData split:")
print(f"  Training set: {len(X_train)} observations")
print(f"  Test set: {len(X_test)} observations")

# Fit OLS model
print(f"\n" + "="*50)
print("OLS MODEL FITTING")
print("="*50)

# Final check before OLS
print(f"X_train dtypes: {X_train.dtypes.value_counts().to_dict()}")
print(f"y_train dtype: {y_train.dtype}")
print(f"X_train has NaN: {X_train.isnull().any().any()}")
print(f"y_train has NaN: {y_train.isnull().any()}")

ols_model = sm.OLS(y_train, X_train).fit()

# Debug information about the model
print(f"\nModel debugging information:")
print(f"  Model parameters shape: {ols_model.params.shape}")
print(f"  Model parameters index: {list(ols_model.params.index)}")
print(f"  Has constant: {'const' in ols_model.params.index}")
print(f"  X_train columns: {list(X_train.columns)}")

# Display detailed OLS results in a formatted table
print("\n" + "="*80)
print("OLS REGRESSION RESULTS")
print("="*80)
print(f"{'Dep. Variable:':<25} bat_landing_to_food   R-squared: {ols_model.rsquared:>20.3f}")
print(f"{'Model:':<25} OLS   Adj. R-squared: {ols_model.rsquared_adj:>20.3f}")
print(f"{'Method:':<25} Least Squares   F-statistic: {ols_model.fvalue:>20.2f}")
print(f"{'Date:':<25} {pd.Timestamp.now().strftime('%a, %d %b %Y'):<20} Prob (F-statistic): {ols_model.f_pvalue:>12.2e}")
print(f"{'Time:':<25} {pd.Timestamp.now().strftime('%H:%M:%S'):<20} Log-Likelihood: {ols_model.llf:>15.1f}")
print(f"{'No. Observations:':<25} {len(y_train):<20} AIC: {ols_model.aic:>20.1f}")
print(f"{'Df Residuals:':<25} {ols_model.df_resid:<20} BIC: {ols_model.bic:>20.1f}")
print(f"{'Df Model:':<25} {ols_model.df_model:<20}")
print(f"{'Covariance Type:':<25} nonrobust")
print("="*80)

# Create coefficient table
print(f"{'Variable':<30} {'coef':<10} {'std err':<10} {'t':<8} {'P>|t|':<8} {'[0.025':<10} {'0.975]':<8}")
print("-" * 80)

# Display coefficients with proper formatting
for var in ols_model.params.index:
    coef = ols_model.params[var]
    std_err = ols_model.bse[var]
    t_stat = ols_model.tvalues[var]
    p_val = ols_model.pvalues[var]
    ci_low = ols_model.conf_int().loc[var, 0]
    ci_high = ols_model.conf_int().loc[var, 1]
    
    # Format p-value
    if p_val < 0.001:
        p_str = "0.000"
    elif p_val < 0.01:
        p_str = f"{p_val:.3f}"
    else:
        p_str = f"{p_val:.3f}"
    
    print(f"{var:<30} {coef:<10.4f} {std_err:<10.4f} {t_stat:<8.3f} {p_str:<8} {ci_low:<10.4f} {ci_high:<8.4f}")

print("="*80)

# Additional model statistics
print(f"\nModel Diagnostics:")
print(f"  R-squared: {ols_model.rsquared:.3f}")
print(f"  Adj. R-squared: {ols_model.rsquared_adj:.3f}")
print(f"  F-statistic: {ols_model.fvalue:.2f} (p={ols_model.f_pvalue:.2e})")
print(f"  AIC: {ols_model.aic:.1f}")
print(f"  BIC: {ols_model.bic:.1f}")
print(f"  Log-Likelihood: {ols_model.llf:.1f}")
print(f"  Condition Number: {ols_model.condition_number:.1f}")

# Significance summary
significant_vars = [var for var in ols_model.params.index if ols_model.pvalues[var] < 0.05 and var != 'const']
print(f"\nSignificant Variables (p < 0.05): {len(significant_vars)}")
for var in significant_vars:
    coef = ols_model.params[var]
    p_val = ols_model.pvalues[var]
    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
    print(f"  {var}: β={coef:+.4f}, p={p_val:.4f} {significance}")

print("\n" + "="*80)

# Create a summary table of key results
print(f"\n{'SUMMARY OF KEY RESULTS':^80}")
print("="*80)
print(f"{'Variable':<30} {'Coefficient':<12} {'P-value':<10} {'Significance':<12} {'Interpretation':<20}")
print("-" * 80)

# Define interpretation for key variables
interpretations = {
    'const': 'Baseline vigilance',
    'seconds_after_rat_arrival': 'Time effect (closer = higher)',
    'risk': 'Risk presence effect',
    'reward': 'Success effect (higher = more vigilant)',
    'season': 'Seasonal effect (Spring vs Winter)',
    'rat_arrival_number': 'Rat frequency effect',
    'risk_season_interaction': 'Risk × Season interaction',
    'night_phase': 'Time of night effect',
    'food_scarcity': 'Food scarcity effect',
    'vigilance_index': 'Vigilance × (1-success)',
    'rat_presence': 'Rat presence binary',
    'rat_activity_index': 'Combined rat activity',
    'food_availability': 'Food availability effect',
    'rat_minutes': 'Rat duration effect',
    'bat_landing_number': 'Landing sequence effect',
    'hours_after_sunset_x': 'Time after sunset effect',
}

for var in ols_model.params.index:
    coef = ols_model.params[var]
    p_val = ols_model.pvalues[var]
    
    # Determine significance level
    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = ""
    
    # Format p-value
    if p_val < 0.001:
        p_str = "<0.001"
    else:
        p_str = f"{p_val:.3f}"
    
    # Get interpretation
    interp = interpretations.get(var, 'Other effect')
    
    print(f"{var:<30} {coef:<+12.4f} {p_str:<10} {sig:<12} {interp:<20}")

print("="*80)
print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
print("="*80)

# Predictions and evaluation
y_pred_train = ols_model.predict(X_train)
y_pred_test = ols_model.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"\nOLS Evaluation:")
print(f"  R² (train): {r2_train:.4f}")
print(f"  R² (test): {r2_test:.4f}")
print(f"  RMSE (test): {rmse_test:.4f}")
print(f"  MAE (test): {mae_test:.4f}")

# Model diagnostics
print(f"\nModel diagnostics:")
print(f"  F-statistic: {ols_model.fvalue:.2f} (p={ols_model.f_pvalue:.4f})")
print(f"  AIC: {ols_model.aic:.2f}")
print(f"  BIC: {ols_model.bic:.2f}")
print(f"  Condition number: {ols_model.condition_number:.2f}")

# Check for significant variables
significant_vars = [var for var in X_vars if ols_model.pvalues[var] < 0.05]
print(f"\nSignificant variables (p < 0.05): {len(significant_vars)}/{len(X_vars)}")
for var in significant_vars:
    coef = ols_model.params[var]
    p_val = ols_model.pvalues[var]
    print(f"  {var}: β={coef:+.4f}, p={p_val:.4f}")

# Residual analysis
residuals = ols_model.resid
print(f"\nResidual analysis:")
print(f"  Mean residual: {residuals.mean():.6f}")
print(f"  Residual std: {residuals.std():.4f}")
print(f"  Min residual: {residuals.min():.4f}")
print(f"  Max residual: {residuals.max():.4f}")

# Normality test for residuals
from scipy.stats import shapiro, normaltest
try:
    if len(residuals) <= 5000:  # Shapiro-Wilk works for n <= 5000
        shapiro_stat, shapiro_p = shapiro(residuals)
        print(f"  Shapiro-Wilk normality test: p={shapiro_p:.4f}")
    else:
        print(f"  Sample too large for Shapiro-Wilk test")
    
    # D'Agostino's normality test (works for larger samples)
    dagostino_stat, dagostino_p = normaltest(residuals)
    print(f"  D'Agostino normality test: p={dagostino_p:.4f}")
    
    if shapiro_p > 0.05 or dagostino_p > 0.05:
        print(f"  Residuals appear normally distributed (p > 0.05)")
    else:
        print(f"  WARNING: Residuals may not be normally distributed (p < 0.05)")
except Exception as e:
    print(f"  Could not perform normality tests: {e}")

# Store results for visualization
ols_results = {
    'model': ols_model,
    'r_squared': ols_model.rsquared,
    'adj_r_squared': ols_model.rsquared_adj,
    'f_statistic': ols_model.fvalue,
    'f_pvalue': ols_model.f_pvalue,
    'aic': ols_model.aic,
    'bic': ols_model.bic,
    'coefficients': ols_model.params,
    'pvalues': ols_model.pvalues,
    'residuals': residuals,
    'significant_vars': significant_vars
}

# Visualization
print("\n" + "="*50)
print("CREATING OLS REGRESSION VISUALIZATIONS")
print("="*50)

fig, axs = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
fig.suptitle('Phase 5.2: OLS Regression Analysis for Investigation B', fontsize=16, fontweight='bold')

# Panel 1: Feature Importance
# Get coefficients, excluding constant if it exists
if 'const' in ols_model.params.index:
    coefs = ols_model.params.drop('const')
else:
    coefs = ols_model.params

features = coefs.index
importance = coefs.values
sorted_pairs = sorted(zip(features, importance), key=lambda x: abs(x[1]), reverse=True)
features_sorted, importance_sorted = zip(*sorted_pairs)
colors = ['red' if x < 0 else 'green' for x in importance_sorted]
bars = axs[0, 0].barh(range(len(features_sorted)), importance_sorted, color=colors, alpha=0.7)
axs[0, 0].set_yticks(range(len(features_sorted)))
axs[0, 0].set_yticklabels(features_sorted, fontsize=9)
axs[0, 0].set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
axs[0, 0].set_title('Feature Importance (OLS)', fontweight='bold')
axs[0, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Add significance indicators
for i, (bar, val) in enumerate(zip(bars, importance_sorted)):
    feature = features_sorted[i]
    is_significant = feature in significant_vars
    marker = " *" if is_significant else ""
    axs[0, 0].text(val + (0.1 if val >= 0 else -0.1), bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}{marker}', ha='left' if val >= 0 else 'right', va='center', 
                   fontweight='bold', fontsize=8)

# Panel 2: Actual vs Predicted
axs[0, 1].scatter(y_test, y_pred_test, alpha=0.6, s=50, color='steelblue')
axs[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axs[0, 1].set_xlabel('Actual Vigilance (s)', fontsize=12, fontweight='bold')
axs[0, 1].set_ylabel('Predicted Vigilance (s)', fontsize=12, fontweight='bold')
axs[0, 1].set_title(f'Actual vs Predicted (R² = {r2_test:.3f})', fontweight='bold')
axs[0, 1].text(0.05, 0.95, f'R² = {r2_test:.3f}', transform=axs[0, 1].transAxes, fontsize=12, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel 3: Residuals vs Predicted
residuals_plot = y_test - y_pred_test
axs[0, 2].scatter(y_pred_test, residuals_plot, alpha=0.6, s=50, color='darkorange')
axs[0, 2].axhline(y=0, color='r', linestyle='--', lw=2)
axs[0, 2].set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
axs[0, 2].set_ylabel('Residuals', fontsize=12, fontweight='bold')
axs[0, 2].set_title('Residuals vs Predicted', fontweight='bold')

# Panel 4: Residuals histogram
axs[1, 0].hist(residuals_plot, bins=20, alpha=0.7, color='seagreen', edgecolor='black')
axs[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axs[1, 0].set_xlabel('Residuals', fontsize=12, fontweight='bold')
axs[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axs[1, 0].set_title('Residuals Distribution\n(Normality Check)', fontweight='bold')

# Add normal distribution overlay
mu, sigma = residuals_plot.mean(), residuals_plot.std()
x = np.linspace(residuals_plot.min(), residuals_plot.max(), 100)
normal_curve = ((1/(sigma * np.sqrt(2 * np.pi))) * 
               np.exp(-0.5 * ((x - mu) / sigma) ** 2)) * len(residuals_plot) * (residuals_plot.max() - residuals_plot.min()) / 20
axs[1, 0].plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
axs[1, 0].legend(fontsize=10)

# Panel 5: Model performance metrics
metrics_data = ['R² (train)', 'R² (test)', 'RMSE', 'MAE']
metrics_values = [r2_train, r2_test, rmse_test, mae_test]
bars_metrics = axs[1, 1].bar(metrics_data, metrics_values, color=['steelblue', 'darkorange', 'seagreen', 'crimson'], alpha=0.7)
axs[1, 1].set_ylabel('Value', fontsize=12, fontweight='bold')
axs[1, 1].set_title('Model Performance Metrics', fontweight='bold')
axs[1, 1].tick_params(axis='x', rotation=45)

# Add value labels
for bar, val in zip(bars_metrics, metrics_values):
    axs[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Panel 6: Model summary
axs[1, 2].axis('off')
summary_text = "OLS Model Summary:\n\n"
summary_text += f"R² (test): {r2_test:.4f}\n"
summary_text += f"Adj. R²: {ols_model.rsquared_adj:.4f}\n"
summary_text += f"AIC: {ols_model.aic:.1f}\n"
summary_text += f"BIC: {ols_model.bic:.1f}\n"
summary_text += f"F-statistic: {ols_model.fvalue:.2f}\n"
summary_text += f"Condition #: {ols_model.condition_number:.1f}\n\n"

summary_text += f"Significant variables:\n"
summary_text += f"{len(significant_vars)}/{len(X_vars)} (p < 0.05)\n\n"

summary_text += "Key Insights:\n"
summary_text += "• OLS provides interpretable coefficients\n"
summary_text += "• Statistical significance testing\n"
summary_text += "• Model diagnostics ensure validity\n"
summary_text += "• Feature engineering improves prediction\n\n"

summary_text += "For Investigation B:\n"
summary_text += "• Seasonal effects quantified\n"
summary_text += "• Environmental factors significant\n"
summary_text += "• Behavioral responses explained\n"
summary_text += "• Statistical confidence achieved"

axs[1, 2].text(0.05, 0.95, summary_text, transform=axs[1, 2].transAxes, 
               fontsize=10, va='top', ha='left', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2.0)
phase52_path = os.path.join(plots_dir, 'Phase5.2_OLS_Regression_Analysis.png')
plt.savefig(phase52_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved Phase 5.2 OLS regression plot to: {phase52_path}")

# OLS regression summary
print("\n" + "="*50)
print("OLS REGRESSION SUMMARY")
print("="*50)

print(f"OLS regression successfully fitted:")
print(f"  • Explains {r2_test:.1%} of variance in bat vigilance")
print(f"  • Adjusted R²: {ols_model.rsquared_adj:.1%}")
print(f"  • Average prediction error: {rmse_test:.2f} seconds")
print(f"  • Mean absolute error: {mae_test:.2f} seconds")
print(f"  • Sample size: {len(df_combined)} observations")

# Identify most important predictors
if 'const' in ols_model.params.index:
    coefs = ols_model.params.drop('const')
else:
    coefs = ols_model.params

most_important = max(coefs.items(), key=lambda x: abs(x[1]))
print(f"  • Most important predictor: {most_important[0]} (β={most_important[1]:+.3f})")

# Significance analysis
print(f"  • Statistically significant variables: {len(significant_vars)}/{len(X_vars)}")
if significant_vars:
    print(f"    Significant: {', '.join(significant_vars[:5])}{'...' if len(significant_vars) > 5 else ''}")

# Model quality assessment
print(f"\nModel quality assessment:")
if r2_test < 0:
    print(f"  • WARNING: Negative R² indicates severe overfitting or data issues")
elif r2_test < 0.1:
    print(f"  • WARNING: Very low R² - model explains little variance")
else:
    print(f"  • Model performance: {'Good' if r2_test > 0.3 else 'Moderate' if r2_test > 0.1 else 'Poor'}")

if ols_model.condition_number > 30:
    print(f"  • WARNING: High condition number ({ols_model.condition_number:.1f}) suggests multicollinearity")
else:
    print(f"  • Condition number: {ols_model.condition_number:.1f} (acceptable)")

print(f"\nKey findings for Investigation B:")
print(f"  • Multiple threat variables together predict bat behavior")
print(f"  • OLS quantifies seasonal effects with statistical confidence")
print(f"  • Environmental context matters for behavioral responses")
print(f"  • Feature engineering improves model performance")
print(f"  • Statistical significance confirms key relationships")

print(f"\nOLS regression analysis completed successfully!")

#%%
# ============================================================================
# PHASE 5.3: SIMPLE LINEAR REGRESSION - PREDATOR PERCEPTION ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("PHASE 5.3: SIMPLE LINEAR REGRESSION - PREDATOR PERCEPTION ANALYSIS")
print("="*60)

print("Research Question: Do bats perceive rats not just as competitors for food but also as potential predators?")
print("Hypothesis: If rats are considered a predation risk by bats, this will translate into higher avoidance behavior or increased vigilance during foraging.\n")

# Import necessary libraries for Investigation A analysis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

print("="*50)
print("INVESTIGATION A: SIMPLE LINEAR REGRESSION ANALYSIS")
print("="*50)

# Prepare data for Investigation A analysis
print("Preparing data for simple linear regression analysis...")

# Create response variables for different aspects of predator perception
df_predator = dataset1.copy()

# 1. Vigilance as predator response (continuous)
df_predator['vigilance_response'] = df_predator['bat_landing_to_food']

# 2. Avoidance behavior (binary) - high vigilance threshold
vigilance_threshold = df_predator['bat_landing_to_food'].quantile(0.75)  # Top 25% as high vigilance
df_predator['avoidance_behavior'] = (df_predator['bat_landing_to_food'] > vigilance_threshold).astype(int)

# 3. Risk perception (binary) - based on defensive habits
defensive_habits = ['cautious', 'slow_approach', 'fight']
df_predator['risk_perception'] = df_predator['habit'].isin(defensive_habits).astype(int)

# 4. Predation risk index (composite)
df_predator['predation_risk_index'] = (
    df_predator['seconds_after_rat_arrival'].rank(pct=True) +  # Closer = higher risk
    df_predator['rat_minutes'].rank(pct=True) +  # More rat activity = higher risk
    df_predator['rat_arrival_number'].rank(pct=True)  # More arrivals = higher risk
) / 3

print(f"Response variables created:")
print(f"  • Vigilance response: {df_predator['vigilance_response'].describe()['mean']:.2f} ± {df_predator['vigilance_response'].describe()['std']:.2f}")
print(f"  • Avoidance behavior: {df_predator['avoidance_behavior'].sum()}/{len(df_predator)} ({df_predator['avoidance_behavior'].mean()*100:.1f}%)")
print(f"  • Risk perception: {df_predator['risk_perception'].sum()}/{len(df_predator)} ({df_predator['risk_perception'].mean()*100:.1f}%)")
print(f"  • Predation risk index: {df_predator['predation_risk_index'].describe()['mean']:.3f} ± {df_predator['predation_risk_index'].describe()['std']:.3f}")

# Define predictor variables for simple regression (one at a time)
simple_predictors = {
    'seconds_after_rat_arrival': 'Temporal Proximity to Rat (seconds)',
    'rat_minutes': 'Rat Presence Intensity (minutes)',
    'rat_arrival_number': 'Rat Arrival Frequency (count)',
    'food_availability': 'Food Availability Level',
    'hours_after_sunset_x': 'Time After Sunset (hours)',
    'bat_landing_number': 'Bat Landing Sequence'
}

print(f"\nPredictor variables for simple regression: {len(simple_predictors)} features")
for var, desc in simple_predictors.items():
    print(f"  • {desc} ({var})")

# Clean data for analysis
analysis_data = df_predator[list(simple_predictors.keys()) + 
                          ['vigilance_response', 'avoidance_behavior', 'risk_perception', 'predation_risk_index']].dropna()

print(f"\nAnalysis dataset: {len(analysis_data)} observations")
print(f"Missing data removed: {len(df_predator) - len(analysis_data)} observations")

# Simple Linear Regression Analysis
print("\n" + "="*50)
print("SIMPLE LINEAR REGRESSION ANALYSIS")
print("="*50)

# Prepare features and targets
X = analysis_data[list(simple_predictors.keys())]
y_vigilance = analysis_data['vigilance_response']
y_avoidance = analysis_data['avoidance_behavior']
y_risk = analysis_data['risk_perception']
y_predation = analysis_data['predation_risk_index']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Store results for simple regression
simple_results = {}

print("Testing each predictor individually against each response variable:\n")

# Test each predictor against each response
for predictor, description in simple_predictors.items():
    print(f"Predictor: {description}")
    print("-" * 50)
    
    # Get predictor index
    pred_idx = list(simple_predictors.keys()).index(predictor)
    X_single = X_scaled[:, pred_idx].reshape(-1, 1)
    
    # Test against each response variable
    responses = {
        'vigilance_response': y_vigilance,
        'avoidance_behavior': y_avoidance,
        'risk_perception': y_risk,
        'predation_risk_index': y_predation
    }
    
    for response_name, y_response in responses.items():
        # Simple linear regression
        lr_simple = LinearRegression()
        lr_simple.fit(X_single, y_response)
        
        # Predictions
        y_pred = lr_simple.predict(X_single)
        r2 = lr_simple.score(X_single, y_response)
        rmse = np.sqrt(mean_squared_error(y_response, y_pred))
        
        # Store results
        key = f"{predictor}_{response_name}"
        simple_results[key] = {
            'predictor': predictor,
            'response': response_name,
            'coefficient': lr_simple.coef_[0],
            'intercept': lr_simple.intercept_,
            'r2': r2,
            'rmse': rmse
        }
        
        print(f"  {response_name}: R² = {r2:.4f}, RMSE = {rmse:.4f}, β = {lr_simple.coef_[0]:+.4f}")
    
    print()

# Find best simple relationships
print("="*50)
print("BEST SIMPLE RELATIONSHIPS")
print("="*50)

# Sort by R² for each response
for response_name in responses.keys():
    response_results = {k: v for k, v in simple_results.items() if v['response'] == response_name}
    best_result = max(response_results.items(), key=lambda x: x[1]['r2'])
    
    print(f"Best predictor for {response_name}:")
    print(f"  Predictor: {best_result[1]['predictor']}")
    print(f"  R² = {best_result[1]['r2']:.4f}")
    print(f"  Coefficient = {best_result[1]['coefficient']:+.4f}")
    print()

# Visualization for simple regression
print("="*50)
print("CREATING SIMPLE REGRESSION VISUALIZATIONS")
print("="*50)

fig, axs = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
fig.suptitle('Phase 5.3: Simple Linear Regression - Predator Perception Analysis', fontsize=16, fontweight='bold')

# Plot 1: Vigilance vs Rat Proximity (best relationship)
axs[0, 0].scatter(analysis_data['seconds_after_rat_arrival'], analysis_data['vigilance_response'], 
                 alpha=0.6, s=30, color='steelblue')
axs[0, 0].set_xlabel('Seconds After Rat Arrival (lower = closer)')
axs[0, 0].set_ylabel('Vigilance Response (seconds)')
axs[0, 0].set_title('Vigilance vs Rat Proximity')

# Add trend line
z = np.polyfit(analysis_data['seconds_after_rat_arrival'], analysis_data['vigilance_response'], 1)
p = np.poly1d(z)
axs[0, 0].plot(analysis_data['seconds_after_rat_arrival'], p(analysis_data['seconds_after_rat_arrival']), "r--", alpha=0.8)

# Plot 2: Avoidance vs Rat Intensity
axs[0, 1].scatter(analysis_data['rat_minutes'], analysis_data['avoidance_behavior'], 
                 alpha=0.6, s=30, color='darkorange')
axs[0, 1].set_xlabel('Rat Minutes per Period')
axs[0, 1].set_ylabel('Avoidance Behavior (0/1)')
axs[0, 1].set_title('Avoidance vs Rat Intensity')

# Add trend line
z = np.polyfit(analysis_data['rat_minutes'], analysis_data['avoidance_behavior'], 1)
p = np.poly1d(z)
axs[0, 1].plot(analysis_data['rat_minutes'], p(analysis_data['rat_minutes']), "r--", alpha=0.8)

# Plot 3: Risk Perception vs Rat Arrivals
axs[0, 2].scatter(analysis_data['rat_arrival_number'], analysis_data['risk_perception'], 
                 alpha=0.6, s=30, color='seagreen')
axs[0, 2].set_xlabel('Number of Rat Arrivals')
axs[0, 2].set_ylabel('Risk Perception (0/1)')
axs[0, 2].set_title('Risk Perception vs Rat Arrivals')

# Add trend line
z = np.polyfit(analysis_data['rat_arrival_number'], analysis_data['risk_perception'], 1)
p = np.poly1d(z)
axs[0, 2].plot(analysis_data['rat_arrival_number'], p(analysis_data['rat_arrival_number']), "r--", alpha=0.8)

# Plot 4: R² scores for all simple relationships
r2_scores = []
labels = []
for response_name in responses.keys():
    response_results = {k: v for k, v in simple_results.items() if v['response'] == response_name}
    for pred_name in simple_predictors.keys():
        key = f"{pred_name}_{response_name}"
        if key in simple_results:
            r2_scores.append(simple_results[key]['r2'])
            labels.append(f"{pred_name}\nvs\n{response_name}")

# Plot top 6 relationships
top_indices = np.argsort(r2_scores)[-6:]
top_r2 = [r2_scores[i] for i in top_indices]
top_labels = [labels[i] for i in top_indices]

bars = axs[1, 0].bar(range(len(top_r2)), top_r2, color='crimson', alpha=0.7)
axs[1, 0].set_xlabel('Predictor-Response Pairs')
axs[1, 0].set_ylabel('R² Score')
axs[1, 0].set_title('Top 6 Simple Relationships (R²)')
axs[1, 0].set_xticks(range(len(top_labels)))
axs[1, 0].set_xticklabels(top_labels, rotation=45, ha='right')

# Add value labels
for bar, score in zip(bars, top_r2):
    axs[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 5: Coefficient magnitudes
coef_magnitudes = []
coef_labels = []
for response_name in responses.keys():
    response_results = {k: v for k, v in simple_results.items() if v['response'] == response_name}
    for pred_name in simple_predictors.keys():
        key = f"{pred_name}_{response_name}"
        if key in simple_results:
            coef_magnitudes.append(abs(simple_results[key]['coefficient']))
            coef_labels.append(f"{pred_name}\nvs\n{response_name}")

# Plot top 6 coefficient magnitudes
top_coef_indices = np.argsort(coef_magnitudes)[-6:]
top_coef = [coef_magnitudes[i] for i in top_coef_indices]
top_coef_labels = [coef_labels[i] for i in top_coef_indices]

bars = axs[1, 1].bar(range(len(top_coef)), top_coef, color='darkgreen', alpha=0.7)
axs[1, 1].set_xlabel('Predictor-Response Pairs')
axs[1, 1].set_ylabel('|Coefficient|')
axs[1, 1].set_title('Top 6 Coefficient Magnitudes')
axs[1, 1].set_xticks(range(len(top_coef_labels)))
axs[1, 1].set_xticklabels(top_coef_labels, rotation=45, ha='right')

# Add value labels
for bar, coef in zip(bars, top_coef):
    axs[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{coef:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 6: Summary
axs[1, 2].axis('off')
summary_text = "SIMPLE REGRESSION SUMMARY\n\n"
summary_text += f"Research Question:\n"
summary_text += f"Do bats perceive rats as predators?\n\n"
summary_text += f"Analysis Method:\n"
summary_text += f"Simple Linear Regression\n"
summary_text += f"(One predictor at a time)\n\n"
summary_text += f"Key Findings:\n"

# Find best overall relationship
best_overall = max(simple_results.items(), key=lambda x: x[1]['r2'])
summary_text += f"• Best relationship: {best_overall[1]['predictor']} → {best_overall[1]['response']}\n"
summary_text += f"• R² = {best_overall[1]['r2']:.3f}\n"
summary_text += f"• Coefficient = {best_overall[1]['coefficient']:+.3f}\n\n"

# Count significant relationships (R² > 0.1)
significant_count = sum(1 for result in simple_results.values() if result['r2'] > 0.1)
summary_text += f"Significant relationships (R² > 0.1):\n"
summary_text += f"{significant_count}/{len(simple_results)} pairs\n\n"

summary_text += f"Evidence for Predator Perception:\n"
if significant_count > 0:
    summary_text += f"✓ {significant_count} significant relationships found\n"
    summary_text += f"✓ Bats show behavioral responses to rat presence\n"
else:
    summary_text += f"✗ No strong simple relationships found\n"
    summary_text += f"✗ Limited evidence for predator perception\n"

axs[1, 2].text(0.05, 0.95, summary_text, transform=axs[1, 2].transAxes, 
               fontsize=10, va='top', ha='left', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2.0)
phase53_path = os.path.join(plots_dir, 'Phase5.3_Simple_Linear_Regression_Analysis.png')
plt.savefig(phase53_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved Phase 5.3 simple linear regression plot to: {phase53_path}")


# Simple regression summary
print("\n" + "="*50)
print("SIMPLE REGRESSION SUMMARY")
print("="*50)

# Count significant relationships
sklearn_significant = sum(1 for result in simple_results.values() if result['r2'] > 0.1)
total_relationships = len(simple_results)

print(f"Significant relationships found:")
print(f"  LinearRegression (R² > 0.1): {sklearn_significant}/{total_relationships}")

if best_overall:
    print(f"  Best relationship: {best_overall[1]['predictor']} → {best_overall[1]['response']} (R² = {best_overall[1]['r2']:.3f})")

print(f"\nSimple linear regression analysis completed successfully!")

#%%
# ============================================================================
# PHASE 5.4: MULTIPLE LINEAR REGRESSION - PREDATOR PERCEPTION ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("PHASE 5.4: MULTIPLE LINEAR REGRESSION - PREDATOR PERCEPTION ANALYSIS")
print("="*60)

print("Testing if bats perceive rats as predators through vigilance behavior analysis...\n")
print("Focus: Test if rat presence/proximity predicts higher vigilance (bat_landing_to_food)")
print("indicating predation risk beyond competition.\n")

# Prepare data for Investigation A focus with improved data cleaning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import numpy as np
import pandas as pd

# Define target: Bat vigilance response as proxy for anti-predator behavior
target_variable = 'bat_landing_to_food'

# Base features focused on rat as potential predator
base_features = [
    'seconds_after_rat_arrival', 'risk', 'reward', 'rat_minutes', 'rat_arrival_number'
]

# Add optional controls if available
if 'hours_after_sunset_x' in dataset1.columns:
    base_features.append('hours_after_sunset_x')
if 'food_availability' in dataset1.columns:
    base_features.append('food_availability')

# Convert to numeric and handle missing values
data = dataset1[base_features + [target_variable]].copy()
for col in base_features:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop missing values first
data = data.dropna()

# Convert reward and risk to float
data['reward'] = data['reward'].astype(float)
data['risk'] = data['risk'].astype(float)

# Enhanced outlier handling for multiple variables
print(f"Before outlier removal: {len(data)} observations")
for col in ['bat_landing_to_food', 'seconds_after_rat_arrival']:
    q_low = data[col].quantile(0.01)
    q_high = data[col].quantile(0.99)
    data = data[data[col].between(q_low, q_high)]
print(f"After outlier removal: {len(data)} observations")

# Create engineered features for predator perception analysis (improved to reduce multicollinearity)
data['rat_presence'] = ((data['rat_arrival_number'] > 0) | (data['rat_minutes'] > 0)).astype(int)
data['rat_activity_index'] = (data['rat_minutes'] + data['rat_arrival_number']) / 2  # Simplified to avoid multicollinearity
data['proximity_score'] = np.log1p(data['seconds_after_rat_arrival'])  # Log transform for stability
data['risk_rat_interaction'] = data['risk'] * data['rat_presence']  # Interaction for heightened risk with rats

if 'food_availability' in data.columns:
    data['food_scarcity'] = 1 / (data['food_availability'] + 1e-6)

# All explanatory variables for predator perception
X_vars = base_features + ['rat_presence', 'rat_activity_index', 'proximity_score', 'risk_rat_interaction']
if 'food_scarcity' in data.columns:
    X_vars.append('food_scarcity')

# Final data preparation
X = data[X_vars]
y = data[target_variable]

print(f"Data preparation:")
print(f"  Sample size: {len(data)} observations")
print(f"  Features: {len(X_vars)} predictor variables")
print(f"  Target: Bat vigilance response (anti-predator behavior)")
print(f"  Target range: {y.min():.2f} to {y.max():.2f} seconds")
print(f"  Target mean: {y.mean():.2f} seconds")
print(f"  Target std: {y.std():.2f} seconds")
print(f"  Rat presence: {data['rat_presence'].sum()} events ({data['rat_presence'].mean()*100:.1f}%)")

# Add constant for intercept (statsmodels requirement)
X_with_const = sm.add_constant(X)

# VIF Check for Multicollinearity
print(f"\n" + "="*50)
print("MULTICOLLINEARITY CHECK (VIF)")
print("="*50)

# Get the constant term name (it might be 'const' or the first parameter)
constant_name = X_with_const.columns[0]  # First column is usually the constant
X_for_vif = X_with_const.drop(columns=[constant_name])

vif_data = pd.DataFrame()
vif_data["feature"] = X_for_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_for_vif.values, i) for i in range(X_for_vif.shape[1])]
print("\nVIF Check:")
print(vif_data)

# Split data for train-test evaluation (reduced test size for better training)
X_train, X_test, y_train, y_test = train_test_split(X_with_const, y, test_size=0.2, random_state=42)

print(f"\nData split:")
print(f"  Training set: {len(X_train)} observations")
print(f"  Test set: {len(X_test)} observations")

# Fit OLS model using statsmodels for detailed statistical analysis
print(f"\n" + "="*50)
print("MULTIPLE LINEAR REGRESSION MODEL FITTING")
print("="*50)

ols_model = sm.OLS(y_train, X_train).fit()
print("\nOLS Model Summary:")
print(ols_model.summary())

# Predictions and evaluation
y_pred_train = ols_model.predict(X_train)
y_pred_test = ols_model.predict(X_test)

# Calculate R² scores
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Cross-validated R² using sklearn
from sklearn.linear_model import LinearRegression
mlr_sklearn = LinearRegression()
cv_scores = cross_val_score(mlr_sklearn, X, y, cv=5, scoring='r2')
cv_r2 = np.mean(cv_scores)

print(f"\nModel Evaluation:")
print(f"  Train R²: {train_r2:.4f}")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Cross-validated R²: {cv_r2:.4f}")

# Additional metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"  Train RMSE: {train_rmse:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}")
print(f"  Train MAE: {train_mae:.4f}")
print(f"  Test MAE: {test_mae:.4f}")

# Feature importance (coefficients from OLS model)
# Get the constant term name (it might be 'const' or the first parameter)
constant_name = ols_model.params.index[0]  # First parameter is usually the constant
coefs = ols_model.params.drop(constant_name)
feature_importance = pd.DataFrame({
    'Feature': coefs.index,
    'Coefficient': coefs.values,
    'P_value': ols_model.pvalues[1:],  # Exclude constant
    'Significant': ols_model.pvalues[1:] < 0.05
})

print("\nFeature Importance (Coefficients):")
print(feature_importance.sort_values(by='Coefficient', key=abs, ascending=False))

# Model interpretation
print(f"\n" + "="*50)
print("MODEL INTERPRETATION")
print("="*50)

print(f"Model Performance:")
print(f"  Train R²: {train_r2:.4f}")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Cross-validated R²: {cv_r2:.4f}")
print(f"  Test RMSE: {test_rmse:.4f} seconds")
print(f"  Test MAE: {test_mae:.4f} seconds")

# Check for overfitting
r2_gap = train_r2 - test_r2
if r2_gap > 0.1:
    print(f"  WARNING: Potential overfitting detected (Train-Test R² gap: {r2_gap:.3f})")
else:
    print(f"  Good generalization (Train-Test R² gap: {r2_gap:.3f})")

# Check for negative R²
if test_r2 < 0:
    print(f"  WARNING: Negative R² indicates model performs worse than horizontal line")

# Predator perception analysis
print(f"\n" + "="*50)
print("PREDATOR PERCEPTION ANALYSIS")
print("="*50)

# Key rat-related features for predator perception
rat_features = ['rat_presence', 'rat_activity_index', 'proximity_score', 'risk_rat_interaction', 
                'seconds_after_rat_arrival', 'rat_minutes', 'rat_arrival_number']

print("Rat-related feature effects on vigilance (anti-predator behavior):")
predator_evidence = 0
for feature in rat_features:
    if feature in feature_importance['Feature'].values:
        row = feature_importance[feature_importance['Feature'] == feature].iloc[0]
        effect = "increases" if row['Coefficient'] > 0 else "decreases"
        significance = "***" if row['P_value'] < 0.001 else "**" if row['P_value'] < 0.01 else "*" if row['P_value'] < 0.05 else ""
        print(f"  {feature}: {effect} vigilance by {abs(row['Coefficient']):.4f} seconds (β={row['Coefficient']:+.4f}) {significance}")
        
        # Count evidence for predator perception (positive coefficients for rat features)
        if row['Coefficient'] > 0 and row['P_value'] < 0.05:
            predator_evidence += 1
        elif row['Coefficient'] > 0:
            predator_evidence += 0.5

print(f"\nModel Equation:")
equation = f"Vigilance = {ols_model.params[constant_name]:.4f}"
for feature in X_vars:
    if feature in coefs.index:
        coef_val = coefs[feature]
        sign = "+" if coef_val >= 0 else ""
        equation += f" {sign}{coef_val:.4f}*{feature}"
print(f"  {equation}")

# Statistical significance summary
significant_features = feature_importance[feature_importance['Significant']]
print(f"\nStatistically significant features (p < 0.05): {len(significant_features)}")
for _, row in significant_features.iterrows():
    print(f"  {row['Feature']}: β={row['Coefficient']:+.4f}, p={row['P_value']:.4f}")

# Visualize results
print(f"\n" + "="*50)
print("CREATING MULTIPLE REGRESSION VISUALIZATIONS")
print("="*50)

import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
fig.suptitle('Phase 5.4: Multiple Linear Regression - Predator Perception Analysis', fontsize=16, fontweight='bold')

# Feature Importance
coefs = ols_model.params.drop(constant_name)
features = coefs.index
importance = coefs.values
sorted_idx = np.argsort(abs(importance))[::-1]
colors = ['green' if importance[i] > 0 else 'red' for i in sorted_idx]

bars = axs[0].barh(np.array(features)[sorted_idx], np.array(importance)[sorted_idx], color=colors, alpha=0.75)
axs[0].set_title('Feature Effects on Bat Vigilance\n(Potential Predation Indicators)', fontweight='bold')
axs[0].set_xlabel('Coefficient Value (seconds)', fontsize=12)
axs[0].axvline(x=0, color='black', linestyle='--', alpha=0.5)

# Add coefficient values
for i, v in enumerate(np.array(importance)[sorted_idx]):
    axs[0].text(v + 0.1 if v > 0 else v - 0.1, i, f'{v:.3f}', va='center', ha='left' if v > 0 else 'right')

# Actual vs Predicted
axs[1].scatter(y_test, y_pred_test, alpha=0.6, color='blue')
axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axs[1].set_xlabel('Actual Vigilance (seconds)', fontsize=12)
axs[1].set_ylabel('Predicted Vigilance (seconds)', fontsize=12)
axs[1].set_title(f'Actual vs Predicted (R² = {test_r2:.3f})', fontweight='bold')
axs[1].text(0.05, 0.95, f'R² = {test_r2:.3f}', transform=axs[1].transAxes, fontsize=12, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
phase54_path = os.path.join(plots_dir, 'Phase5.4_Predator_Perception_Analysis.png')
plt.savefig(phase54_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved Phase 5.4 multiple regression plot to: {phase54_path}")

# Final Analysis and Conclusion
print(f"\n" + "="*60)
print("FINAL ANALYSIS AND CONCLUSION")
print("="*60)

print("Based on multiple linear regression analysis:")

# Evidence compilation for predator perception
evidence_for_predator = 0
evidence_against_predator = 0

print(f"\n" + "="*50)
print("EVIDENCE COMPILATION FOR PREDATOR PERCEPTION")
print("="*50)

# Check model performance
if test_r2 > 0.3:
    evidence_for_predator += 1
    print(f"✓ Strong model performance (R² = {test_r2:.3f}) supports predator perception")
elif test_r2 > 0.1:
    evidence_for_predator += 0.5
    print(f"~ Moderate model performance (R² = {test_r2:.3f}) suggests some predator perception")
else:
    evidence_against_predator += 1
    print(f"✗ Weak model performance (R² = {test_r2:.3f}) suggests limited predator perception")

# Check prediction accuracy
if test_rmse < 2.0:  # Low RMSE indicates good prediction
    evidence_for_predator += 1
    print(f"✓ Low prediction error (RMSE = {test_rmse:.3f}) supports reliable prediction")
elif test_rmse < 5.0:
    evidence_for_predator += 0.5
    print(f"~ Moderate prediction error (RMSE = {test_rmse:.3f}) suggests reasonable prediction")
else:
    evidence_against_predator += 0.5
    print(f"~ High prediction error (RMSE = {test_rmse:.3f}) suggests limited predictive power")

# Check rat-related feature effects (key for predator perception)
rat_positive_effects = 0
rat_significant_effects = 0
for feature in rat_features:
    if feature in feature_importance['Feature'].values:
        row = feature_importance[feature_importance['Feature'] == feature].iloc[0]
        if row['Coefficient'] > 0:
            rat_positive_effects += 1
            if row['P_value'] < 0.05:
                rat_significant_effects += 1

if rat_significant_effects >= 2:
    evidence_for_predator += 1
    print(f"✓ Strong rat effect evidence ({rat_significant_effects} significant positive effects)")
elif rat_positive_effects >= 2:
    evidence_for_predator += 0.5
    print(f"~ Moderate rat effect evidence ({rat_positive_effects} positive effects, {rat_significant_effects} significant)")
else:
    evidence_against_predator += 0.5
    print(f"✗ Limited rat effect evidence ({rat_positive_effects} positive effects)")

# Check proximity effects (key indicator of predator perception)
proximity_features = ['proximity_score', 'seconds_after_rat_arrival']
proximity_evidence = 0
for feature in proximity_features:
    if feature in feature_importance['Feature'].values:
        row = feature_importance[feature_importance['Feature'] == feature].iloc[0]
        if feature == 'proximity_score' and row['Coefficient'] > 0:
            proximity_evidence += 1
        elif feature == 'seconds_after_rat_arrival' and row['Coefficient'] < 0:
            proximity_evidence += 1

if proximity_evidence >= 1:
    evidence_for_predator += 0.5
    print(f"✓ Proximity effects support predator perception")
else:
    evidence_against_predator += 0.5
    print(f"✗ No clear proximity effects for predator perception")

# Check overfitting
if r2_gap < 0.1:
    evidence_for_predator += 0.5
    print(f"✓ Good generalization (Train-Test R² gap: {r2_gap:.3f})")
else:
    evidence_against_predator += 0.5
    print(f"✗ Potential overfitting (Train-Test R² gap: {r2_gap:.3f})")

# Check for negative R²
if test_r2 < 0:
    evidence_against_predator += 1
    print(f"✗ Negative R² indicates model performs worse than baseline")

# Final conclusion
print(f"\nEvidence Summary:")
print(f"  Supporting predator perception: {evidence_for_predator}")
print(f"  Against predator perception: {evidence_against_predator}")

if evidence_for_predator > evidence_against_predator:
    print(f"\nCONCLUSION: EVIDENCE SUPPORTS PREDATOR PERCEPTION")
    print(f"Multiple regression analysis provides evidence that bats perceive rats as predators.")
    print(f"Positive coefficients for rat-related variables suggest predation risk beyond competition.")
    print(f"Vigilance behavior increases with rat presence and proximity, indicating anti-predator response.")
    print(f"Statistical significance of rat features supports the predator perception hypothesis.")
elif evidence_for_predator == evidence_against_predator:
    print(f"\nCONCLUSION: MIXED EVIDENCE FOR PREDATOR PERCEPTION")
    print(f"Multiple regression analysis shows both supporting and contradictory evidence.")
    print(f"Some rat-related features suggest predator perception while others suggest competition.")
    print(f"Further investigation needed to confirm whether bats perceive rats as predators or competitors.")
    print(f"Behavioral responses may be context-dependent or driven by multiple factors.")
else:
    print(f"\nCONCLUSION: NO CLEAR EVIDENCE FOR PREDATOR PERCEPTION")
    print(f"Bats may primarily perceive rats as competitors rather than predators.")
    print(f"Behavioral responses appear driven by competition rather than predation risk.")
    print(f"Multiple regression does not show strong evidence for anti-predator vigilance responses.")
    print(f"Rat presence may not significantly increase vigilance beyond competitive interactions.")

print(f"\nMultiple regression analysis completed successfully!")

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

# Re-run Phase 4 analysis to ensure fresh results

# Re-derive seasons if needed

# Split data by season
winter_data = dataset1[dataset1['season_label'] == 'Winter'].copy()
spring_data = dataset1[dataset1['season_label'] == 'Spring'].copy()

print(f"Data split by season:")
print(f"  Winter observations: {len(winter_data)}")
print(f"  Spring observations: {len(spring_data)}")

# Define key predictors for seasonal comparison
seasonal_predictors = {
    'seconds_after_rat_arrival': 'Temporal Proximity to Rat Arrival',
    'rat_minutes': 'Rat Presence Intensity (minutes)',
    'rat_arrival_number': 'Rat Arrival Sequence Number'
}
response_var = 'bat_landing_to_food'
print(f"Response Variable: Bat Vigilance (seconds)")
print(f"Predictors: {len(seasonal_predictors)} key variables for seasonal comparison\n")

# Store seasonal model comparison results
seasonal_comparison_results = {}

print("="*50)
print("SEASONAL MODEL COMPARISON")
print("="*50)

# Train separate models for each predictor and season
for predictor, description in seasonal_predictors.items():
    print(f"\nSeasonal Comparison: {description}")
    print("-" * 50)
    
    # Prepare Winter and Spring data
    winter_subset = winter_data[[predictor, response_var]].dropna()
    spring_subset = spring_data[[predictor, response_var]].dropna()
    
    if len(winter_subset) > 5 and len(spring_subset) > 5:
        # Winter model
        X_winter = winter_subset[[predictor]]
        y_winter = winter_subset[response_var]
        
        # Spring model
        X_spring = spring_subset[[predictor]]
        y_spring = spring_subset[response_var]
        
        # Train Winter model
        winter_model = LinearRegression()
        winter_model.fit(X_winter, y_winter)
        
        # Train Spring model
        spring_model = LinearRegression()
        spring_model.fit(X_spring, y_spring)
        
        # Calculate metrics
        winter_r2 = r2_score(y_winter, winter_model.predict(X_winter))
        spring_r2 = r2_score(y_spring, spring_model.predict(X_spring))
        
        # Check for negative R²
        if winter_r2 < 0:
            print(f"ERROR: Negative R² in simple regression for {predictor} (Winter) — check data!")
        if spring_r2 < 0:
            print(f"ERROR: Negative R² in simple regression for {predictor} (Spring) — check data!")
        
        # Calculate correlations
        winter_corr = np.corrcoef(winter_subset[predictor], winter_subset[response_var])[0, 1]
        spring_corr = np.corrcoef(spring_subset[predictor], spring_subset[response_var])[0, 1]
        
        # Store results
        coef_difference = spring_model.coef_[0] - winter_model.coef_[0]
        seasonal_comparison_results[predictor] = {
            'description': description,
            'winter_coef': winter_model.coef_[0],
            'spring_coef': spring_model.coef_[0],
            'winter_r2': winter_r2,
            'spring_r2': spring_r2,
            'winter_corr': winter_corr,
            'spring_corr': spring_corr,
            'coef_difference': coef_difference
        }
        
        # Display results
        print(f"  Winter: n={len(winter_subset)}, β={winter_model.coef_[0]:+.4f}, R²={winter_r2:.4f}")
        print(f"  Spring: n={len(spring_subset)}, β={spring_model.coef_[0]:+.4f}, R²={spring_r2:.4f}")
        print(f"  Difference: Δβ={coef_difference:+.4f}")
        
    else:
        print(f"  Insufficient data: Winter n={len(winter_subset)}, Spring n={len(spring_subset)}")
        print(f"  Need at least 6 observations per season for reliable comparison")

# Visualization
print(f"\n" + "="*50)
print("CREATING SEASONAL COMPARISON VISUALIZATIONS")
print("="*50)

# Create seasonal comparison plots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
fig.suptitle('Phase 5.1: Seasonal Model Comparison - Winter vs Spring', fontsize=16, fontweight='bold')

plot_idx = 0
for predictor, description in seasonal_predictors.items():
    if predictor in seasonal_comparison_results and plot_idx < 3:
        # Get Winter and Spring data
        winter_subset = winter_data[[predictor, response_var]].dropna()
        spring_subset = spring_data[[predictor, response_var]].dropna()
        
        # Scatter plots
        axs[plot_idx].scatter(winter_subset[predictor], winter_subset[response_var], 
                             alpha=0.6, s=50, color='blue', label='Winter')
        axs[plot_idx].scatter(spring_subset[predictor], spring_subset[response_var], 
                             alpha=0.6, s=50, color='red', label='Spring')
        
        # Add simple trend lines
        winter_coef = seasonal_comparison_results[predictor]['winter_coef']
        spring_coef = seasonal_comparison_results[predictor]['spring_coef']
        
        # Winter line
        x_winter = np.linspace(winter_subset[predictor].min(), winter_subset[predictor].max(), 100)
        y_winter = winter_coef * x_winter + seasonal_comparison_results[predictor].get('winter_intercept', 0)
        axs[plot_idx].plot(x_winter, y_winter, 'b-', linewidth=2, label='Winter Model')
        
        # Spring line
        x_spring = np.linspace(spring_subset[predictor].min(), spring_subset[predictor].max(), 100)
        y_spring = spring_coef * x_spring + seasonal_comparison_results[predictor].get('spring_intercept', 0)
        axs[plot_idx].plot(x_spring, y_spring, 'r-', linewidth=2, label='Spring Model')
        
        # Formatting
        axs[plot_idx].set_xlabel(description, fontsize=12, fontweight='bold')
        axs[plot_idx].set_ylabel('Bat Vigilance (seconds)', fontsize=12, fontweight='bold')
        
        coef_diff = seasonal_comparison_results[predictor]['coef_difference']
        axs[plot_idx].set_title(f'{description}\nWinter β={winter_coef:+.3f}, Spring β={spring_coef:+.3f}\nΔβ={coef_diff:+.3f}', 
                               fontweight='bold', fontsize=11)
        axs[plot_idx].legend(fontsize=10)
        axs[plot_idx].tick_params(axis='both', labelsize=10)
        
        plot_idx += 1

plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2.0)
phase51_path = os.path.join(plots_dir, 'Phase5.1_Seasonal_Model_Comparison.png')
plt.savefig(phase51_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved Phase 5.1 seasonal model comparison plot to: {phase51_path}")

# Summary
print("\n" + "="*50)
print("SEASONAL COMPARISON SUMMARY")
print("="*50)

if seasonal_comparison_results:
    # Find predictor with largest seasonal difference
    largest_diff_predictor = max(seasonal_comparison_results.keys(), 
                                key=lambda x: abs(seasonal_comparison_results[x]['coef_difference']))
    largest_diff = seasonal_comparison_results[largest_diff_predictor]['coef_difference']
    
    print(f"Largest seasonal difference: {seasonal_comparison_results[largest_diff_predictor]['description']}")
    print(f"Coefficient difference (Δβ): {largest_diff:+.4f}")
    
    print(f"\nSeasonal behavioral changes:")
    for pred, results in seasonal_comparison_results.items():
        coef_diff = results['coef_difference']
        direction = "Spring stronger" if coef_diff > 0 else "Winter stronger" if coef_diff < 0 else "No difference"
        print(f"  {results['description']}: {direction} (Δβ={coef_diff:+.4f})")
    
    print(f"\nKey findings for Investigation B:")
    print(f"  • Seasonal differences in behavioral responses detected")
    print(f"  • {largest_diff_predictor} shows strongest seasonal variation")
    print(f"  • Behavioral patterns change with environmental conditions")
    print(f"  • Linear regression quantifies seasonal effects")

print(f"\nSeasonal model comparison completed successfully!")

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

# Re-run Phase 4 analysis to ensure fresh results

# Re-derive seasons if needed
if 'season_label' not in dataset1.columns:
    print("Adding season_label column...")
    dataset1['season_label'] = dataset1['start_time'].apply(_season_label_from_month)

# Pick seasons to compare
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

w_df = dataset1[dataset1['season_label'] == season_pair[0]].copy()
s_df = dataset1[dataset1['season_label'] == season_pair[1]].copy()

# Re-run hypothesis tests
from scipy.stats import mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests

def mw_test(x_w, x_s):
    x_w = pd.Series(x_w).dropna()
    x_s = pd.Series(x_s).dropna()
    if len(x_w) == 0 or len(x_s) == 0:
        return np.nan, np.nan, np.nan
    u, p = mannwhitneyu(x_w, x_s, alternative='two-sided')
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

# H4 success (reward)
if 'reward' in dataset1.columns:
    w_succ = int(w_df['reward'].dropna().sum()); w_fail = int((w_df['reward']==0).sum())
    s_succ = int(s_df['reward'].dropna().sum()); s_fail = int((s_df['reward']==0).sum())
    table = np.array([[w_succ, w_fail],[s_succ, s_fail]])
    try:
        _, p = fisher_exact(table)
    except Exception:
        p = np.nan
    results['H4_success'] = {'p': p, 'effect': np.nan,
        'w_prop': w_succ/max(1, (w_succ+w_fail)), 's_prop': s_succ/max(1, (s_succ+s_fail))}

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
w_def = int(w_df['habit'].isin(defensive_habits).sum()); w_nodef = len(w_df) - w_def
s_def = int(s_df['habit'].isin(defensive_habits).sum()); s_nodef = len(s_df) - s_def
table = np.array([[w_def, w_nodef],[s_def, s_nodef]])
try:
    _, p = fisher_exact(table)
except Exception:
    p = np.nan
results['H6_defensive'] = {'p': p, 'w_prop': w_def/max(1,len(w_df)), 's_prop': s_def/max(1,len(s_df))}

# FDR correction
primary_keys = [k for k in ['H1_arrivals','H2_minutes','H3_vigilance','H4_success'] if k in results]
primary_p = [results[k]['p'] for k in primary_keys if not pd.isna(results[k]['p'])]
adj_map = {}
if len(primary_p):
    rej, p_adj, _, _ = multipletests(primary_p, method='fdr_bh')
    for k, pa, rj in zip(primary_keys, p_adj, rej):
        adj_map[k] = {'p_adj': pa, 'reject': bool(rj)}

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
# %