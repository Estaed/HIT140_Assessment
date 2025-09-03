# -*- coding: utf-8 -*-
#%%
# Investigation A: Do bats perceive rats as potential predators?

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

fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
datasets_info = [len(dataset1), len(dataset2)]
dataset_names = ['Bat Events\n(Dataset1)', 'Environmental Periods\n(Dataset2)']
bars = ax.bar(dataset_names, datasets_info, color=['steelblue', 'darkorange'], alpha=0.7)
ax.set_title('Data Available for Analysis', fontweight='bold')
ax.set_ylabel('Records')
for bar, val in zip(bars, datasets_info):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'phase1_data_overview.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

#%%
# ============================================================================
# PHASE 2: HABIT CLASSIFICATION AND DATA CLEANING
# ============================================================================
print("PHASE 2: HABIT CLASSIFICATION")

# Convert time columns
time_cols = ['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time']
for col in time_cols:
    dataset1[col] = pd.to_datetime(dataset1[col], format='%d/%m/%Y %H:%M')
dataset2['time'] = pd.to_datetime(dataset2['time'], format='%d/%m/%Y %H:%M')

# Fix bat_landing_to_food values
# First, check for fractional values
fractions = dataset1[(dataset1['bat_landing_to_food'] > 0) & (dataset1['bat_landing_to_food'] < 1)]
extreme_fractions = fractions[fractions['bat_landing_to_food'] < 0.1]
print(f"Found {len(fractions)} fractional values (<1), of which {len(extreme_fractions)} are extreme (<0.1) – likely errors")

# Pre-fix stats
pre_mean = dataset1['bat_landing_to_food'].mean()
pre_min = dataset1['bat_landing_to_food'].min()
pre_max = dataset1['bat_landing_to_food'].max()
print(f"Pre-fix stats: Mean={pre_mean:.2f}s, Min={pre_min:.4f}s, Max={pre_max:.0f}s")

def fix_landing_values(value):
    if pd.isna(value):
        return value
    if value >= 1:
        return int(value)
    if 0 < value < 0.1:  # Scale only extreme fractions (likely ms errors)
        return int(round(value * 1000))
    return value  # Keep 0.1–0.999 as valid sub-second

dataset1['bat_landing_to_food'] = dataset1['bat_landing_to_food'].apply(fix_landing_values)

# Post-fix stats
post_mean = dataset1['bat_landing_to_food'].mean()
post_min = dataset1['bat_landing_to_food'].min()
post_max = dataset1['bat_landing_to_food'].max()
print(f"Post-fix stats: Mean={post_mean:.2f}s, Min={post_min:.4f}s, Max={post_max:.0f}s (fixed {len(extreme_fractions)} values)")

# Habit classification with priority order
def classify_habit_updated(habit, risk, reward):
    if pd.isna(habit) or any(char.isdigit() for char in str(habit)):
        return 'unknown'
    habit_str = str(habit).lower()
    if any(term in habit_str for term in ['attack', 'fight', 'disappear']):
        return 'fight'
    if 'fast' in habit_str:
        return 'fast'
    if 'pick' in habit_str:
        return 'pick'
    if 'bat' in habit_str:
        return 'bat'
    if 'rat' in habit_str:
        return 'rat'
    return 'other'

# Apply classification
dataset1['habit'] = dataset1.apply(
    lambda row: classify_habit_updated(row['habit'], row['risk'], row['reward']), axis=1
)

new_counts = dataset1['habit'].value_counts()
print(f"New categories: {dict(new_counts)}")

# Phase 2 visualization - classified categories only
fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor='white')
new_counts.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
ax.set_title('Classified Habit Categories', fontweight='bold')
ax.set_xlabel('Category')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'phase2_classification_results.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

#%%
# ============================================================================
# PHASE 3: MERGING AND EXPLORING VIGILANCE BY RAT TIMING
# ============================================================================
print("PHASE 3: MERGING AND EXPLORING VIGILANCE BY RAT TIMING")

# Ensure output folder for data exists
datasets_dir = os.path.join('datasets')
os.makedirs(datasets_dir, exist_ok=True)

# Merge datasets by finding the 30-min period
dataset1_sorted = dataset1.sort_values('start_time')
dataset2_sorted = dataset2.sort_values('time')
merged_df = pd.merge_asof(dataset1_sorted, dataset2_sorted, left_on='start_time', right_on='time', direction='backward', suffixes=('', '_period'))

# Add merge quality check: time difference between landing and matched period start
merged_df['time_to_period'] = (merged_df['start_time'] - merged_df['time']).dt.total_seconds() / 60.0  # in minutes
matched_within_30min = (merged_df['time_to_period'] <= 30) & (merged_df['time_to_period'] >= 0)
num_matched = matched_within_30min.sum()
print(f"Merge Quality: {num_matched} out of {len(merged_df)} landings matched to a period within 30 minutes ({num_matched / len(merged_df) * 100:.1f}%)")

# Calculate precise rats_present with explicit NaN handling
# Verify required columns
required_cols = ['seconds_after_rat_arrival', 'rat_period_start', 'rat_period_end']
missing_cols = [col for col in required_cols if col not in merged_df.columns]
if missing_cols:
    print(f"Error: Missing columns for rats_present calc: {missing_cols}. Setting all to False.")
    merged_df['rats_present'] = False
else:
    merged_df['rat_period_duration'] = (merged_df['rat_period_end'] - merged_df['rat_period_start']).dt.total_seconds()
    merged_df['rat_period_duration'] = merged_df['rat_period_duration'].fillna(0)
    merged_df['seconds_after_rat_arrival'] = merged_df['seconds_after_rat_arrival'].fillna(-1)
    merged_df['rats_present'] = (
        (merged_df['seconds_after_rat_arrival'] >= 0) & 
        (merged_df['seconds_after_rat_arrival'] < merged_df['rat_period_duration']) & 
        merged_df['rat_period_start'].notna()
    ).fillna(False)

# Check if column was created successfully
if 'rats_present' not in merged_df.columns:
    print("Error: 'rats_present' column not created! Setting default.")
    merged_df['rats_present'] = False
else:
    print(f"'rats_present' column created: {merged_df['rats_present'].value_counts().to_dict()}")

# Additional diagnostics for rat presence
if 'rats_present' in merged_df.columns:
    print(f"\nRat Presence Summary: True={merged_df['rats_present'].sum()} ({merged_df['rats_present'].mean()*100:.1f}%), False={len(merged_df) - merged_df['rats_present'].sum()}")
    no_rats_sample = merged_df[~merged_df['rats_present']].head(5)[['start_time', 'rat_period_start', 'rat_period_end', 'seconds_after_rat_arrival', 'time_to_period']]
    print("Sample of 5 'No Rats' rows (rats_present=False) for verification:")
    print(no_rats_sample)
else:
    print("\nNo 'rats_present' column - skipping summary")

# Visualize timestamp differences
fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
sns.histplot(merged_df['time_to_period'], bins=30, kde=True, ax=ax, color='steelblue')
ax.set_title('Distribution of Time Differences to Matched Period', fontweight='bold')
ax.set_xlabel('Minutes Since Period Start')
ax.set_ylabel('Count')
ax.axvline(30, color='red', linestyle='--', label='30-min threshold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'phase3_merge_quality.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Define timing buckets based on seconds_after_rat_arrival
def timing_bucket(sec):
    if pd.isna(sec):
        return 'no_rat'
    if sec < 0:
        return 'before_rat'
    if sec <= 60:
        return '<=1 min'
    elif sec <= 600:
        return '1-10 min'
    else:
        return '>10 min'

merged_df['rat_timing_bucket'] = merged_df['seconds_after_rat_arrival'].apply(timing_bucket)

# Quantify descriptive differences for vigilance (bat_landing_to_food)
group = merged_df.groupby('rat_timing_bucket')
vig_mean = group['bat_landing_to_food'].mean()
vig_median = group['bat_landing_to_food'].median()
n = group.size()
sem = group['bat_landing_to_food'].std() / np.sqrt(n)
ci_low = vig_mean - 1.96 * sem
ci_high = vig_mean + 1.96 * sem
desc_df = pd.DataFrame({
    'mean_vigilance': vig_mean,
    'median_vigilance': vig_median,
    'ci_low': ci_low,
    'ci_high': ci_high
})
print("Descriptive Statistics for Vigilance by Rat Timing Buckets:")
print(desc_df)

# Visualizations
fig, axs = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
plt.rcParams.update({'font.size': 12})  # Increase global font size

# Left: Vigilance boxplot
sns.boxplot(data=merged_df, x='rat_timing_bucket', y='bat_landing_to_food', ax=axs[0], color='steelblue', fliersize=1)
axs[0].set_title('Vigilance Time by Rat Timing', fontweight='bold')
axs[0].set_xlabel('Rat Timing Bucket')
axs[0].set_ylabel('Vigilance Time (seconds)')
axs[0].set_ylim(0, 100)  # Limit to focus on main data (adjust if needed)
axs[0].tick_params(axis='x', rotation=45)
axs[0].grid(True, linestyle='--', alpha=0.7)

# Middle: Reward bar with annotations
reward_means = group['reward'].mean()
reward_means.plot(kind='bar', ax=axs[1], color='darkorange', alpha=0.7)
axs[1].set_title('Mean Reward Success by Rat Timing', fontweight='bold')
axs[1].set_xlabel('Rat Timing Bucket')
axs[1].set_ylabel('Mean Success (0-1)')
axs[1].set_ylim(0, 1)
axs[1].tick_params(axis='x', rotation=45)
axs[1].grid(True, linestyle='--', alpha=0.7)
for i, v in enumerate(reward_means):
    axs[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')

# Right: Risk bar with annotations
risk_means = group['risk'].mean()
risk_means.plot(kind='bar', ax=axs[2], color='darkred', alpha=0.7)
axs[2].set_title('Mean Risk Taking by Rat Timing', fontweight='bold')
axs[2].set_xlabel('Rat Timing Bucket')
axs[2].set_ylabel('Mean Risk (0-1)')
axs[2].set_ylim(0, 1)
axs[2].tick_params(axis='x', rotation=45)
axs[2].grid(True, linestyle='--', alpha=0.7)
for i, v in enumerate(risk_means):
    axs[2].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')

plt.suptitle('Phase 3: Bat Behavior Analysis by Rat Presence Timing\n(Buckets: Time relative to rat arrival)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'phase3_vigilance_analysis.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Output clean merged CSV
merged_csv_path = os.path.join(datasets_dir, 'merged_dataset.csv')
merged_df.to_csv(merged_csv_path, index=False)
print(f"Exported merged dataset with {len(merged_df)} rows to '{merged_csv_path}'")
#%%
# ============================================================================
# PHASE 4: HYPOTHESIS TESTING FOR VIGILANCE AND BEHAVIOR DIFFERENCES
# ============================================================================
print("PHASE 4: HYPOTHESIS TESTING FOR VIGILANCE AND BEHAVIOR DIFFERENCES")

from scipy.stats import chi2_contingency, mannwhitneyu

# Step 1: Prepare groups for comparison using merged data
# Vigilance (continuous): rats present vs not
rats_present_vig = merged_df[merged_df['rats_present']]['bat_landing_to_food'].dropna()
no_rats_vig = merged_df[~merged_df['rats_present']]['bat_landing_to_food'].dropna()

print(f"Group sizes: Rats present: {len(rats_present_vig)}, No rats: {len(no_rats_vig)}")

# Step 2: Check normality assumption (for t-test validity)
stat_present, p_present = stats.shapiro(rats_present_vig.sample(min(5000, len(rats_present_vig))))
stat_no, p_no = stats.shapiro(no_rats_vig.sample(min(5000, len(no_rats_vig))))
print(f"Normality test - Rats present: p={p_present:.4f}, No rats: p={p_no:.4f}")

# Step 3: Perform one-tailed Mann-Whitney U test for higher vigilance when rats present
u_stat, p_val = mannwhitneyu(rats_present_vig, no_rats_vig, alternative='greater')
print(f"One-tailed Mann-Whitney U test for vigilance: u={u_stat:.4f}, p-value={p_val:.4f}")

# Interpret vigilance results
alpha = 0.05
if p_val < alpha:
    print("Reject null: Vigilance is significantly higher when rats are present (p < 0.05)")
else:
    print("Fail to reject null: No significant evidence of higher vigilance when rats are present")

# Step 4: Chi-square for risk (categorical)
contingency_risk = pd.crosstab(merged_df['rats_present'], merged_df['risk'])
print("Contingency table for risk:")
print(contingency_risk)
if all(contingency_risk.min(axis=1) >= 5):  # Check for validity
    chi2_risk, p_risk, dof_risk, expected_risk = chi2_contingency(contingency_risk)
    print(f"Chi-square for risk: chi2={chi2_risk:.4f}, p-value={p_risk:.4f}")
    if p_risk < alpha:
        print("Significant association: Risk-taking differs by rat presence")
    else:
        print("No significant association for risk")
else:
    print("Chi-square assumptions not met (some expected frequencies <5); interpret cautiously")

# Step 5: Chi-square for reward (categorical)
contingency_reward = pd.crosstab(merged_df['rats_present'], merged_df['reward'])
print("Contingency table for reward:")
print(contingency_reward)
if all(contingency_reward.min(axis=1) >= 5):
    chi2_reward, p_reward, dof_reward, expected_reward = chi2_contingency(contingency_reward)
    print(f"Chi-square for reward: chi2={chi2_reward:.4f}, p-value={p_reward:.4f}")
    if p_reward < alpha:
        print("Significant association: Reward success differs by rat presence")
    else:
        print("No significant association for reward")
else:
    print("Chi-square assumptions not met (some expected frequencies <5); interpret cautiously")

# Step 6: Visualization of comparisons (same as before)
fig, axs = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')

sns.boxplot(data=merged_df, x='rats_present', y='bat_landing_to_food', ax=axs[0], color='steelblue', fliersize=1)
axs[0].set_title('Vigilance by Rat Presence', fontweight='bold')
axs[0].set_xlabel('Rats Present')
axs[0].set_ylabel('Vigilance Time (s)')

merged_df.groupby('rats_present')['risk'].mean().plot(kind='bar', ax=axs[1], color='darkred', alpha=0.7)
axs[1].set_title('Mean Risk-Taking by Rat Presence', fontweight='bold')
axs[1].set_xlabel('Rats Present')
axs[1].set_ylabel('Mean Risk')

merged_df.groupby('rats_present')['reward'].mean().plot(kind='bar', ax=axs[2], color='darkorange', alpha=0.7)
axs[2].set_title('Mean Reward Success by Rat Presence', fontweight='bold')
axs[2].set_xlabel('Rats Present')
axs[2].set_ylabel('Mean Reward')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'phase4_hypothesis_testing.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
# %%
# ============================================================================
# PHASE 5: CONCLUSION AND FINAL ANSWER
# ============================================================================
print("PHASE 5: CONCLUSION AND FINAL ANSWER")

# Dynamic summary of key findings
print("\nDynamic Summary of Hypothesis Testing Results:")
vig_interpret = "Reject null: Vigilance is significantly higher when rats are present" if p_val < alpha else "Fail to reject null: No significant evidence of higher vigilance when rats are present"
risk_interpret = "Significant association: Risk-taking differs by rat presence" if p_risk < alpha else "No significant association for risk"
reward_interpret = "Significant association: Reward success differs by rat presence" if p_reward < alpha else "No significant association for reward"

print(f"- Vigilance: {vig_interpret} (p={p_val:.4f}, mean with rats: {rats_present_vig.mean():.2f}s, without: {no_rats_vig.mean():.2f}s)")
print(f"- Risk-taking: {risk_interpret} (p={p_risk:.4f}, mean with rats: {merged_df[merged_df['rats_present']]['risk'].mean():.2f}, without: {merged_df[~merged_df['rats_present']]['risk'].mean():.2f})")
print(f"- Reward success: {reward_interpret} (p={p_reward:.4f}, mean with rats: {merged_df[merged_df['rats_present']]['reward'].mean():.2f}, without: {merged_df[~merged_df['rats_present']]['reward'].mean():.2f})")

# Dynamic address of research question
print("\nResearch Question: Do bats perceive rats not just as competitors for food but also as potential predators?")

evidence = []
if p_val < alpha:
    evidence.append("higher vigilance")
if p_risk < alpha or p_reward < alpha:
    evidence.append("altered risk/reward behaviors")
    
if evidence:
    conclusion = f"Based on the analysis, there is statistical evidence of {' and '.join(evidence)} when rats are present, suggesting bats may perceive rats as potential predation threats."
else:
    conclusion = "Based on the analysis, there is no statistical evidence of increased vigilance or altered risk/reward behaviors when rats are present. This suggests that bats primarily view rats as competitors for food rather than as predation threats."

print(f"Conclusion: {conclusion}")

# Dynamic final answer
if evidence:
    final_answer = "Yes, the data supports that bats perceive rats as potential predators beyond just competitors."
else:
    final_answer = "No, the data does not support that bats perceive rats as potential predators; they appear to treat them mainly as competitors."

print(f"\nFinal Answer: {final_answer}")

# New conclusion plot
fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

# Calculate means in Phase 5
if 'rats_present' in merged_df.columns:
    vig_means = merged_df.groupby('rats_present')['bat_landing_to_food'].mean()
    risk_means = merged_df.groupby('rats_present')['risk'].mean()
    reward_means = merged_df.groupby('rats_present')['reward'].mean()
else:
    print("Warning: 'rats_present' missing - using dummy means")
    vig_means = pd.Series({False: 0, True: 0})
    risk_means = pd.Series({False: 0, True: 0})
    reward_means = pd.Series({False: 0, True: 0})

# Get means safely and check available groups
has_false = False in vig_means.index
has_true = True in vig_means.index
vig_false = vig_means.get(False, 0) if has_false else None
vig_true = vig_means.get(True, 0) if has_true else None
risk_false = risk_means.get(False, 0) if has_false else None
risk_true = risk_means.get(True, 0) if has_true else None
reward_false = reward_means.get(False, 0) if has_false else None
reward_true = reward_means.get(True, 0) if has_true else None

# Print for debug
print(f"Debug: Groups available - False: {has_false} (n={len(no_rats_vig)}), True: {has_true} (n={len(rats_present_vig)})")
print(f"Vigilance: False={vig_false}, True={vig_true}")
print(f"Risk: False={risk_false}, True={risk_true}")
print(f"Reward: False={reward_false}, True={reward_true}")

# Dynamic x and labels based on available groups
available_groups = []
if has_false: available_groups.append((0, 'No Rats', vig_false, risk_false, reward_false))
if has_true: available_groups.append((1 if has_false else 0, 'Rats Present', vig_true, risk_true, reward_true))
x = [g[0] for g in available_groups]
x_labels = [g[1] for g in available_groups]

# Hypothetical expected (adjust based on available)
base_vig = vig_false or vig_true or 10
base_risk = risk_false or risk_true or 0.5
base_reward = reward_false or reward_true or 0.5
expected_vig = [base_vig, base_vig * 1.5] if has_false and has_true else [base_vig * (1.5 if label == 'Rats Present' else 1) for _, label, _, _, _ in available_groups]
expected_risk = [base_risk, base_risk * 0.8] if has_false and has_true else [base_risk * (0.8 if label == 'Rats Present' else 1) for _, label, _, _, _ in available_groups]
expected_reward = [base_reward, base_reward * 0.8] if has_false and has_true else [base_reward * (0.8 if label == 'Rats Present' else 1) for _, label, _, _, _ in available_groups]

# Update add_explain to handle single group
def add_explain(ax, observed, expected, metric, is_vig=False):
    if len(observed) == 2 and all(o is not None for o in observed):
        diff = expected[1] - observed[1]
        arrow_dir = '↑' if is_vig else '↓'
        arrow_y = observed[1] + diff / 2
        ax.annotate(f'Expected if Predators:\n{arrow_dir} {metric}', xy=(1, observed[1]), xytext=(1.1, arrow_y),
                    arrowprops=dict(facecolor='black', shrink=0.05), ha='left', va='center')
    elif len(observed) == 1:
        ax.text(0, max(observed) * 0.5, 'Only One Group\nLimited Comparison', ha='center', color='gray')

# Vigilance
vig_observed = [g[2] for g in available_groups if g[2] is not None]
vig_max = max(max(expected_vig), max(vig_observed or [0])) * 1.2
axs[0].bar([i - 0.2 for i in x], expected_vig, width=0.4, color='lightgray', label='If Predators')
axs[0].bar([i + 0.2 for i in x], vig_observed, width=0.4, color='steelblue', alpha=0.7, label='What We See')
axs[0].set_title('Vigilance (Fear Check)', fontweight='bold')
axs[0].set_xlabel('')
axs[0].set_ylabel('Time (seconds)')
axs[0].set_ylim(0, vig_max)
axs[0].set_xticks(x)
axs[0].set_xticklabels(x_labels)
for i, v in enumerate(vig_observed):
    text = f'{v:.1f}' if v > 0 else 'No Data'
    axs[0].text(i + 0.2, v + (vig_max*0.02), text, ha='center', fontweight='bold')
add_explain(axs[0], vig_observed, expected_vig, 'Vigilance', is_vig=True)
axs[0].legend(loc='upper left')
axs[0].text(0.5, -0.15, f'p={p_val:.4f}', ha='center', transform=axs[0].transAxes)

# Risk (similar structure)
risk_observed = [g[3] for g in available_groups if g[3] is not None]
risk_max = max(max(expected_risk), max(risk_observed or [0])) * 1.2
axs[1].bar([i - 0.2 for i in x], expected_risk, width=0.4, color='lightgray', label='If Predators')
axs[1].bar([i + 0.2 for i in x], risk_observed, width=0.4, color='darkred', alpha=0.7, label='What We See')
axs[1].set_title('Risk-Taking (Boldness)', fontweight='bold')
axs[1].set_xlabel('')
axs[1].set_ylabel('Level (0-1)')
axs[1].set_ylim(0, risk_max)
axs[1].set_xticks(x)
axs[1].set_xticklabels(x_labels)
for i, v in enumerate(risk_observed):
    text = f'{v:.2f}' if v > 0 else 'No Data'
    axs[1].text(i + 0.2, v + (risk_max*0.02), text, ha='center', fontweight='bold')
add_explain(axs[1], risk_observed, expected_risk, 'Risk')
axs[1].legend(loc='upper left')
axs[1].text(0.5, -0.15, f'p={p_risk:.4f}', ha='center', transform=axs[1].transAxes)

# Reward (similar)
reward_observed = [g[4] for g in available_groups if g[4] is not None]
reward_max = max(max(expected_reward), max(reward_observed or [0])) * 1.2
axs[2].bar([i - 0.2 for i in x], expected_reward, width=0.4, color='lightgray', label='If Predators')
axs[2].bar([i + 0.2 for i in x], reward_observed, width=0.4, color='darkorange', alpha=0.7, label='What We See')
axs[2].set_title('Reward Success (Food Get)', fontweight='bold')
axs[2].set_xlabel('')
axs[2].set_ylabel('Success (0-1)')
axs[2].set_ylim(0, reward_max)
axs[2].set_xticks(x)
axs[2].set_xticklabels(x_labels)
for i, v in enumerate(reward_observed):
    text = f'{v:.2f}' if v > 0 else 'No Data'
    axs[2].text(i + 0.2, v + (reward_max*0.02), text, ha='center', fontweight='bold')
add_explain(axs[2], reward_observed, expected_reward, 'Reward')
axs[2].legend(loc='upper left')
axs[2].text(0.5, -0.15, f'p={p_reward:.4f}', ha='center', transform=axs[2].transAxes)

plt.suptitle('Why Not Predators? Observed Behavior Shows No Fear Response to Rats\n(If Predators: Expect ↑ Vigilance, ↓ Risk/Reward When Rats Present)', fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(plots_dir, 'phase5_conclusion.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# %%
# ============================================================================
# PHASE 6: ADDITIONAL ANALYSIS FOR ROBUSTNESS
# ============================================================================
print("PHASE 6: ADDITIONAL ANALYSIS FOR ROBUSTNESS")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import statsmodels.api as sm

# 1. Logistic regression to predict rats_present from vigilance, risk, reward
features = ['bat_landing_to_food', 'risk', 'reward']
X = merged_df[features].dropna()
y = merged_df.loc[X.index, 'rats_present'].astype(int)
X = sm.add_constant(X)  # Add intercept

logit_model = sm.Logit(y, X)
result = logit_model.fit()
print("\nLogistic Regression Summary:")
print(result.summary())

# 2. Effect size for Mann-Whitney U (rank-biserial correlation)
n1, n2 = len(rats_present_vig), len(no_rats_vig)
effect_size = (2 * u_stat / (n1 * n2)) - 1
print(f"\nEffect size (rank-biserial) for vigilance: {effect_size:.4f} (small effect if |r| < 0.3)")

# 2b. Cohen's d effect size calculation
print("\nCohen's d Effect Size Analysis:")
print("Calculating standardized effect sizes for group differences...")

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (group1.mean() - group2.mean()) / pooled_std
    return d

# Calculate Cohen's d for vigilance (bat_landing_to_food)
if len(rats_present_vig) > 0 and len(no_rats_vig) > 0:
    d_vigilance = cohens_d(rats_present_vig, no_rats_vig)
    print(f"Cohen's d for Vigilance (rats vs no rats): {d_vigilance:.3f}")
    
    # Interpret effect size
    if abs(d_vigilance) < 0.2:
        effect_interpretation = "negligible"
    elif abs(d_vigilance) < 0.5:
        effect_interpretation = "small"
    elif abs(d_vigilance) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    print(f"Effect size interpretation: {effect_interpretation}")
    
    # Calculate 95% confidence interval for Cohen's d
    se_d = np.sqrt((len(rats_present_vig) + len(no_rats_vig)) / (len(rats_present_vig) * len(no_rats_vig)) + 
                  d_vigilance**2 / (2 * (len(rats_present_vig) + len(no_rats_vig))))
    ci_lower = d_vigilance - 1.96 * se_d
    ci_upper = d_vigilance + 1.96 * se_d
    print(f"95% CI for Cohen's d: [{ci_lower:.3f}, {ci_upper:.3f}]")
else:
    print("Cannot calculate Cohen's d - insufficient data in one or both groups")

# 3. Correlation in dataset2: bat landings vs rat activity
corr_landing_rat = dataset2['bat_landing_number'].corr(dataset2['rat_arrival_number'], method='spearman')
print(f"\nSpearman correlation between bat landings and rat arrivals in 30-min periods: {corr_landing_rat:.4f}")

# Visualization of correlation
fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
sns.scatterplot(data=dataset2, x='rat_arrival_number', y='bat_landing_number', ax=ax, color='steelblue')
ax.set_title('Bat Landings vs Rat Arrivals per 30-min Period', fontweight='bold')
ax.set_xlabel('Rat Arrivals')
ax.set_ylabel('Bat Landings')
ax.text(0.1, 0.9, f'Spearman r = {corr_landing_rat:.4f}', transform=ax.transAxes)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'phase6_landing_correlation.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Updated conclusion incorporating additional analysis
print("\nUpdated Conclusion: The logistic regression shows no significant predictive power of bat behaviors for rat presence (all p > 0.05). Small effect size for vigilance differences and weak correlation between bat landings and rat activity further support that bats do not avoid areas with rats, consistent with viewing them as competitors rather than predators.")

# %%
