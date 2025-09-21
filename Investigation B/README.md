# Investigation B: Do Behaviours Change with Seasonal Conditions (Winter vs Spring)?

## ✅ PROJECT STATUS: IN PROGRESS

This investigation compares bat foraging and anti-predator behaviours between Winter (scarce food, fewer rat encounters) and Spring (abundant food, more rat encounters). It extends Investigation A by introducing seasonal context and testing whether key behaviours differ across seasons.

## Project Overview
- Dataset1: Individual bat landing events with behavioural measures
- Dataset2: 30-minute environmental periods with rat activity context

## File Structure and Implementation

### 🧪 investigation_B.py – Main Analysis Script
- Status: ACTIVE – Seasonal analysis with hypothesis testing and GLMs
- Purpose: Assess whether behaviours change between Winter and Spring
- Key Features:
  - Data loading and cleaning (with habit classification fixes)
  - Merge with environmental context; derive `season_label`
  - Seasonal EDA and hypothesis testing (Winter vs Spring)
  - GLM models including season and interactions
  - Final seasonal summary and verdict
- Analysis Phases:
  - Phase 1: Data Overview and Loading
  - Phase 2: Updated Habit Classification and Cleaning
  - Phase 3: Seasonal Distributions and Outcomes
  - Phase 3.1: IQR Analysis and Correlation Mapping
  - Phase 4: Seasonal Hypothesis Testing (Winter vs Spring)
  - Phase 5: GLM Effects on Feeding Success (with season terms)
  - Phase 6: Final Seasonal Summary and Verdict

## 📊 Current Output Files

### Datasets (`datasets/`)
- `dataset1_cleaned.csv`
- `dataset1_merged_with_dataset2.csv`

### Visualizations (`plots/`)
- `Phase1_Data_Overview.png`
- `Phase2_Classification_Summary.png`
- `Phase3_Threat_Distributions_and_Outcomes.png`
- `Phase3.1_IQR_and_Correlation_Analysis.png`
- `Phase4_Hypotheses_Overview_(Threat_to_Responses).png`
- `Phase5_GLM_Effects_on_Feeding_Success.png`
- `Phase6_Final_Summary_and_Verdict.png`

## 🎯 Key Questions and Hypotheses

Primary question: Do bat behaviours differ between Winter and Spring?

Seasonal comparisons (Winter vs Spring):
1. Rat encounter frequency (`rat_arrival_number`)
2. Rat presence intensity (`rat_minutes`)
3. Vigilance (`bat_landing_to_food`)
4. Feeding success (`reward`)
5. Risk (`risk`)
6. Defensive behaviours (cautious, slow_approach, fight)

Methods:
- Mann–Whitney U tests and Fisher exact tests with FDR correction (Phase 4)
- GLM (binomial) for `reward` with season and interaction terms (Phase 5)

## 🚀 How to Run

```bash
# Run the complete investigation
python investigation_B.py
```

Requirements:
- Python 3.x
- Packages: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels

## 📈 Outputs and Interpretation
- Seasonal verdicts are labeled as:
  - “Higher in Spring” (p < 0.05)
  - “Higher in Winter” (p < 0.05)
  - “No seasonal difference” (p ≥ 0.05)
- Phase 6 summarizes counts of significant seasonal differences and direction.

## 🔬 Methodology Notes
1. Seasonal labels are derived from `start_time` (Northern Hemisphere mapping).
2. Habit classification includes corrections and smart imputations for unknowns.
3. FDR is applied across primary seasonal tests (arrivals, minutes, vigilance, success).
4. GLM includes season indicator (`is_spring`) and interactions:
   - `rat_minutes_is_spring`, `rat_arrival_number_is_spring`

## 📚 References
- Seasonal framing and GLM approach adapted from Investigation A and standard GLM practice.

---

Note: This document summarizes Investigation B with a seasonal focus. See Investigation A for baseline non-seasonal analysis and shared methodology.


