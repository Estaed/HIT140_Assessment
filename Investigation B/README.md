# Investigation B: Do Behaviours Change with Seasonal Conditions (Winter vs Spring)?

## ✅ **PROJECT STATUS: IN PROGRESS**

This project analyzes whether bat behaviours differ between Winter (scarce food, fewer rat encounters) and Spring (abundant food, more rat encounters) by examining behavioural patterns, seasonal comparisons, and statistical analysis using two complementary datasets.

## Project Overview
The investigation uses two datasets to analyze bat foraging behaviour with seasonal context:
- **Dataset1**: Individual bat landing events with detailed behavioural analysis
- **Dataset2**: Environmental context from 30-minute observation periods

## File Structure and Implementation

### 🤖 **investigation_B.py** - Main Analysis Script
- **Status**: 🚧 **IN PROGRESS** - Seasonal analysis with hypothesis testing and GLMs
- **Purpose**: Assess whether behaviours change between Winter and Spring
- **Key Features**:
  - Data loading and cleaning (standardized habit classification)
  - Environmental merge and `season_label` derivation
  - Seasonal hypothesis testing (Winter vs Spring) with FDR correction
  - GLM analysis including season indicator and interactions
  - Comprehensive visualization across 7 phases
- **Analysis Phases**:
  - **Phase 1**: Data Overview and Loading
  - **Phase 2**: Classification Summary and Data Cleaning
  - **Phase 3**: Seasonal Distributions and Behavioural Outcomes
  - **Phase 3.1**: IQR Analysis and Correlation Mapping
  - **Phase 4**: Seasonal Hypotheses Overview (Winter vs Spring)
  - **Phase 5**: GLM Effects on Feeding Success (with season terms)
  - **Phase 6**: Final Seasonal Summary and Verdict

## 📊 **Current Output Files**

### **Datasets** (`datasets/` folder):
- `dataset1_cleaned.csv` - Cleaned individual bat landing events
- `dataset1_merged_with_dataset2.csv` - Merged dataset with environmental context

### **Visualizations** (`plots/` folder):
- `Phase1_Data_Overview.png` - Initial data loading and overview
- `Phase2_Classification_Summary.png` - Behavioural classification results
- `Phase3_Threat_Distributions_and_Outcomes.png` - Seasonal threat/outcome patterns
- `Phase3.1_IQR_and_Correlation_Analysis.png` - Distributions and correlations
- `Phase4_Hypotheses_Overview_(Threat_to_Responses).png` - Seasonal test results
- `Phase5_GLM_Effects_on_Feeding_Success.png` - GLM with season and interactions
- `Phase6_Final_Summary_and_Verdict.png` - Final seasonal conclusions and verdict

## 🎯 **Key Findings (Planned Tests)**

**Seasonal Comparisons (Winter vs Spring):**
1. **H1**: Rat encounter frequency (`rat_arrival_number`)
2. **H2**: Rat presence intensity (`rat_minutes`)
3. **H3**: Vigilance (`bat_landing_to_food`)
4. **H4**: Feeding success (`reward`)
5. **H5**: Risk (`risk`, 0/1)
6. **H6**: Defensive behaviours (cautious, slow_approach, fight)

### **Statistical Analysis**:
- **Phase 3.1**: IQR analysis and correlation mapping
- **Phase 4**: Mann–Whitney U / Fisher exact for seasonal differences (FDR applied)
- **Phase 5**: Binomial GLM for success with season and interaction terms
- **Phase 6**: Season-aware verdict system and summary

## 🚀 **How to Run**

```bash
# Run the complete investigation
python investigation_B.py
```

**Requirements**:
- Python 3.x
- Required packages: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels

## 📈 **Project Results**

This investigation provides seasonal comparisons of bat–rat interactions through:
- **Behavioural Classification**: Standardized categories for robust comparisons
- **Statistical Analysis**: IQR analysis, seasonal hypothesis testing, GLMs
- **Visualization**: Seven phases focusing on seasonal patterns and outcomes
- **Final Verdict**: Evidence-based season-aware conclusions

## 📁 **Output Structure**

```
Investigation B/
├── investigation_B.py          # Main analysis script
├── dataset1.csv               # Original bat landing events
├── dataset2.csv               # Original environmental data
├── datasets/
│   ├── dataset1_cleaned.csv
│   └── dataset1_merged_with_dataset2.csv
└── plots/
    ├── Phase1_Data_Overview.png
    ├── Phase2_Classification_Summary.png
    ├── Phase3_Threat_Distributions_and_Outcomes.png
    ├── Phase3.1_IQR_and_Correlation_Analysis.png
    ├── Phase4_Hypotheses_Overview_(Threat_to_Responses).png
    ├── Phase5_GLM_Effects_on_Feeding_Success.png
    └── Phase6_Final_Summary_and_Verdict.png
```

## 🔬 **Methodology**

1. **Data Integration**: Merged individual bat events with environmental context and derived seasons
2. **Behavioural Classification**: Standardized and imputed categories for analysis
3. **Statistical Quality Assessment**: IQR analysis, outlier detection, correlations
4. **Seasonal Hypothesis Testing**: Winter vs Spring comparisons with FDR
5. **Multivariate Analysis**: GLM models controlling for confounders, season, interactions
6. **Evidence Synthesis**: Season-aware verdict and visualization

## 📚 **References**

Seasonal GLM framing and methodology follow the same conventions as Investigation A. See Investigation A references for primary GLM methodology and data context.

---

**Note**: This is the seasonal extension of Investigation A. It focuses on Winter vs Spring differences and uses the same analytical framework with season-specific tests and visualizations.


