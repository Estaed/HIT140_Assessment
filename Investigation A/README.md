# Investigation A: Bat Behavioral Analysis - Do Bats Perceive Rats as Predators?

## ✅ **PROJECT STATUS: COMPLETED**

This project analyzes whether bats perceive rats as potential predators or competitors by examining behavioral patterns, vigilance responses, and statistical analysis using two complementary datasets.

## Project Overview
The investigation uses two datasets to analyze bat foraging behavior in the presence of rats:
- **Dataset1**: Individual bat landing events with detailed behavioral analysis
- **Dataset2**: Environmental context from 30-minute observation periods

## File Structure and Implementation

### 🤖 **investigation_A.py** - Main Analysis Script
- **Status**: ✅ **COMPLETED** - Final version with comprehensive analysis
- **Purpose**: Complete investigation with statistical analysis and hypothesis testing
- **Key Features**:
  - Data loading and cleaning
  - Risk-reward analysis and behavioral classification
  - Statistical hypothesis testing (8 hypotheses)
  - Generalized Linear Model (GLM) analysis
  - Comprehensive visualization across 7 phases
- **Analysis Phases**:
  - **Phase 1**: Data Overview and Loading
  - **Phase 2**: Classification Summary and Data Cleaning
  - **Phase 3**: Threat Distributions and Behavioral Outcomes
  - **Phase 3.1**: IQR Analysis and Correlation Mapping
  - **Phase 4**: Hypotheses Overview (Threat to Response Analysis)
  - **Phase 5**: GLM Effects on Feeding Success
  - **Phase 6**: Final Summary and Verdict

## 📊 **Current Output Files**

### **Datasets** (`datasets/` folder):
- `dataset1_cleaned.csv` - Cleaned individual bat landing events
- `dataset1_merged_with_dataset2.csv` - Merged dataset with environmental context

### **Visualizations** (`plots/` folder):
- `Phase1_Data_Overview.png` - Initial data loading and overview
- `Phase2_Classification_Summary.png` - Behavioral classification results
- `Phase3_Threat_Distributions_and_Outcomes.png` - Threat analysis and outcomes
- `Phase3.1_IQR_and_Correlation_Analysis.png` - Statistical distributions and correlations
- `Phase4_Hypotheses_Overview_(Threat_to_Responses).png` - Hypothesis testing results
- `Phase5_GLM_Effects_on_Feeding_Succes.png` - Statistical model effects
- `Phase6_Final_Summary_and_Verdict.png` - Final conclusions and verdict

## 🎯 **Key Findings**

### **Hypotheses Tested (8 total)**:
**Core Threat Gradient Hypotheses:**
1. **H1**: **Temporal Proximity → Vigilance** - Bats show higher vigilance when rats recently arrived (negative correlation expected)
2. **H2**: **Threat Intensity → Success** - Bats reduce feeding success with higher rat presence intensity (negative correlation expected)
3. **H3**: **Threat Frequency → Vigilance** - Bats increase vigilance with more frequent rat encounters (positive correlation expected)
4. **H4**: **Threat Frequency → Success** - Bats reduce feeding success when rat encounters are frequent (negative correlation expected)

**Behavioral Response Hypotheses:**
5. **H5**: **Risk-Taking Under Threat** - Bats reduce risk-taking behaviors under higher threat conditions (proximity, intensity, frequency)
6. **H6**: **Defensive Behaviors Under Threat** - Bats show more defensive behaviors (cautious, slow approach, fight) with higher threat levels
7. **H7**: **Time-of-Night Effects** - Anti-predator behavior changes with time (control variable for temporal effects)
8. **H8**: **Composite Threat Response** - Overall threat index predicts increased vigilance and defensive behaviors

### **Statistical Analysis**:
- **Phase 3.1**: IQR analysis and correlation mapping for data quality assessment
- **Phase 4**: Spearman correlation analysis for threat-response relationships  
- **Phase 5**: Generalized Linear Model (GLM) analysis with controlled effects
- **Phase 6**: Evidence-based verdict system and comprehensive statistical summary

## 🚀 **How to Run**

```bash
# Run the complete investigation
python investigation_A.py
```

**Requirements**:
- Python 3.x
- Required packages: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn

## 📈 **Project Results**

The investigation provides a comprehensive analysis of bat-rat interactions through:
- **Behavioral Classification**: Categorizing bat behaviors as predator-avoidance or competition
- **Statistical Analysis**: IQR analysis, correlation mapping, and distribution assessment
- **Hypothesis Testing**: Rigorous hypothesis testing with evidence-based conclusions
- **Visualization**: Seven detailed phases of analysis with clear visual outputs
- **Final Verdict**: Evidence-based conclusion on whether bats perceive rats as predators

## 📁 **Output Structure**

```
Investigation A/
├── investigation_A.py          # Main analysis script
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
    ├── Phase5_GLM_Effects_on_Feeding_Succes.png
    └── Phase6_Final_Summary_and_Verdict.png
```

## 🔬 **Methodology**

1. **Data Integration**: Merged individual bat events with environmental context
2. **Behavioral Classification**: Categorized behaviors based on risk-reward patterns  
3. **Statistical Quality Assessment**: IQR analysis, outlier detection, and correlation mapping
4. **Hypothesis Testing**: Tested 8 specific hypotheses using threat gradient analysis
5. **Multivariate Analysis**: GLM models controlling for confounding variables
6. **Evidence Synthesis**: Combined statistical results into final verdict with evidence classification

## 📚 **References**

**GLM Methodology Source:**
Chen, X., Harten, L., Rachum, A., Attia, L., & Yovel, Y. (2025). Complex competition interactions between Egyptian fruit bats and black rats in the real world. Mendeley Data, V2. https://data.mendeley.com/datasets/gt7j39b2cf/2

*This project utilizes the Generalized Linear Model (GLM) methodology and statistical approaches described in the referenced research, which investigates similar bat-rat behavioral interactions using comparable datasets and analytical frameworks.*

---

**Note**: This is a completed data science project for HIT140. The analysis provides comprehensive evidence-based conclusions about bat-rat behavioral interactions through rigorous statistical methodology and clear visualization.