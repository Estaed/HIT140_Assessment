# Investigation A: Bat Behavioral Analysis - Do Bats Perceive Rats as Predators?

## 🚧 **PROJECT STATUS: IN PROGRESS - NOT FINISHED YET**

This project analyzes whether bats perceive rats as potential predators or competitors by examining behavioral patterns, vigilance responses, and using machine learning classification.

## Project Overview
The investigation uses two datasets to analyze bat foraging behavior in the presence of rats:
- **Dataset1**: Individual bat landing events with detailed behavioral analysis
- **Dataset2**: Environmental context from 30-minute observation periods

## File Structure and Current Implementation

### 🤖 **investigation_A_ml.py** - ML-Enhanced Investigation (MAIN FILE)
- **Status**: ✅ Recently updated with fixed ML classification
- **Purpose**: Complete investigation with machine learning enhancement
- **Key Features**:
  - Uses SVM for competition vs predator classification
  - Separates ambiguous 'bat_and_rat' behaviors into clear categories
  - Vigilance-based training logic (fixed classification issue)
  - Comprehensive statistical analysis with ML-separated categories
- **Phases**:
  - Phase 1: Data Loading and Risk-Reward Analysis
  - Phase 2: Habit Classification and Data Cleaning
  - Phase 3: Smart Imputation for Unknown Values
  - Phase 4: ML Classification (Competition vs Predator)
  - Phase 5: Enhanced Statistical Analysis
  - Phase 6: Final Conclusions with ML Results

### 🧮 **ml_algorithm_comparison.py** - Algorithm Comparison Tool
- **Status**: ✅ Updated with competition/predator terminology
- **Purpose**: Tests and compares ML algorithms for classification
- **Key Features**:
  - Tests 13 different algorithms (KNN, SVM, Random Forest, etc.)
  - Uses competition vs predator terminology (not bat vs rat)
  - Cross-validation performance evaluation
  - **Result**: SVM (Linear) identified as best performer

### 📊 **investigation_A.py** - Base Investigation (Legacy)
- **Status**: ⚠️ Older version without ML enhancement
- **Purpose**: Original investigation without ML classification
- **Note**: Use `investigation_A_ml.py` for most current analysis

## 🔄 **Current Workflow**

### **Recommended Usage:**
1. **Run ML-enhanced version**: `python investigation_A_ml.py`
   - This includes all latest improvements and ML classification
   - Fixes the classification bias issue
   - Uses vigilance-based training for better results

### **Optional Algorithm Testing:**
2. **Test algorithms**: `python ml_algorithm_comparison.py`
   - Compare different ML approaches
   - Validate SVM as best choice

## 🎯 **Key Improvements Made**

### **Recently Fixed Issues:**
- ✅ **Classification Bias**: Fixed issue where all bat_and_rat entries were classified as 'predator'
- ✅ **Training Logic**: Switched from risk-reward only to vigilance-based classification
- ✅ **Algorithm Choice**: Updated to use SVM (best performer from comparison)
- ✅ **Terminology**: Consistent use of 'competition' vs 'predator' (clearer than 'bat' vs 'rat')

### **Current ML Approach:**
- **Training Data**: 
  - Competition: Fast approach (vigilance <10s) + success patterns
  - Predator: Cautious approach (vigilance >8s) + risk-taking patterns
- **Classification**: SVM with linear kernel for bat_and_rat entries
- **Feature Set**: Vigilance, timing, risk/reward, environmental factors

## 📈 **Expected Results**
The ML classification should now produce:
- Mixed classification of bat_and_rat entries (both competition and predator behaviors)
- More accurate behavioral separation based on vigilance patterns
- Better statistical analysis with properly separated categories

## ⚠️ **Still To Do**
- [ ] Feature engineering improvements
- [ ] Statistical analysis refinement
- [ ] Performance optimization
- [ ] Final validation and testing
- [ ] Documentation completion

## 🚀 **How to Run**

```bash
# Run the main ML-enhanced investigation
python investigation_A_ml.py

# Optional: Test algorithm comparison
python ml_algorithm_comparison.py
```

## 📁 **Output Files**
- **Datasets**: `datasets/ml_analysis/` - Cleaned and classified data
- **Plots**: `plots/ml_analysis/` - Visualization results for each phase
- **Algorithm Comparison**: `plots/ml_algorithm_comparison.png`

---
**Note**: This is an ongoing data science project for HIT140. The ML implementation has been recently updated and improved, but further refinement of feature engineering and statistical analysis is still in progress.