# Investigation A: Bat Predation Risk Analysis

## Project Overview
This project analyzes whether bats perceive rats as predators or just competitors by examining behavioral patterns and vigilance responses.

## File Structure and Workflow

### 🔍 **ml_algorithm_comparison.py** - ML Algorithm Testing Tool
- **Purpose**: Tests and compares multiple ML algorithms to find the best one for habit imputation
- **What it does**: 
  - Tests 13 different algorithms (KNN, Decision Trees, Random Forest, SVM, etc.)
  - Uses cross-validation to evaluate performance
  - Identifies the best algorithm based on accuracy, overfitting, and other metrics
- **Output**: Algorithm comparison results and recommendations
- **Accuracy claims**: Results from cross-validation testing (not final implementation)

### 📊 **investigation_A.py** - Main Investigation Implementation
- **Purpose**: Implements the complete investigation using the best algorithm found
- **Phases**:
  - Phase 1: Data Loading and Initial Exploration
  - Phase 2: Data Cleaning and Habit Analysis  
  - Phase 3: Missing Value Imputation (uses best algorithm from comparison)
  - Phase 4: Feature Engineering
  - Phase 5: Statistical Analysis
  - Phase 6: Visualization and Conclusion

## 🔄 **Correct Workflow**

1. **First**: Run `ml_algorithm_comparison.py` to test algorithms and find the best one
2. **Then**: Run `investigation_A.py` to implement the complete investigation
3. **Result**: The investigation uses the best algorithm parameters identified by the comparison tool

## ⚠️ **Important Notes**

- **Accuracy claims**: The 73.15% accuracy mentioned in both files comes from the ML comparison tool's cross-validation testing
- **Different purposes**: 
  - `ml_algorithm_comparison.py` = Testing/benchmarking tool
  - `investigation_A.py` = Actual implementation
- **XGBoost issue**: Fixed in the comparison tool with error handling and simplified parameters

## 🚀 **How to Run**

```bash
# Step 1: Test algorithms and find the best one
python ml_algorithm_comparison.py

# Step 2: Run the complete investigation
python investigation_A.py
```

## 📈 **Current Best Algorithm**
- **Gradient Boosting** with parameters:
  - `n_estimators=100`
  - `max_depth=3` 
  - `learning_rate=0.05`
  - `subsample=0.7`
  - `random_state=42`

## 🔧 **Recent Fixes**
- Fixed XGBoost NaN values by simplifying parameters
- Added error handling for failed algorithms
- Clarified relationship between testing tool and implementation
- Added comprehensive documentation about workflow 