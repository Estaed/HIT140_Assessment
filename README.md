# HIT140 Data Science Project (CDU) – Investigations A & B

## 📚 Project Overview
This repository contains a project for the HIT140 Data Science class. It includes two complementary investigations analyzing bat behaviour and bat–rat interactions using two datasets:
- Dataset1: Individual bat landing events with detailed behavioural measures
- Dataset2: 30-minute environmental periods with rat activity context

## 📁 Repository Structure
- Investigation A/ – Non-seasonal analysis: predator vs competitor signals
- Investigation B/ – Seasonal analysis: Winter vs Spring comparisons
- Presentations/ – Slides and media used for reporting

## 🎯 Research Questions
- Investigation A: Do bats perceive rats as potential predators (vs competitors)?
  - Approach: Behavioural classification, threat–response correlations, hypothesis testing (8 hypotheses), and GLMs.
  - Short summary: Evidence synthesized across multiple phases indicates mixed-to-predator-leaning signals, with defensive/vigilance patterns assessed against rat activity and timing.

- Investigation B: Do behaviours change with seasonal conditions (Winter vs Spring)?
  - Approach: Seasonal EDA, Mann–Whitney/Fisher tests with FDR correction, and GLMs including season and interaction terms.
  - Short summary: Season-aware results are reported as “Higher in Spring”, “Higher in Winter”, or “No seasonal difference”, with a final tally and interpretation.

## 🚀 How to Run
From each investigation folder:
```bash
# Investigation A
python Investigation A/investigation_A.py

# Investigation B
python Investigation B/investigation_B.py
```

## 🧰 Requirements
- Python 3.x
- Packages: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels

## 🔎 Notes
- Visual outputs are saved under each investigation’s `plots/` folder.
- Cleaned/merged datasets are written to each investigation’s `datasets/` folder.

---

This is a project for the HIT140 Data Science class at CDU. It provides structured analyses, clear visualizations, and concise evidence summaries for both the predator-perception question (Investigation A) and the seasonal-change question (Investigation B).