# An Explainable Risk-Triage System for Public Procurement Oversight in Kenya

An explainable AI system that ranks Kenyan government contracts by integrity risk and presents them to auditors as a prioritized review queue. Built for the CMU Africa AI System Design course (04-652). Addresses UN SDG 16: Peace, Justice and Strong Institutions.

---

## 📌 Problem Statement

Kenya publishes over 260,000 public procurement records through the Public Procurement Regulatory Authority (PPRA). However, manual audit capacity is severely limited. This system engineers domain-validated red-flag indicators from Open Contracting Data Standard (OCDS) releases and combines a rule-based composite index with anomaly detection models to automatically rank contracts by audit review priority.

---

## 📊 Dataset Overview

- **Data Source:** Open Contracting Partnership Data Registry (Kenya PPRA, Publication 147)
  [https://data.open-contracting.org/en/publication/147](https://data.open-contracting.org/en/publication/147)
- **Files:** 5 linked relational CSV files (`main`, `awards`, `suppliers`, `contracts`, `parties`)
- **Sample Size ($N$):** 277,325 merged contract observations
- **Feature Count ($P$):** 20 raw feature columns before engineering
- **Target Variable ($Y$):** Continuous composite risk score $R(c) \in [0, 1]$, engineered from five normalized red-flag indicators (Fazekas Corruption Risk Index methodology)
- **Temporal Coverage:** July 2018 to August 2026

---

## 🎯 Task Type

**Unsupervised Risk Ranking**
No ground-truth corruption labels exist in any public procurement release.
Framing this as supervised regression would require inventing labels, making
any reported accuracy meaningless. The target variable Y is a composite risk
score R(c) ∈ [0,1] constructed from five domain-validated red-flag indicators
following the Fazekas Corruption Risk Index methodology. The ML component is
an Isolation Forest anomaly detector that identifies contracts unusual in the
combination of their features, complementing the rule-based index.

---

## 📁 Repository Structure

```text
kenya-procurement-triage/
├── .gitignore                      # Excludes raw data, binaries, and local environments
├── requirements.txt                # Fixed package dependencies
├── README.md                       # Project master guide & reproduction instructions
├── data/
│   ├── raw/                        # Original unaltered dataset (excluded from git)
│   └── processed/                  # Cleaned, encoded, and scaled final dataset
├── phase1_strategy/
│   └── Phase1_Strategy_Report.pdf  # Pre-implementation audit & leakage plan
├── phase2_implementation/
│   ├── Phase2_Final_Report.pdf     # Decision log & strategy vs. reality audit
│   └── notebooks/
│       └── pipeline_execution.ipynb# Executable Jupyter Notebook with diagnostics
├── src/
│   ├── __init__.py                 # Package marker
│   ├── transformers.py             # Custom Scikit-Learn feature transformers
│   └── pipeline_builder.py         # Modular ColumnTransformer pipeline definitions
└── figures/                        # Saved publication-quality diagnostic plots (300 DPI)
```

## ⚙️ How to Reproduce

### 1. Clone the Repository & Setup Environment

```bash
git clone [https://github.com/Razan-O-Elobeid/kenya-procurement-triage.git](https://github.com/Razan-O-Elobeid/kenya-procurement-triage.git)
cd kenya-procurement-triage

# Create and activate virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install exact dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

1. Download `kenya_ppra_full.csv.tar.gz` from the [Open Contracting Data Registry](https://data.open-contracting.org/en/publication/147).
2. Place the raw file in the `data/raw/` folder.

### 3. Run Pipeline & Diagnostics

To execute data preprocessing and build the Scikit-Learn transformers:

```bash

python -m src.pipeline_builder

```
