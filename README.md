# An Explainable Risk-Triage System for Public Procurement Oversight in Kenya

An explainable AI system that ranks Kenyan government contracts by integrity risk and presents them to auditors as a prioritized review queue. Built for the CMU Africa AI System Design course (04-652). Addresses UN SDG 16: Peace, Justice and Strong Institutions.

---

## 📌 Problem Statement

Kenya publishes over 260,000 public procurement records through the Public Procurement Regulatory Authority (PPRA). However, manual audit capacity is severely limited. This system engineers domain-validated red-flag indicators from Open Contracting Data Standard (OCDS) releases and combines a rule-based composite index with anomaly detection to rank contracts by audit review priority.

---

## 📊 Dataset Overview

- **Data Source:** Open Contracting Partnership Data Registry (Kenya PPRA, Publication 147)
  <https://data.open-contracting.org/en/publication/147>
- **Files:** 5 linked relational CSV files (`main`, `awards`, `awards_suppliers`, `contracts`, `parties`)
- **Sample Size ($N$):** 277,325 merged rows; the 109,163 awarded contracts form the scoring unit
- **Feature Count ($P$):** 23 raw columns before engineering
- **Target Variable ($Y$):** Composite risk score $R(c) \in [0, 1]$, the average of the available red-flag indicators (Fazekas Corruption Risk Index methodology)
- **Temporal Coverage:** July 2018 to August 2026

---

## 🎯 Task Type

**Unsupervised Risk Ranking.** No ground-truth corruption labels exist in any public procurement release. Framing this as supervised regression or classification would require inventing labels, making any reported accuracy meaningless. The target $R(c)$ is constructed from domain-validated red-flag indicators following the Fazekas Corruption Risk Index methodology. The ML component is an Isolation Forest anomaly detector that identifies contracts unusual in the combination of their features, complementing the rule-based index.

---

## 📁 Repository Structure

```text
kenya-procurement-triage/
├── .gitignore                      # Excludes raw data, binaries, and local environments
├── requirements.txt                # Pinned package dependencies
├── README.md                       # Project guide and reproduction instructions
├── data/
│   ├── raw/                        # Original unaltered dataset (excluded from git)
│   └── processed/                  # Cleaned, encoded, and scaled final dataset
├── phase1_strategy/
│   └── Phase1_Strategy_Report.pdf  # Pre-implementation audit and leakage plan
├── phase2_implementation/
│   ├── Phase2_Final_Report.pdf     # Decision log and strategy vs. reality audit (Phase 2)
│   └── notebooks/
│       └── pipeline_execution.ipynb # Executable notebook with diagnostics (Phase 2)
├── src/
│   ├── __init__.py                 # Package marker
│   ├── transformers.py             # Custom Scikit-Learn feature transformers
│   └── pipeline_builder.py         # Modular ColumnTransformer pipeline definitions
└── figures/                        # Publication-quality diagnostic plots (300 DPI)
```

---

## ⚙️ How to Reproduce

Tested with Python 3.13 (Google Colab).

### 1. Clone the Repository and Set Up the Environment

```bash
git clone https://github.com/Razan-O-Elobeid/kenya-procurement-triage.git
cd kenya-procurement-triage

# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare the Data

1. Download `kenya_ppra_full.csv.tar.gz` (approximately 55 MB) from the [Open Contracting Data Registry](https://data.open-contracting.org/en/publication/147).
2. Place the file in the `data/raw/` folder. Raw data is excluded from version control.

### 3. Run the Pipeline

The preprocessing pipeline is implemented in Phase 2. Once added, run:

```bash
jupyter notebook phase2_implementation/notebooks/pipeline_execution.ipynb
```
