# An Explainable Risk-Triage System for Public Procurement Oversight in Kenya

An explainable AI system that ranks Kenyan government contracts by integrity risk and presents them to auditors as a prioritized review queue. Built for the CMU Africa AI System Design course (04-652). Addresses UN SDG 16: Peace, Justice and Strong Institutions.

---

## 📌 Problem Statement

Kenya publishes over 260,000 public procurement records through the Public Procurement Regulatory Authority (PPRA). However, manual audit capacity is severely limited. This system engineers domain-validated red-flag indicators from Open Contracting Data Standard (OCDS) releases and combines a rule-based composite index with anomaly detection to rank contracts by audit review priority.

---

## 🎯 Task Type

**Unsupervised Risk Ranking.** No ground-truth corruption labels exist in any public procurement release. Framing this as supervised regression or classification would require inventing labels, making any reported accuracy meaningless. The target $R(c)$ is constructed from domain-validated red-flag indicators following the Fazekas Corruption Risk Index methodology. The ML component is an Isolation Forest anomaly detector that identifies contracts unusual in the combination of their features, complementing the rule-based index.

- **Sample Size ($N$):** 277,325 merged rows; the 109,163 awarded contracts form the scoring unit
- **Feature Count ($P$):** 23 raw columns in the merged working table
- **Target Variable ($Y$):** Composite risk score $R(c) \in [0, 1]$, the average of the available red-flag indicators

---

## 📊 Data Overview

Everything in this section describes the **raw data, before any preprocessing**. The figures and numbers are reproduced by `phase1_strategy/data_overview.py`.

### Source

- **Publisher:** Kenya Public Procurement Regulatory Authority (PPRA), via the Open Contracting Partnership Data Registry, publication 147: <https://data.open-contracting.org/en/publication/147>
- **Standard:** Open Contracting Data Standard (OCDS) 1.1
- **Format:** one archive, `kenya_ppra_full.csv.tar.gz` (about 55 MB), containing five linked CSV files
- **Coverage:** national government and all 47 county governments, July 2018 to August 2026
- **Licence:** open government data under Executive Order No. 2 of 2018; no explicit licence is declared

### The Five Files

The release follows the life of a public purchase: a buyer announces a **tender**, picks a winner (the **award**), names the winning **supplier**, signs a **contract**, and every organization involved is listed as a **party**.

| File | Rows | Columns | One row is | What it tells us |
|---|---:|---:|---|---|
| `main.csv` | 259,709 | 15 | a tender | who is buying, what, how the supplier is chosen, and the bidding dates |
| `awards.csv` | 109,163 | 9 | an award | the value of each award and the planned contract period |
| `awards_suppliers.csv` | 79,870 | 5 | a named winner | which supplier won each award |
| `contracts.csv` | 109,163 | 12 | a signed contract | when the contract was signed and its status |
| `parties.csv` | 813,078 | 16 | an organization in a process | every buyer and supplier, with registration and contact details |

**How they link.** Every file links back to `main.csv` through `main_ocid`, which matches `ocid` in `main.csv`. Awards link to suppliers through `awards_suppliers.awards_id` and to contracts through `contracts.awardID`, both matching `awards.id`.

**What the row counts already show.** 259,709 tenders produced only 109,163 awards: 168,162 tenders (60.6%) have no recorded award. And 109,163 awards have only 79,870 named winners: 26.8% of awards do not record who won.

### Features in Each File

Missing percentages are measured on each file before merging.

#### `main.csv`: tenders (259,709 rows)

| Column | Type | Description | Missing % |
|---|---|---|---:|
| `id` | Nominal | Record ID of the compiled release | 0.00 |
| `tag` | Nominal | Release stage; always "compiled" | 0.00 |
| `date` | Datetime | When the record was published, not when the tender happened | 0.00 |
| `ocid` | Nominal | Open Contracting ID: one per procurement process; the main join key | 0.00 |
| `language` | Nominal | Always "en" | 0.00 |
| `initiationType` | Nominal | Always "tender" | 0.00 |
| `buyer_id` | Discrete | ID of the buying government entity (1,211 distinct buyers) | 0.00 |
| `buyer_name` | Nominal | Name of the buying entity | 0.00 |
| `tender_id` | Nominal | The buyer's own reference number | 0.00 |
| `tender_title` | Text | What is being purchased, in free text | 0.00 |
| `tender_mainProcurementCategory` | Nominal | Goods, works or services | 0.26 |
| `tender_procurementMethod` | Ordinal | How the supplier was chosen: open, selective, direct, and three rare methods | 0.00 |
| `tender_tenderPeriod_startDate` | Datetime | When bidding opened | 0.09 |
| `tender_tenderPeriod_endDate` | Datetime | The bidding deadline | 0.09 |
| `tender_awardCriteria` | Nominal | The rule used to pick the winner, such as lowest price | 79.30 |

#### `awards.csv`: awards (109,163 rows)

| Column | Type | Description | Missing % |
|---|---|---|---:|
| `id` | Nominal | Award ID; links to suppliers and contracts | 0.00 |
| `title` | Text | What was awarded | 0.00 |
| `description` | Text | Longer description of the award | 95.25 |
| `value_amount` | Continuous | Award value | 0.00 |
| `value_currency` | Nominal | Always "KES" (Kenyan Shilling) | 0.00 |
| `contractPeriod_startDate` | Datetime | Planned contract start | 34.60 |
| `contractPeriod_endDate` | Datetime | Planned contract end | 10.46 |
| `main_ocid` | Nominal | Link to the tender in `main.csv` | 0.00 |
| `main_id` | Nominal | Link to the release record in `main.csv` | 0.00 |

#### `awards_suppliers.csv`: winning suppliers (79,870 rows)

| Column | Type | Description | Missing % |
|---|---|---|---:|
| `id` | Nominal | Supplier record ID; not a stable firm ID (23.76% of names appear under more than one ID) | 0.00 |
| `name` | Text | Supplier name as written, with inconsistent spelling and case | 0.00 |
| `main_ocid` | Nominal | Link to the tender | 0.00 |
| `main_id` | Nominal | Link to the release record | 0.00 |
| `awards_id` | Nominal | Link to the award in `awards.csv` | 0.00 |

#### `contracts.csv`: signed contracts (109,163 rows)

| Column | Type | Description | Missing % |
|---|---|---|---:|
| `id` | Nominal | Contract reference, including the buyer's contract number | 0.00 |
| `title` | Text | What the contract is for | 0.00 |
| `description` | Text | Longer description | 95.25 |
| `dateSigned` | Datetime | When the contract was signed | 13.64 |
| `awardID` | Nominal | Link to the award in `awards.csv` | 0.00 |
| `value_amount` | Continuous | Contract value; matches the award value | 0.00 |
| `value_currency` | Nominal | Always "KES" | 0.00 |
| `period_startDate` | Datetime | Contract start | 3.12 |
| `period_endDate` | Datetime | Contract end | 10.46 |
| `main_ocid` | Nominal | Link to the tender | 0.00 |
| `main_id` | Nominal | Link to the release record | 0.00 |
| `status` | Nominal | Only ever filled as "terminated" (105 contracts) | 99.90 |

#### `parties.csv`: organizations (813,078 rows)

| Column | Type | Description | Missing % |
|---|---|---|---:|
| `id` | Nominal | Organization ID | 0.00 |
| `name` | Text | Organization name | 0.02 |
| `roles` | Nominal | "buyer" or "supplier" | 0.00 |
| `main_ocid` | Nominal | Link to the tender | 0.00 |
| `main_id` | Nominal | Link to the release record | 0.00 |
| `address_region`, `address_locality`, `address_countryName` | Nominal | Where the organization is based | 32.10 |
| `address_postalCode` | Nominal | Postal code | 85.15 |
| `address_streetAddress` | Text | Street address | 84.63 |
| `identifier_id` | Nominal | Registration number; present for 100% of suppliers | 32.10 |
| `identifier_scheme` | Nominal | Which registry issued the number | 32.10 |
| `identifier_legalName` | Text | Official registered name | 32.12 |
| `contactPoint_name` | Text | Contact person | 82.84 |
| `contactPoint_email` | Text | Contact email | 85.69 |
| `contactPoint_telephone` | Text | Contact phone number | 87.05 |

### The Merged Working Table

For analysis, `main`, `awards`, `awards_suppliers` and `contracts` are joined into one table of **277,325 rows by 23 columns**: one row per award, plus one row for each tender with no award. It is larger than `main.csv` because 91,547 tenders carry the 109,163 awards between them. Across the merged table, 27.73% of cells are missing. Only one row is complete across all 23 columns; excluding `status` and `tender_awardCriteria`, 38,538 rows (13.90%) are complete. The column-by-column profile is Table 3 of the Phase 1 report.

### Statistical Diagnostics

**Category frequencies (tender level).**

| Procurement method | Tenders | % |
|---|---:|---:|
| open | 233,594 | 89.94 |
| selective | 17,268 | 6.65 |
| direct | 8,744 | 3.37 |
| SpeciallyPermitted | 56 | 0.02 |
| AlternativeSelection | 41 | 0.02 |
| community participation | 6 | 0.00 |

Procurement category: works 115,930 (44.6%), goods 85,268 (32.8%), services 57,833 (22.3%), missing 678 (0.3%).

**Award values (all 109,163 awards).**

| Statistic | Value |
|---|---|
| Mean / median | 15,409,360 / 498,150 KES |
| Maximum | 160.6 billion KES |
| Quartiles Q1 / Q3 | 111,537 / 2,599,744 KES |
| Skewness | 161.0 (extremely right-skewed) |
| Skewness of log(1 + value) | −1.38, or −0.51 once the zeros are excluded |
| Zero-value awards | 1,129 (1.03%) |
| IQR upper bound (Q3 + 1.5 × IQR) | 6,332,053 KES; 12.0% of awards lie above it |
| 99th percentile | 135,000,000 KES |

**Dates.**

| Measure | Value |
|---|---|
| Advertisement period (bidding end − start) | 42.2% negative; median 0 days among the rest; 92.8% under 7 days |
| Decision period (signing − bidding end) | n = 94,140; median 22 days; 7.6% signed before bidding closed; skewness −2.69 |

**Missingness mechanism tests.** A chi-square test of independence between each column's missingness and the procurement competition level (open / restricted / non-competitive). A small p-value rejects Missing Completely at Random (MCAR). Cramér's V measures how strong the link is; with this many rows it matters more than p.

| Column | Missing % | χ² | p | Cramér's V | Missing % open / restricted / non-competitive | MCAR |
|---|---:|---:|---:|---:|---|---|
| `dateSigned` | 13.66 | 137.5 | 1.4e-30 | 0.035 | 13.93 / 12.96 / 7.35 | rejected |
| `contractPeriod_startDate` | 34.60 | 118.1 | 2.3e-26 | 0.033 | 34.60 / 28.95 / 40.57 | rejected |
| `contractPeriod_endDate` | 10.46 | 114.5 | 1.4e-25 | 0.032 | 10.42 / 7.39 / 14.68 | rejected |
| `supplier_name` | 26.83 | 21.7 | 1.9e-05 | 0.014 | 26.66 / 28.65 / 29.44 | rejected |
| `tender_mainProcurementCategory` | 0.26 | 1420.3 | 3.8e-309 | 0.074 | 0.17 / 1.67 / 0.01 | rejected |
| `tender_awardCriteria` | 79.30 | 493.7 | 6.2e-108 | 0.044 | 79.21 / 84.01 / 72.35 | rejected |

Award columns are tested on awarded rows only, tender columns on unique tenders. Every column rejects MCAR, but every link is weak (V below 0.1): the gaps are not random, yet procurement method explains only a small part of them. Missing Not at Random cannot be tested from data alone.

**Suppliers.** 23.76% of normalized supplier names appear under more than one supplier ID, while no ID carries more than one name, so the ID marks a record, not a firm. In 79.3% of processes, `parties.csv` lists more suppliers than there are winners (median 3), suggesting losing bidders are recorded too.

### Figures

**Missingness pattern.** The black block at the top is the 60.6% of tenders with no award: their award columns do not exist, rather than being missing.

![Missingness heatmap](figures/eda_01_missingness_heatmap.png)

**Missing rate by competition level.** The visual version of the MAR tests: rates differ by level, but only slightly.

![Missing rate by procurement method](figures/eda_02_missing_by_method.png)

**Tenders per year.** How the records spread over time, and the share with a recorded award.

![Tenders per year](figures/eda_03_tenders_per_year.png)

**Method and category frequencies.** Open tenders dominate; three methods have fewer than 60 tenders each.

![Method and category frequencies](figures/eda_04_method_category.png)

**Award values.** On a log scale the values become readable; the box plot shows the IQR bound and the long upper tail.

![Award value distribution](figures/eda_05_value_distribution.png)

**Date diagnostics.** Many advertisement periods are negative, which is a recording problem; contracts signed before bidding closed are a possible red flag.

![Date diagnostics](figures/eda_06_date_diagnostics.png)

### Strategy Tracking

How each finding above shaped the Phase 1 preprocessing strategy. Section numbers refer to `phase1_strategy/Phase1_Strategy_Report.pdf`.

| Finding in the raw data | Strategy decision | Report section |
|---|---|---|
| 60.6% of tenders have no award | Score awarded contracts only; audit the rest | 2.2 |
| `date` records publication, not the tender | Split by tender end date, not upload date | 6.1 |
| `tender_awardCriteria` 79.3% missing, MCAR rejected | Keep as a binary flag, never impute | 3.1 |
| 26.8% of awards name no winner | Binary flag `supplier_missing` | 3.1 |
| `status` only ever says "terminated" | Binary flag `is_terminated` | 3.1 |
| `dateSigned` missing depends on method (V = 0.035) | Median within procurement method | 3.1 |
| Category missing ten times more for restricted tenders | Mode within procurement method | 3.1 |
| Contract dates MAR but used by no indicator | Excluded from model inputs, not imputed | 3.1 |
| Award skewness 161.0; IQR bound flags 12.0% | IQR for detection only; cap at the 99th percentile | 3.2 |
| 1,129 zero-value awards | Flag; exclude from value-based indicators | 3.2 |
| Supplier IDs are not stable firm IDs | Three-stage entity resolution | 3.3 |
| Advertisement period 42.2% negative | Rejected; replaced by decision period | 4 |
| Decision period valid for 94,140 awards | Indicator $f_3$ | 4 |
| `parties.csv` lists extra suppliers in 79.3% of processes | Candidate single-bidding indicator $f_6$ | 4 |
| Three methods have under 60 tenders each | Group six methods into three competition levels | 5.1 |
| Log of value still skewed (−1.38) | Rank-based tests (Spearman) instead of Pearson | 5.3 |

---

## 📁 Repository Structure

```text
kenya-procurement-triage/
├── .gitignore                      # Excludes raw data, binaries, and local environments
├── requirements.txt                # Pinned package dependencies
├── README.md                       # Project guide and data overview
├── data/
│   ├── raw/                        # Original unaltered dataset (excluded from git)
│   └── processed/                  # Cleaned, encoded, and scaled final dataset
├── phase1_strategy/
│   ├── Phase1_Strategy_Report.pdf  # Pre-implementation audit and leakage plan
│   └── data_overview.py            # Reproduces the data overview figures and numbers
├── phase2_implementation/
│   ├── Phase2_Final_Report.pdf     # Decision log and strategy vs. reality audit (Phase 2)
│   └── notebooks/
│       └── pipeline_execution.ipynb # Executable notebook with diagnostics (Phase 2)
├── src/
│   ├── __init__.py                 # Package marker
│   ├── transformers.py             # Custom Scikit-Learn feature transformers
│   └── pipeline_builder.py         # Modular ColumnTransformer pipeline definitions
└── figures/                        # Publication-quality plots (300 DPI)
```

---

## ⚙️ How to Reproduce

Tested with Python 3.13.

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

### 3. Reproduce the Data Overview

```bash
python phase1_strategy/data_overview.py
```

This prints the file sizes, missingness and statistics above, and saves the six figures to `figures/`.
