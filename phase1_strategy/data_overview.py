"""Phase 1 data overview: describe the raw release before any preprocessing.

Run from the repo root:  python phase1_strategy/data_overview.py
Reads data/raw/kenya_ppra_full.csv.tar.gz and saves figures to figures/.
"""
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "kenya_ppra_full.csv.tar.gz"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="paper")

FILES = ["main", "awards", "awards_suppliers", "contracts", "parties"]
DATES = ["date", "tender_tenderPeriod_startDate", "tender_tenderPeriod_endDate",
         "contractPeriod_startDate", "contractPeriod_endDate", "dateSigned"]
LEVEL = {"open": "open", "selective": "restricted", "AlternativeSelection": "restricted",
         "communityparticipartion": "restricted", "direct": "non-competitive",
         "SpeciallyPermitted": "non-competitive"}


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def to_dt(s):
    try:
        return pd.to_datetime(s, format="ISO8601", utc=True, errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(s, utc=True, errors="coerce")


# ---------- load the five files ----------
with tarfile.open(DATA, "r:gz") as t:
    raw = {f: pd.read_csv(t.extractfile(f"full/{f}.csv"), low_memory=False) for f in FILES}

print("\nFILES")
for f, d in raw.items():
    print(f"  {f}.csv: {d.shape[0]:,} rows x {d.shape[1]} columns")

print("\nMISSING % PER COLUMN, BY FILE")
for f, d in raw.items():
    miss = (d.isna().mean() * 100).round(2)
    print(f"  {f}.csv:", ", ".join(f"{c} {p}" for c, p in miss[miss > 0].items()) or "none")

# ---------- merged working table (23 columns) ----------
main, awards, sup, con = raw["main"], raw["awards"], raw["awards_suppliers"], raw["contracts"]
aw = (awards.merge(sup[["awards_id", "name"]], left_on="id", right_on="awards_id", how="left")
            .rename(columns={"name": "supplier_name"})
            .merge(con[["awardID", "dateSigned", "status"]], left_on="id", right_on="awardID", how="left"))
keep = ["main_ocid", "value_amount", "value_currency", "supplier_name",
        "contractPeriod_startDate", "contractPeriod_endDate", "dateSigned", "status"]
df = main.merge(aw[keep], left_on="ocid", right_on="main_ocid", how="left")
for c in DATES:
    df[c] = to_dt(df[c])

awarded = df["value_amount"].notna()
tenders = df.drop_duplicates("ocid").copy()
print(f"\nmerged: {df.shape[0]:,} x {df.shape[1]} | unawarded tenders: {(~awarded).sum():,} "
      f"({(~awarded).mean():.1%}) | overall missing {df.isna().mean().mean():.2%}")

# ---------- figure 1: missingness heatmap ----------
order = df.isna().mean().sort_values(ascending=False)
sample = df.sample(min(3000, len(df)), random_state=0)
sample = sample.loc[sample["value_amount"].notna().sort_values().index]  # unawarded first
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(sample[order.index].isna().to_numpy(), aspect="auto", cmap="Greys", interpolation="nearest")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"{c} ({p:.1%})" for c, p in order.items()], rotation=75, ha="right", fontsize=6)
ax.set_ylabel("rows (random sample of 3,000, unawarded first)")
ax.set_title("Missingness pattern of the merged table (black = missing)")
save(fig, "eda_01_missingness_heatmap.png")

# ---------- figure 2: missingness by competition level ----------
lvl = df["tender_procurementMethod"].map(LEVEL)
lvl_t = tenders["tender_procurementMethod"].map(LEVEL)
rates = {c: df.loc[awarded, c].isna().groupby(lvl[awarded]).mean() * 100
         for c in ["dateSigned", "contractPeriod_startDate", "contractPeriod_endDate", "supplier_name"]}
for c in ["tender_mainProcurementCategory", "tender_awardCriteria"]:
    rates[c] = tenders[c].isna().groupby(lvl_t).mean() * 100
rates = pd.DataFrame(rates).T[["open", "restricted", "non-competitive"]]
fig, ax = plt.subplots(figsize=(9, 4))
rates.plot.barh(ax=ax, color=["#4c72b0", "#dd8452", "#c44e52"])
ax.set_xlabel("missing % (award columns: awarded rows; tender columns: unique tenders)")
ax.set_title("Missing rate by procurement competition level")
ax.legend(title="level", loc="lower right")
save(fig, "eda_02_missing_by_method.png")

# ---------- figure 3: tenders per year ----------
year = tenders["tender_tenderPeriod_endDate"].dt.year
has_award = tenders["ocid"].isin(df.loc[awarded, "ocid"])
per_year = pd.crosstab(year[year.between(2018, 2026)], has_award[year.between(2018, 2026)])
per_year.columns = ["no award" if not c else "awarded" for c in per_year.columns]
per_year.index = per_year.index.astype(int)
fig, ax = plt.subplots(figsize=(8, 4))
per_year.plot.bar(stacked=True, ax=ax, color=["#bbbbbb", "#4c72b0"])
ax.set_xlabel("year of tender end date")
ax.set_ylabel("tenders")
ax.set_title("Tenders per year, by whether an award is recorded")
save(fig, "eda_03_tenders_per_year.png")

# ---------- figure 4: method and category frequencies ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
m = tenders["tender_procurementMethod"].value_counts()
axes[0].barh(m.index, m.values, color="#4c72b0")
axes[0].set_xscale("log")
axes[0].set_xlabel("tenders (log scale)")
axes[0].set_title("Procurement method")
for i, n in enumerate(m.values):
    axes[0].text(n, i, f" {n:,}", va="center", fontsize=7)
cat = tenders["tender_mainProcurementCategory"].fillna("missing").value_counts()
axes[1].bar(cat.index, cat.values, color="#55a868")
axes[1].set_ylabel("tenders")
axes[1].set_title("Procurement category")
for i, n in enumerate(cat.values):
    axes[1].text(i, n, f"{n:,}", ha="center", va="bottom", fontsize=7)
fig.suptitle("Category frequencies (tender level)")
fig.tight_layout()
save(fig, "eda_04_method_category.png")

# ---------- figure 5: award value distribution ----------
v = awards["value_amount"]
q1, q3 = v.quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
print(f"\nvalue_amount: mean {v.mean():,.0f} | median {v.median():,.0f} | max {v.max():,.0f} | "
      f"skew {stats.skew(v):.1f} | zeros {(v == 0).sum():,}")
print(f"  IQR upper {upper:,.0f} ({(v > upper).mean():.1%} above) | P99 {v.quantile(0.99):,.0f}")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(np.log10(v[v > 0]), bins=60, ax=axes[0], color="#4c72b0")
axes[0].set_xlabel("log10(award value in KES), zero values excluded")
axes[0].set_title(f"Distribution (raw skewness {stats.skew(v):.0f}; {(v == 0).sum():,} zeros)")
sns.boxplot(y=np.log10(1 + v), ax=axes[1], color="lightsteelblue")
axes[1].axhline(np.log10(1 + upper), color="darkred", ls="--", label=f"IQR upper bound ({upper:,.0f} KES)")
axes[1].set_ylabel("log10(1 + award value in KES)")
axes[1].set_title(f"Box plot: {(v > upper).mean():.1%} of awards above the IQR bound")
axes[1].legend(loc="lower right", fontsize=7)
fig.suptitle("Award values, raw data (all 109,163 awards)" if len(v) == 109163 else "Award values, raw data")
fig.tight_layout()
save(fig, "eda_05_value_distribution.png")

# ---------- figure 6: date diagnostics ----------
advert = (tenders["tender_tenderPeriod_endDate"] - tenders["tender_tenderPeriod_startDate"]).dt.days.dropna()
sign = (df.loc[awarded, "dateSigned"] - df.loc[awarded, "tender_tenderPeriod_endDate"]).dt.days.dropna()
print(f"\nadvertisement period: negative {(advert < 0).mean():.1%} | median of the rest "
      f"{advert[advert >= 0].median():.0f} days")
print(f"days to signing: n {len(sign):,} | median {sign.median():.0f} | negative {(sign < 0).mean():.1%} | "
      f"skew {stats.skew(sign):.2f}")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, s, title in [(axes[0], advert, "Advertisement period (end - start)"),
                     (axes[1], sign, "Days from bidding close to signing")]:
    clipped = s.clip(-365, 365)
    sns.histplot(clipped, bins=73, ax=ax, color="#4c72b0")
    ax.axvline(0, color="darkred", ls="--")
    ax.set_xlabel("days (clipped to +/- 365)")
    ax.set_title(f"{title}: {(s < 0).mean():.1%} negative")
fig.suptitle("Date diagnostics: negative values are recording problems or red flags")
fig.tight_layout()
save(fig, "eda_06_date_diagnostics.png")

print("\ndone. figures in", FIG)
