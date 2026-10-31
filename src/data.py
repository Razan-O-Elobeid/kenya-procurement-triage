"""Load the PPRA release, build the working table and split it by time."""
import tarfile
from pathlib import Path

import pandas as pd

DATE_COLS = [
    "date",
    "tender_tenderPeriod_startDate",
    "tender_tenderPeriod_endDate",
    "contractPeriod_startDate",
    "contractPeriod_endDate",
    "dateSigned",
]


def read_tables(path, names):
    """Read CSVs from the extracted folder (data/raw/full) or the .tar.gz archive."""
    path = Path(path)
    if path.is_dir():
        return [pd.read_csv(path / f"{n}.csv", low_memory=False) for n in names]
    with tarfile.open(path, "r:gz") as t:
        return [pd.read_csv(t.extractfile(f"full/{n}.csv"), low_memory=False) for n in names]


def load_merged_table(path):
    """Join main, awards, awards_suppliers and contracts into one table (23 columns)."""
    main, awards, sup, con = read_tables(path, ["main", "awards", "awards_suppliers", "contracts"])

    aw = (awards
          .merge(sup[["awards_id", "name"]], left_on="id", right_on="awards_id", how="left")
          .rename(columns={"name": "supplier_name"})
          .merge(con[["awardID", "dateSigned", "status"]],
                 left_on="id", right_on="awardID", how="left"))
    keep = ["main_ocid", "value_amount", "value_currency", "supplier_name",
            "contractPeriod_startDate", "contractPeriod_endDate", "dateSigned", "status"]
    return main.merge(aw[keep], left_on="ocid", right_on="main_ocid", how="left")


def parse_dates(df):
    """Parse ISO timestamps to UTC. Bad values become NaT."""
    out = df.copy()
    for col in DATE_COLS:
        try:
            out[col] = pd.to_datetime(out[col], format="ISO8601", utc=True, errors="coerce")
        except (TypeError, ValueError):  # older pandas without ISO8601
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out


def temporal_split(df, start=2018, train_end=2023, end=2026):
    """Keep awarded contracts inside the window, sorted by tender end date.
    Returns the scoring table (unique row index, used later as a row id)
    and a boolean train mask."""
    awarded = df[df["value_amount"].notna()]
    year = awarded["tender_tenderPeriod_endDate"].dt.year
    aw = (awarded[year.between(start, end)]
          .sort_values("tender_tenderPeriod_endDate", kind="stable")
          .reset_index(drop=True))
    train = (aw["tender_tenderPeriod_endDate"].dt.year <= train_end).to_numpy()
    return aw, train
