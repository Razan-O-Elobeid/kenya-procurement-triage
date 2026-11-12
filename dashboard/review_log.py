"""Append-only review log (protocol section 11). SQLite rules block UPDATE and DELETE.

This is enforced at the database-rule level. It is not tamper-proof against someone
editing the file directly.
"""
import sqlite3
from datetime import datetime, timezone

from dashboard.text import STATUSES

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT NOT NULL,
    bundle TEXT NOT NULL,
    row_id INTEGER NOT NULL,
    contract_id TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ({", ".join(repr(s) for s in STATUSES)})),
    reviewer TEXT NOT NULL CHECK (length(trim(reviewer)) > 0),
    note TEXT NOT NULL DEFAULT ''
);
CREATE TRIGGER IF NOT EXISTS reviews_no_update BEFORE UPDATE ON reviews
BEGIN SELECT RAISE(ABORT, 'review log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS reviews_no_delete BEFORE DELETE ON reviews
BEGIN SELECT RAISE(ABORT, 'review log is append-only'); END;
"""


def connect(path):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def add_review(path, bundle, row_id, contract_id, model, status, reviewer, note=""):
    with connect(path) as con:
        con.execute("INSERT INTO reviews (created_utc, bundle, row_id, contract_id, model, status, reviewer, note) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (datetime.now(timezone.utc).isoformat(timespec="seconds"), bundle, int(row_id),
                     str(contract_id), model, status, reviewer.strip(), note.strip()))
    con.close()


def read_reviews(path, bundle=None):
    import pandas as pd
    with connect(path) as con:
        q = "SELECT * FROM reviews" + (" WHERE bundle = ?" if bundle else "") + " ORDER BY id"
        df = pd.read_sql_query(q, con, params=(bundle,) if bundle else None)
    con.close()
    return df


def latest_status(reviews):
    """Current status per (row_id, model): the latest row wins."""
    if reviews.empty:
        return {}
    last = reviews.sort_values("id").groupby(["row_id", "model"]).tail(1)
    return {(int(r), m): s for r, m, s in zip(last["row_id"], last["model"], last["status"])}
