"""Supplier entity resolution: candidate pairs, acronym rule and a labelling sample.

Stage 1 (name cleaning) lives in features.normalise_names.
Stage 2 here: fuzzy matching of every name against every other (no blocking,
so typos in the first word are still caught), plus acronyms.
The data has no company-level ID, so the match threshold is chosen
on a hand-labelled sample rather than assumed.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

STOP = {"AND", "OF", "THE", "LTD", "CO", "LIMITED", "COMPANY", "&"}
LEGAL = {"CO", "COMPANY"}  # sometimes part of an acronym (KPLC), sometimes not
BANDS = [80, 85, 90, 95, 100]


def _sorted_words(name):
    return " ".join(sorted(name.split()))


def candidate_pairs(names, min_score=80, chunk=1000):
    """All pairs of distinct cleaned names with token sort ratio >= min_score.
    Words are sorted once up front, so a plain ratio equals token sort ratio."""
    s = pd.Series(pd.unique(pd.Series(names).dropna())).reset_index(drop=True)
    keyed = s.map(_sorted_words).tolist()
    rows = []
    for start in range(0, len(keyed), chunk):
        m = process.cdist(keyed[start:start + chunk], keyed, scorer=fuzz.ratio,
                          score_cutoff=min_score, dtype=np.uint8, workers=-1)
        i, j = np.nonzero(m)
        keep = j > i + start  # each pair once, no self-pairs
        i, j = i[keep], j[keep]
        score = m[i, j].astype(int)
        # uint8 rounds 99.5 up to 100, so keep 100 for identical word sets only
        same = np.array([keyed[a + start] == keyed[b] for a, b in zip(i, j)], dtype=bool)
        score = np.where(same, score, np.minimum(score, 99))
        rows.append(pd.DataFrame({"name_a": s.iloc[i + start].to_numpy(),
                                  "name_b": s.iloc[j].to_numpy(),
                                  "score": score}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["name_a", "name_b", "score"])
    return out.assign(kind="fuzzy")


def _content_words(name):
    return [w for w in name.split() if w not in STOP]


def _initials(name):
    """Initials with and without CO/COMPANY, e.g. KPL and KPLC."""
    words = _content_words(name)
    with_legal = [w for w in name.split() if w not in STOP - LEGAL]
    return {"".join(w[0] for w in words), "".join(w[0] for w in with_legal)}


def acronym_pairs(names):
    """Short single-word names matched to longer names with the same initials,
    e.g. KPLC and KENYA POWER AND LIGHTING CO."""
    s = pd.Series(pd.unique(pd.Series(names).dropna()))
    words = s.map(_content_words)
    short = s[words.map(len).eq(1) & words.str[0].str.fullmatch(r"[A-Z]{2,6}").eq(True)]
    long_ = s[words.map(len).ge(2)]
    index = {}
    for name in long_:
        for ini in _initials(name):
            index.setdefault(ini, []).append(name)
    rows = []
    for name in short:
        for other in index.get(_content_words(name)[0], []):
            rows.append((name, other, int(fuzz.token_sort_ratio(name, other))))
    return pd.DataFrame(rows, columns=["name_a", "name_b", "score"]).drop_duplicates().assign(kind="acronym")


def labelling_sample(pairs, volume, per_cell=22, n_acronym=24, high_q=0.95, seed=0):
    """Stratified sample: similarity band x supplier volume, plus acronym pairs.
    volume: number of awards per cleaned name. Returns (sample, stratum sizes)."""
    p = pairs.copy()
    p["awards_a"] = p["name_a"].map(volume).fillna(0).astype(int)
    p["awards_b"] = p["name_b"].map(volume).fillna(0).astype(int)
    cut = volume.quantile(high_q)
    p["volume"] = np.where(p[["awards_a", "awards_b"]].max(axis=1) >= cut, "high", "small")
    band = pd.cut(p["score"], BANDS, right=False, labels=["80-85", "85-90", "90-95", "95-100"])
    p["stratum"] = np.where(p["kind"].eq("acronym"), "acronym",
                            band.astype(str) + " | " + p["volume"])
    p = p[(p["kind"] == "acronym") | (p["score"] < 100)]  # 100 = same words reordered, merged without labelling

    sizes = p["stratum"].value_counts()
    parts = []
    for stratum, grp in p.groupby("stratum"):
        n = n_acronym if stratum == "acronym" else per_cell
        parts.append(grp.sample(min(n, len(grp)), random_state=seed))
    sample = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    sample.insert(0, "pair_id", range(1, len(sample) + 1))
    sample["label"] = ""
    cols = ["pair_id", "stratum", "score", "name_a", "name_b", "awards_a", "awards_b", "label"]
    return sample[cols], sizes.rename("pairs_in_stratum")


def _metrics(lab, t, use_acronym):
    pred = lab["score"] >= t
    if use_acronym:
        pred = pred | (lab["stratum"] == "acronym")
    else:
        pred = pred & (lab["stratum"] != "acronym")
    y, w = lab["label"] == 1, lab["w"]
    tp, fp = (w * (pred & y)).sum(), (w * (pred & ~y)).sum()
    fn, tn = (w * (~pred & y)).sum(), (w * (~pred & ~y)).sum()
    nan = np.nan
    return {"precision": tp / (tp + fp) if tp + fp else nan,
            "recall": tp / (tp + fn) if tp + fn else nan,
            "false-match rate": fp / (fp + tn) if fp + tn else nan}


def evaluate(labelled, sizes, thresholds=(85, 90, 95), use_acronym=True,
             n_boot=2000, seed=0):
    """Precision, recall and false-match rate per threshold, weighted back to
    stratum sizes, with 95% bootstrap intervals (resampling within strata).
    Recall only covers candidate pairs (score 80+ or acronym)."""
    lab = labelled[labelled["label"].astype(str).isin(["0", "1"])].copy()
    lab["label"] = lab["label"].astype(int)
    lab["w"] = lab["stratum"].map(sizes / lab["stratum"].value_counts())
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in lab.groupby("stratum")]
    boot = {t: [] for t in thresholds}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        b = lab.loc[idx]
        for t in thresholds:
            boot[t].append(_metrics(b, t, use_acronym))
    rows = []
    for t in thresholds:
        point = _metrics(lab, t, use_acronym)
        bt = pd.DataFrame(boot[t])
        for m, v in point.items():
            lo, hi = bt[m].quantile([0.025, 0.975])
            rows.append({"threshold": t, "metric": m, "estimate": v, "low": lo, "high": hi})
    out = pd.DataFrame(rows).round(3)
    return out.pivot(index="threshold", columns="metric", values="estimate"), out, len(lab)
# ---- option A: stronger stage 1 (core names), fixed before the fresh labels ----

DROP = {"LTD", "LIMTED", "CO", "AND", "THE"}  # legal forms and connectors carry no identity
DROP_PAIRS = {("M", "S"), ("T", "A")}         # M/S and T/A after punctuation removal


def core_name(name):
    """Name with legal forms, M/S, T/A, AND, THE and plural S removed."""
    w, out, i = name.split(), [], 0
    while i < len(w):
        if i + 1 < len(w) and (w[i], w[i + 1]) in DROP_PAIRS:
            i += 2
            continue
        out.append(w[i])
        i += 1
    out = [x for x in out if x not in DROP]
    out = [x[:-1] if len(x) > 3 and x.endswith("S") and not x.endswith("SS") else x for x in out]
    return " ".join(out)


def core_pairs(names, min_score=85):
    """Pairs of cleaned names whose cores are identical (spaces and word order ignored)
    or whose cores score >= min_score. core_score is 100 for identical cores."""
    df = pd.DataFrame({"name": pd.unique(pd.Series(names).dropna())})
    df["core"] = df["name"].map(core_name)
    df = df[df["core"] != ""]
    df["key"] = df["core"].map(lambda c: "".join(sorted(c.split())))

    exact = df.merge(df, on="key", suffixes=("_a", "_b"))
    exact = exact[exact["name_a"] < exact["name_b"]]
    exact = exact.assign(core_score=100)[["name_a", "name_b", "core_score"]]

    fz = candidate_pairs(df["core"].unique(), min_score=min_score)
    fz = fz.rename(columns={"name_a": "core_a", "name_b": "core_b"})
    side = df[["name", "core", "key"]]
    fz = (fz.merge(side.rename(columns={"name": "na", "core": "core_a", "key": "key_a"}), on="core_a")
            .merge(side.rename(columns={"name": "nb", "core": "core_b", "key": "key_b"}), on="core_b"))
    fz = fz[fz["key_a"] != fz["key_b"]]  # identical keys are already in exact
    a, b = fz["na"].to_numpy(), fz["nb"].to_numpy()
    fz = pd.DataFrame({"name_a": np.where(a < b, a, b), "name_b": np.where(a < b, b, a),
                       "core_score": np.minimum(fz["score"].to_numpy(), 99)})

    out = pd.concat([exact, fz], ignore_index=True).drop_duplicates(["name_a", "name_b"])
    out["orig_score"] = [int(round(fuzz.token_sort_ratio(x, y))) for x, y in zip(out["name_a"], out["name_b"])]
    out["stratum"] = pd.cut(out["core_score"], [85, 90, 95, 100, 101], right=False,
                            labels=["85-90", "90-95", "95-100", "exact core"]).astype(str)
    return out.reset_index(drop=True)


def fresh_sample(pairs, volume, exclude, per_stratum=15, seed=1):
    """Fresh labelling sample by core-score stratum, skipping pairs already labelled."""
    seen = {frozenset(p) for p in zip(exclude["name_a"], exclude["name_b"])}
    p = pairs[[frozenset(x) not in seen for x in zip(pairs["name_a"], pairs["name_b"])]].copy()
    p["awards_a"] = p["name_a"].map(volume).fillna(0).astype(int)
    p["awards_b"] = p["name_b"].map(volume).fillna(0).astype(int)
    sizes = p["stratum"].value_counts().rename("pairs_in_stratum")
    parts = [g.sample(min(per_stratum, len(g)), random_state=seed) for _, g in p.groupby("stratum")]
    sample = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    sample.insert(0, "pair_id", range(1, len(sample) + 1))
    sample["label"] = ""
    cols = ["pair_id", "stratum", "core_score", "orig_score", "name_a", "name_b", "awards_a", "awards_b", "label"]
    return sample[cols], sizes


# ---- supplier resolver: option A frozen (identical cores or core score >= 95) ----

def _core_key(core):
    return "".join(sorted(core.split()))


class SupplierResolver:
    """Supplier identities built from training names only.

    fit: link training names with identical cores or core score >= 95, take
    connected components, and name each cluster after its most frequent name.
    transform: a known name gets its cluster. An unseen name gets the training
    cluster with the same core, else the best training core scoring >= 95,
    else an identity from its own core. Unseen names are never fuzzy-matched
    to each other, so later data cannot change or create training identities.
    """

    def __init__(self, threshold=95):
        self.threshold = threshold

    def fit(self, names):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        vol = pd.Series(names).dropna().astype(str).value_counts()
        vol = (vol.rename_axis("name").reset_index(name="awards")
                  .sort_values(["awards", "name"], ascending=[False, True], ignore_index=True))
        ids = pd.Index(vol["name"])

        pairs = core_pairs(ids, min_score=self.threshold)
        pairs = pairs[pairs["core_score"] >= self.threshold]
        a, b = ids.get_indexer(pairs["name_a"]), ids.get_indexer(pairs["name_b"])
        graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(ids), len(ids)))
        self.n_clusters_, comp = connected_components(graph, directed=False)

        # names are sorted by volume, so the first name in a cluster is the most frequent
        first = pd.Series(np.arange(len(ids))).groupby(comp).transform("min").to_numpy()
        vol["id"] = ids[first]
        vol["core"] = vol["name"].map(core_name)
        vol["cluster_size"] = vol.groupby("id")["name"].transform("size")
        self.clusters_ = vol
        self.name_map_ = dict(zip(vol["name"], vol["id"]))

        ref = vol[vol["core"] != ""].drop_duplicates("core")
        self.key_map_ = dict(zip(ref["core"].map(_core_key), ref["id"]))
        self.ref_sorted_ = [" ".join(sorted(c.split())) for c in ref["core"]]
        self.ref_ids_ = ref["id"].to_numpy()
        self._memo = {}
        return self

    def _resolve_unseen(self, todo):
        res, query = {}, []
        for n in todo:
            c = core_name(n)
            if c == "":
                res[n] = n
            elif _core_key(c) in self.key_map_:
                res[n] = self.key_map_[_core_key(c)]
            else:
                query.append((n, c))
        for start in range(0, len(query), 500):
            chunk = query[start:start + 500]
            q = [" ".join(sorted(c.split())) for _, c in chunk]
            m = process.cdist(q, self.ref_sorted_, scorer=fuzz.ratio, dtype=np.uint8,
                              score_cutoff=self.threshold, workers=-1)
            best = m.argmax(axis=1)  # ties go to the more frequent training core
            hit = m[np.arange(len(chunk)), best] > 0
            for (n, c), j, h in zip(chunk, best, hit):
                res[n] = self.ref_ids_[j] if h else "~new:" + c  # cleaned names never contain ~ or :
        return res

    def transform(self, names):
        s = pd.Series(names).astype(object)
        out = s.map(self.name_map_)
        todo = [n for n in pd.unique(s[out.isna() & s.notna()]) if n not in self._memo]
        if todo:
            self._memo.update(self._resolve_unseen(todo))
        out = out.where(out.notna(), s.map(self._memo))
        return out.where(s.notna())
