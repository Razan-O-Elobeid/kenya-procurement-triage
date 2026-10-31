"""Statistical diagnostics used to justify preprocessing choices."""
import numpy as np
import pandas as pd
from scipy import stats


def little_mcar(X, max_iter=200, tol=1e-8):
    """Little's MCAR test on a numeric block.
    Means and covariance come from EM under a multivariate normal model.
    Returns (chi2 statistic, degrees of freedom, p-value, number of patterns)."""
    X = np.asarray(X, dtype=float)
    X = X[~np.isnan(X).all(axis=1)]
    n, p = X.shape
    patterns, inv = np.unique(np.isnan(X), axis=0, return_inverse=True)
    inv = np.ravel(inv)

    mu = np.nanmean(X, axis=0)
    S = np.diag(np.nanvar(X, axis=0))
    for _ in range(max_iter):
        sum_x, sum_xx = np.zeros(p), np.zeros((p, p))
        for k, miss in enumerate(patterns):
            rows, obs = X[inv == k], ~miss
            filled = rows.copy()
            C = np.zeros((p, p))
            if miss.any():
                B = S[np.ix_(miss, obs)] @ np.linalg.pinv(S[np.ix_(obs, obs)])
                filled[:, miss] = mu[miss] + (rows[:, obs] - mu[obs]) @ B.T
                C[np.ix_(miss, miss)] = S[np.ix_(miss, miss)] - B @ S[np.ix_(obs, miss)]
            sum_x += filled.sum(axis=0)
            sum_xx += filled.T @ filled + len(rows) * C
        mu_new = sum_x / n
        S_new = sum_xx / n - np.outer(mu_new, mu_new)
        change = np.max(np.abs(mu_new - mu) / (np.abs(mu) + 1e-12))
        mu, S = mu_new, S_new
        if change < tol:
            break

    d2, df = 0.0, -p
    for k, miss in enumerate(patterns):
        obs = ~miss
        diff = X[inv == k][:, obs].mean(axis=0) - mu[obs]
        d2 += (inv == k).sum() * diff @ np.linalg.pinv(S[np.ix_(obs, obs)]) @ diff
        df += obs.sum()
    return d2, df, stats.chi2.sf(d2, df), len(patterns)


def cramers_v(table):
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    return np.sqrt(chi2 / (n * (min(table.shape) - 1)))


def missingness_tests(frame, columns, group, group_names=None):
    """Chi-square test of each column's missingness against a grouping column.
    A small p rejects MCAR; Cramer's V gives the strength of the link."""
    rows = []
    for col in columns:
        miss = frame[col].isna()
        if miss.nunique() < 2:
            continue
        table = pd.crosstab(miss, group)
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rates = (miss.groupby(group).mean() * 100).round(2)
        if group_names:
            rates.index = rates.index.map(group_names)
        rows.append({
            "column": col,
            "missing %": round(miss.mean() * 100, 2),
            "chi2": round(chi2, 1),
            "dof": dof,
            "p": p,
            "Cramer V": round(cramers_v(table), 3),
            "missing % by group": rates.to_dict(),
            "MCAR": "rejected" if p < 0.05 else "not rejected",
        })
    return pd.DataFrame(rows).set_index("column")
