"""Unchanged statistical function blocks from the finalized edge-test producer."""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
MIN_PVALUE = 1e-300

def bh_qvalues(pvals: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    q[valid] = out
    return q


def floor_pvalue(pvalue: float) -> float:
    if not np.isfinite(pvalue):
        return np.nan
    return float(np.clip(pvalue, MIN_PVALUE, 1.0))


def fisher_from_matrix(mat: pd.DataFrame, row: str, col: str) -> tuple[int, int, int, int, float]:
    row = str(row)
    col = str(col)
    if row not in mat.index or col not in mat.columns:
        return 0, 0, 0, 0, np.nan
    a = int(mat.loc[row, col])
    row_total = int(mat.loc[row, :].sum())
    col_total = int(mat.loc[:, col].sum())
    grand_total = int(mat.to_numpy().sum())
    b = row_total - a
    c = col_total - a
    d = grand_total - row_total - col_total + a
    if min(a, b, c, d) < 0 or row_total <= 0 or col_total <= 0:
        return a, b, c, d, np.nan
    _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return a, b, c, d, floor_pvalue(p)

