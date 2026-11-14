"""Теоретическая аппроксимация и наклон log-log"""
from __future__ import annotations

import numpy as np


def model_quadratic(n, c):
    """T(N) = c · N²  (O(N²))"""
    return c * np.asarray(n, dtype=float) ** 2


def model_nlogn(n, c):
    """T(N) = c · N·log2(N)  (O(N log N))"""
    n = np.asarray(n, dtype=float)
    return c * n * np.log2(np.maximum(n, 2.0))


def model_linear(n, c):
    """T(N) = c · N  (O(N))"""
    return c * np.asarray(n, dtype=float)


MODELS = {"N²": model_quadratic, "N log N": model_nlogn, "N": model_linear}


def fit_model(n, t, model_name: str) -> tuple[float, float]:
    """Подбор коэффициента c МНК (scipy.optimize.curve_fit, fallback — замкнутая формула)

    Выход: (c, R²)
    """
    n = np.asarray(n, dtype=float)
    t = np.asarray(t, dtype=float)
    f = MODELS[model_name]
    basis = f(n, 1.0)
    c_closed = float(np.sum(t * basis) / np.sum(basis ** 2))  # МНК для одного параметра
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(f, n, t, p0=[c_closed], maxfev=20000)
        c = float(popt[0])
    except Exception:
        c = c_closed
    pred = f(n, c)
    ss_res = float(np.sum((t - pred) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return c, r2


def loglog_slope(n, t, tail: float = 0.6) -> tuple[float, float, float]:
    """Наклон k прямой log T = k·log N + log c по хвосту точек

    Мелкие N отсекаются, т.к. там доминируют накладные расходы интерпретатора
    Выход: (k, log10(c), R²)
    """
    n = np.asarray(n, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = (n > 0) & (t > 0)
    n, t = n[mask], t[mask]
    if len(n) < 3:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(n)
    n, t = n[order], t[order]
    k = min(len(n), max(3, int(np.ceil(len(n) * tail))))
    n, t = n[-k:], t[-k:]
    slope, intercept = np.polyfit(np.log10(n), np.log10(t), 1)
    pred = slope * np.log10(n) + intercept
    ss_res = float(np.sum((np.log10(t) - pred) ** 2))
    ss_tot = float(np.sum((np.log10(t) - np.log10(t).mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def slope_table(agg) -> "object":
    """Таблица наклонов k по всем (алгоритм, распределение) — доказательство сложности"""
    import pandas as pd
    rows = []
    for (aid, dist), g in agg.groupby(["algorithm", "distribution"]):
        k, b, r2 = loglog_slope(g["n"].to_numpy(), g["time_ms_mean"].to_numpy())
        rows.append({"algorithm": aid, "distribution": dist, "k (наклон log-log)": k,
                     "log10(c)": b, "R²": r2, "точек": len(g)})
    return pd.DataFrame(rows).sort_values(["algorithm", "distribution"]).reset_index(drop=True)