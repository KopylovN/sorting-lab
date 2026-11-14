"""Статические графики matplotlib"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sorts.registry import REGISTRY

from bench.fitting import MODELS, fit_model, loglog_slope

X_LABEL = "Размер массива, N (элементов)"
Y_LABEL = "Время выполнения, T (мс)"


def _style(ax, title: str, xlabel: str = X_LABEL, ylabel: str = Y_LABEL) -> None:
    """Единое оформление: сетка, легенда, подписи осей с размерностями"""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="best", framealpha=0.9)


def _subset(agg: pd.DataFrame, distribution: str) -> pd.DataFrame:
    """Точки одного распределения"""
    return agg[agg["distribution"] == distribution]


def fig_time_linear(agg: pd.DataFrame, distribution: str):
    """График 1: T(N) в линейном масштабе с полосами погрешности ±σ"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for aid, g in _subset(agg, distribution).groupby("algorithm"):
        g = g.sort_values("n")
        spec = REGISTRY[aid]
        ax.errorbar(g["n"], g["time_ms_mean"], yerr=g["time_ms_std"],
                    marker="o", capsize=3, label=spec.name, color=spec.color)
    _style(ax, f"T(N), линейный масштаб — {distribution}")
    fig.tight_layout()
    return fig


def fig_time_loglog(agg: pd.DataFrame, distribution: str):
    """График 2: log-log; наклон прямой = степень k в T = c·N^k"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for aid, g in _subset(agg, distribution).groupby("algorithm"):
        g = g.sort_values("n")
        spec = REGISTRY[aid]
        k, _, _ = loglog_slope(g["n"].to_numpy(), g["time_ms_mean"].to_numpy())
        ax.errorbar(g["n"], g["time_ms_mean"], yerr=g["time_ms_std"], marker="o",
                    capsize=3, color=spec.color, label=f"{spec.name} · k≈{k:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, f"T(N), log-log — {distribution}")
    fig.tight_layout()
    return fig


def fig_fit(agg: pd.DataFrame, algorithm: str, distribution: str, model_name: str):
    """График 3: экспериментальные точки + теоретическая кривая с подобранным c"""
    g = _subset(agg, distribution)
    g = g[g["algorithm"] == algorithm].sort_values("n")
    fig, ax = plt.subplots(figsize=(8, 5))
    if g.empty:
        _style(ax, "нет данных")
        return fig
    n, t = g["n"].to_numpy(float), g["time_ms_mean"].to_numpy(float)
    c, r2 = fit_model(n, t, model_name)
    k, _, _ = loglog_slope(n, t)
    spec = REGISTRY[algorithm]
    ax.errorbar(n, t, yerr=g["time_ms_std"], fmt="o", capsize=3,
                color=spec.color, label="эксперимент (μ ± σ)")
    grid = np.linspace(n.min(), n.max(), 400)
    ax.plot(grid, MODELS[model_name](grid, c), "--", color="black",
            label=f"fit: T = {c:.3e}·{model_name},  R²={r2:.4f}")
    _style(ax, f"{spec.name} — {distribution}\nнаклон log-log k≈{k:.2f}")
    fig.tight_layout()
    return fig


def fig_memory(mem: pd.DataFrame):
    """График 4 (бонус): пиковая дополнительная память vs N"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for aid, g in mem.groupby("algorithm"):
        g = g.sort_values("n")
        spec = REGISTRY[aid]
        ax.plot(g["n"], g["peak_kb_mean"], marker="s", color=spec.color,
                label=f"{spec.name} ({spec.memory})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "Пиковая дополнительная память (tracemalloc)", Y_LABEL := "Память, КБ")
    fig.tight_layout()
    return fig


def fig_counters(agg: pd.DataFrame, distribution: str, metric: str = "comparisons_mean"):
    """Бонус: число элементарных операций vs N (без шума таймера)"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for aid, g in _subset(agg, distribution).groupby("algorithm"):
        g = g.sort_values("n")
        spec = REGISTRY[aid]
        ax.plot(g["n"], g[metric], marker="o", color=spec.color, label=spec.name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    title = "Сравнения элементов" if metric.startswith("comparisons") else "Записи в ячейки"
    _style(ax, f"{title} — {distribution}", ylabel="Количество операций, шт.")
    fig.tight_layout()
    return fig