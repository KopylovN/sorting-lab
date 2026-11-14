"""Интерактивные plotly-графики для Streamlit-дашборда"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sorts.registry import REGISTRY

from bench.fitting import MODELS, fit_model, loglog_slope

COLORS = {aid: s.color for aid, s in REGISTRY.items()}
NAMES = {aid: s.name for aid, s in REGISTRY.items()}
ROLE_COLOR = {"default": "#9ecae1", "compare": "#fd8d3c", "write": "#e6550d",
              "sorted": "#31a354", "pivot": "#756bb1", "min": "#de2d26"}


def time_chart(agg: pd.DataFrame, distribution: str, loglog: bool = False) -> go.Figure:
    """T(N) с усами погрешности ±σ; зум и тултипы из коробки"""
    d = agg[agg["distribution"] == distribution]
    fig = go.Figure()
    for aid, g in d.groupby("algorithm"):
        g = g.sort_values("n")
        fig.add_trace(go.Scatter(
            x=g["n"], y=g["time_ms_mean"], mode="lines+markers", name=NAMES[aid],
            line=dict(color=COLORS[aid]),
            error_y=dict(type="data", array=g["time_ms_std"], visible=True),
            hovertemplate="N=%{x}<br>T=%{y:.3f} мс<extra>" + NAMES[aid] + "</extra>"))
    fig.update_layout(
        title=f"Время сортировки — {distribution}",
        xaxis_title="Размер массива, N (элементов)", yaxis_title="Время выполнения, T (мс)",
        xaxis_type="log" if loglog else "linear", yaxis_type="log" if loglog else "linear",
        hovermode="x unified", height=480)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    return fig


def facet_chart(agg: pd.DataFrame, loglog: bool = False) -> go.Figure:
    """Сравнение всех распределений сразу"""
    fig = px.line(agg, x="n", y="time_ms_mean", color="algorithm",
                  facet_col="distribution", facet_col_wrap=3, markers=True,
                  error_y="time_ms_std", log_x=loglog, log_y=loglog,
                  color_discrete_map=COLORS, labels={"n": "N, элементов",
                  "time_ms_mean": "T, мс", "algorithm": "Алгоритм"}, height=620)
    fig.for_each_xaxis(lambda a: a.update(title_text="N, элементов", showgrid=True))
    fig.for_each_yaxis(lambda a: a.update(title_text="T, мс", showgrid=True))
    fig.update_layout(title="T(N) по всем типам входных данных")
    return fig


def fit_chart(agg: pd.DataFrame, algorithm: str, distribution: str, model: str) -> tuple[go.Figure, float, float, float]:
    """Эксперимент + теоретическая кривая. Возвращает (fig, c, R², k)"""
    g = agg[(agg["algorithm"] == algorithm) & (agg["distribution"] == distribution)].sort_values("n")
    n, t = g["n"].to_numpy(float), g["time_ms_mean"].to_numpy(float)
    c, r2 = fit_model(n, t, model)
    k, _, _ = loglog_slope(n, t)
    grid = np.linspace(n.min(), n.max(), 400)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n, y=t, mode="markers", name="эксперимент (μ ± σ)",
                             error_y=dict(type="data", array=g["time_ms_std"]),
                             marker=dict(color=COLORS[algorithm], size=9)))
    fig.add_trace(go.Scatter(x=grid, y=MODELS[model](grid, c), mode="lines",
                             name=f"МНК: T = {c:.3e}·{model}", line=dict(dash="dash", color="black")))
    fig.update_layout(title=f"{NAMES[algorithm]} — {distribution}: R²={r2:.4f}, наклон log-log k≈{k:.2f}",
                      xaxis_title="N, элементов", yaxis_title="T, мс", height=480)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig, c, r2, k


def memory_chart(mem: pd.DataFrame) -> go.Figure:
    """Пиковая дополнительная память vs N"""
    fig = px.line(mem, x="n", y="peak_kb_mean", color="algorithm", markers=True,
                  log_x=True, log_y=True, color_discrete_map=COLORS,
                  labels={"n": "N, элементов", "peak_kb_mean": "Пиковая доп. память, КБ",
                          "algorithm": "Алгоритм"}, height=460)
    fig.update_layout(title="tracemalloc: пиковая дополнительная память")
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def counters_chart(agg: pd.DataFrame, distribution: str, metric: str) -> go.Figure:
    """Счётчики элементарных операций vs N"""
    d = agg[agg["distribution"] == distribution]
    fig = px.line(d, x="n", y=metric, color="algorithm", markers=True,
                  log_x=True, log_y=True, color_discrete_map=COLORS,
                  labels={"n": "N, элементов", metric: "Количество операций, шт."}, height=440)
    fig.update_layout(title="Элементарные операции (детерминированы, без шума таймера)")
    return fig


def frame_chart(frame: dict, title: str) -> go.Figure:
    """Один кадр анимации: столбцы = элементы, цвет = роль в текущем шаге"""
    a = frame["array"]
    colors = [ROLE_COLOR.get(frame["roles"].get(i, "default"), ROLE_COLOR["default"])
              for i in range(len(a))]
    fig = go.Figure(go.Bar(x=list(range(len(a))), y=a, marker_color=colors,
                           text=a, textposition="outside",
                           hovertemplate="index=%{x}<br>value=%{y}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Индекс", yaxis_title="Значение",
                      height=380, margin=dict(t=60, b=40), showlegend=False,
                      yaxis_range=[0, max(a) * 1.15 if a else 1])
    return fig