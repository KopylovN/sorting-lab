"""Профилирование пиковой дополнительной памяти через tracemalloc"""
from __future__ import annotations

import gc
import tracemalloc
from typing import Callable

import pandas as pd

from sorts.registry import REGISTRY

from .data import generate


def measure_peak(func: Callable, data: list[int], params: dict, repeats: int = 3) -> dict:
    """Пиковая память одного вызова сортировки, усреднённая по повторямеым запускам

    Копия входа создаётся до tracemalloc.start() — иначе в пик попадёт сам массив

    Время не измеряем из-за замедления 
    """
    peaks: list[int] = []
    for _ in range(repeats):
        data_copy = data.copy()
        gc.collect()
        tracemalloc.start()
        func(data_copy, **params)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
    mean_peak = sum(peaks) / len(peaks)
    return {"peak_kb_mean": mean_peak / 1024, "peak_kb_max": max(peaks) / 1024}


def theoretical_extra_bytes(algo_id: str, n: int, base: int = 256) -> int:
    """Ожидаемая доп. память: O(1) для in-place, ~2 списка указателей для radix"""
    spec = REGISTRY[algo_id]
    if spec.inplace:
        return 0
    return 2 * n * 8 + base * 8       # список-приёмник + список pos + счётчики


def measure_memory_grid(
    algorithms: list[str], distributions: list[str], sizes: dict[str, list[int]],
    seed: int = 42, repeats: int = 3, radix_base: int = 256,
    progress_cb: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """Сетка замеров памяти: алгоритм × распределение × N."""
    tasks = [(a, d, n) for a in algorithms for d in distributions for n in sizes.get(a, [])]
    rows: list[dict] = []
    for i, (aid, dist, n) in enumerate(tasks, start=1):
        spec = REGISTRY[aid]
        params = {"base": radix_base} if aid == "radix" else {}
        data = generate(n, dist, seed)
        res = measure_peak(spec.func, data, params, repeats)
        rows.append({"algorithm": aid, "algorithm_name": spec.name, "distribution": dist,
                     "n": n, "inplace": spec.inplace, "memory_theory": spec.memory,
                     "theory_kb": theoretical_extra_bytes(aid, n, radix_base) / 1024, **res})
        if progress_cb:
            progress_cb(i / max(1, len(tasks)), f"Память: {spec.name} · {dist} · N={n}")
    return pd.DataFrame(rows)