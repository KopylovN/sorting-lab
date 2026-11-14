"""Профилирование пиковой дополнительной памяти через tracemalloc"""
from __future__ import annotations

import gc
import tracemalloc
from typing import Callable

import pandas as pd

from sorts.registry import REGISTRY

from .data import generate


def theoretical_extra_bytes(algo_id: str, n: int, base: int = 256) -> int:
    """Ожидаемая доп. память: O(1) для in-place, ~2 списка указателей для radix"""
    spec = REGISTRY[algo_id]
    if spec.inplace:
        return 0
    return 2 * n * 8 + base * 8       # список-приёмник + список pos + счётчики


def measure_peak(func: Callable, data: list[int], params: dict, repeats: int = 1) -> dict:
    """Пиковая память - один прогон, без gc.collect внутри цикла"""
    data_copy = data.copy()
    tracemalloc.start()
    func(data_copy, **params)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"peak_kb_mean": peak / 1024, "peak_kb_max": peak / 1024}


def measure_memory_grid(
    algorithms: list[str], distributions: list[str], sizes: dict[str, list[int]],
    seed: int = 42, repeats: int = 1, radix_base: int = 256,
    progress_cb: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """Сетка замеров памяти, gc.collect() вызывается ОДИН раз в начале"""
    import gc
    gc.collect()                # один раз, не внутри цикла
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