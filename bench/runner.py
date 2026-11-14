"""Изолированные замеры времени: perf_counter, повторы, статистика"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from sorts.registry import REGISTRY

from .checks import correct
from .data import generate

# Грубые константы для оценки длительности прогона (секунды), калибруются на месте
_QUAD_COST = {"insertion": 3.2e-8, "selection": 4.5e-8}
_DIST_FACTOR = {"random": 1.0, "sorted": 0.02, "reversed": 1.8,
                "nearly": 0.10, "nearly_local": 0.02, "duplicates": 0.85}


@dataclass
class BenchConfig:
    """Конфигурация серии экспериментов"""

    algorithms: list[str]
    distributions: list[str]
    sizes: dict[str, list[int]] = field(default_factory=dict)
    repeats: int = 5
    base_seed: int = 42
    radix_base: int = 256
    validate: bool = True
    profile: str = "quick"

    def sizes_for(self, algo_id: str) -> list[int]:
        """Сетка размеров для алгоритма: явная или пресет из реестра"""
        if algo_id in self.sizes and self.sizes[algo_id]:
            return sorted(self.sizes[algo_id])
        spec = REGISTRY[algo_id]
        return list(spec.sizes_quick if self.profile == "quick" else spec.sizes_full)


def estimate_seconds(cfg: BenchConfig) -> float:
    """Очень грубая оценка длительности прогона, чтобы предупредить пользователя"""
    total = 0.0
    for aid in cfg.algorithms:
        for n in cfg.sizes_for(aid):
            for dist in cfg.distributions:
                if aid in _QUAD_COST:
                    f = 1.0 if aid == "selection" else _DIST_FACTOR.get(dist, 1.0)
                    total += _QUAD_COST[aid] * n * n * f * cfg.repeats
                else:
                    total += 4e-7 * n * 3 * cfg.repeats
    return total


def run_benchmark(
    cfg: BenchConfig, progress_cb: Callable[[float, str], None] | None = None
) -> pd.DataFrame:
    """Прогоняет все комбинации (алгоритм × распределение × N × повтор)

    Методика:
      генерация массива и .copy() — вне таймера
      время — time.perf_counter() вокруг одного вызова сортировки
      каждый повтор — новый seed; в таблицу пишутся сырые точки

    Returns: «длинный» DataFrame с сырыми замерами (одна строка = один запуск).
    """
    tasks = [(aid, dist, n)
             for aid in cfg.algorithms
             for dist in cfg.distributions
             for n in cfg.sizes_for(aid)]
    total_runs = max(1, len(tasks) * cfg.repeats)
    rows: list[dict] = []
    done = 0

    for aid, dist, n in tasks:
        spec = REGISTRY[aid]
        params = {"base": cfg.radix_base} if aid == "radix" else {}
        for rep in range(cfg.repeats):
            seed = cfg.base_seed + rep
            data = generate(n, dist, seed)       # вне замера
            data_copy = data.copy()              # вне замера, эталон не стоит трогать

            t_start = time.perf_counter()
            sorted_arr, cmp_cnt, wr_cnt = spec.func(data_copy, **params)
            t_end = time.perf_counter()

            ok = correct(data, sorted_arr) if cfg.validate else True
            rows.append({
                "algorithm": aid, "algorithm_name": spec.name, "family": spec.family,
                "distribution": dist, "n": n, "repeat": rep, "seed": seed,
                "time_s": t_end - t_start, "time_ms": (t_end - t_start) * 1000.0,
                "comparisons": cmp_cnt, "writes": wr_cnt, "correct": ok,
            })
            done += 1
            if progress_cb is not None:
                progress_cb(done / total_runs, f"{spec.name} · {dist} · N={n} · повтор {rep + 1}")

    df = pd.DataFrame(rows)
    if cfg.validate and not df["correct"].all():
        bad = df.loc[~df["correct"], ["algorithm", "distribution", "n"]].drop_duplicates()
        raise AssertionError(f"Провалена проверка корректности:\n{bad}")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Сводка по точкам: μ, σ, медиана, min/max, средние счётчики."""
    agg = (df.groupby(["algorithm", "algorithm_name", "family", "distribution", "n"],
                      as_index=False)
             .agg(time_ms_mean=("time_ms", "mean"),
                  time_ms_std=("time_ms", "std"),
                  time_ms_median=("time_ms", "median"),
                  time_ms_min=("time_ms", "min"),
                  time_ms_max=("time_ms", "max"),
                  comparisons_mean=("comparisons", "mean"),
                  writes_mean=("writes", "mean"),
                  repeats=("time_ms", "size")))
    agg["time_ms_std"] = agg["time_ms_std"].fillna(0.0)
    return agg.sort_values(["distribution", "algorithm", "n"]).reset_index(drop=True)