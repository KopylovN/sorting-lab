"""Полный воспроизводимый прогон из терминала: CSV + PNG для отчёта

Запуск:  python -m bench.cli --profile full --repeats 7 --memory
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless-режим, до импорта pyplot

import pandas as pd

from sorts.registry import REGISTRY
from viz import plots

from .data import DISTRIBUTIONS
from .fitting import slope_table
from .memory import measure_memory_grid
from .runner import BenchConfig, aggregate, estimate_seconds, run_benchmark


def main() -> None:
    """Точка входа CLI"""
    p = argparse.ArgumentParser(description="Бенчмарк сортировок")
    p.add_argument("--profile", choices=["quick", "full"], default="quick")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--radix-base", type=int, default=256)
    p.add_argument("--algorithms", nargs="+", default=list(REGISTRY))
    p.add_argument("--distributions", nargs="+",
                   default=["random", "sorted", "reversed", "nearly", "duplicates"])
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--memory", action="store_true", help="дополнительно замерить tracemalloc")
    args = p.parse_args()

    cfg = BenchConfig(algorithms=args.algorithms, distributions=args.distributions,
                      repeats=args.repeats, base_seed=args.seed,
                      radix_base=args.radix_base, profile=args.profile)
    print(f"Оценка длительности: ~{estimate_seconds(cfg) / 60:.1f} мин")

    raw_dir, fig_dir = args.out / "raw", args.out / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = run_benchmark(cfg, lambda f, m: print(f"\r[{f * 100:5.1f}%] {m:<70}", end=""))
    print()
    df.to_csv(raw_dir / f"raw_{args.profile}.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(raw_dir / f"agg_{args.profile}.csv", index=False)

    st = slope_table(agg)
    st.to_csv(raw_dir / f"slopes_{args.profile}.csv", index=False)
    print("\nНаклоны log-log:\n", st.to_string(index=False))

    for dist in cfg.distributions:
        plots.fig_time_linear(agg, dist).savefig(fig_dir / f"linear_{dist}.png", dpi=150)
        plots.fig_time_loglog(agg, dist).savefig(fig_dir / f"loglog_{dist}.png", dpi=150)
        plots.fig_counters(agg, dist, "comparisons_mean").savefig(
            fig_dir / f"comparisons_{dist}.png", dpi=150)
    for aid in cfg.algorithms:
        model = "N²" if REGISTRY[aid].family == "quadratic" else "N"
        plots.fig_fit(agg, aid, "random", model).savefig(
            fig_dir / f"fit_{aid}_random.png", dpi=150)

    if args.memory:
        sizes = {a: [n for n in REGISTRY[a].sizes_quick if n <= 100000] for a in cfg.algorithms}
        mem = measure_memory_grid(cfg.algorithms, ["random"], sizes, seed=args.seed,
                                  radix_base=args.radix_base)
        mem.to_csv(raw_dir / f"memory_{args.profile}.csv", index=False)
        plots.fig_memory(mem).savefig(fig_dir / "memory.png", dpi=150)

    print(f"\nГотово. Сырые данные: {raw_dir}, графики: {fig_dir}")


if __name__ == "__main__":
    main()