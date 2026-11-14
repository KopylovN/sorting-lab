"""Реестр алгоритмов варианта: метаданные, сетки размеров, цвета графиков"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .insertion import insertion_sort
from .radix import radix_sort
from .selection import selection_sort
from ._types import SortResult

# Сетки требуемых размеров
QUADRATIC_SIZES: tuple[int, ...] = (100, 300, 600, 1000, 2000, 4000, 8000, 10000)
LINEAR_SIZES: tuple[int, ...] = (1000, 5000, 10000, 50000, 100000, 300000, 500000, 1000000)
# Укороченные сетки для интерактивного режима
QUICK_QUADRATIC: tuple[int, ...] = (100, 300, 600, 1000, 2000)
QUICK_LINEAR: tuple[int, ...] = (1000, 5000, 10000, 50000, 100000)


@dataclass(frozen=True)
class AlgoSpec:
    """Паспорт алгоритма для таблиц, графиков и ограничений бенчмарка"""

    id: str
    name: str
    func: Callable[..., SortResult]
    family: str                 # "quadratic" | "linear"
    time_best: str
    time_avg: str
    time_worst: str
    memory: str
    inplace: bool
    stable: bool
    comparison_based: bool
    domain: str
    sizes_full: tuple[int, ...]
    sizes_quick: tuple[int, ...]
    color: str


REGISTRY: dict[str, AlgoSpec] = {
    "insertion": AlgoSpec(
        id="insertion", name="Insertion Sort", func=insertion_sort,
        family="quadratic", time_best="O(N)", time_avg="O(N²)", time_worst="O(N²)",
        memory="O(1)", inplace=True, stable=True, comparison_based=True, domain="int/float",
        sizes_full=QUADRATIC_SIZES, sizes_quick=QUICK_QUADRATIC, color="#1f77b4",
    ),
    "selection": AlgoSpec(
        id="selection", name="Selection Sort", func=selection_sort,
        family="quadratic", time_best="Ω(N²)", time_avg="Θ(N²)", time_worst="O(N²)",
        memory="O(1)", inplace=True, stable=False, comparison_based=True, domain="int/float",
        sizes_full=QUADRATIC_SIZES, sizes_quick=QUICK_QUADRATIC, color="#d62728",
    ),
    "radix": AlgoSpec(
        id="radix", name="LSD Radix Sort", func=radix_sort,
        family="linear", time_best="O(d·N)", time_avg="O(d·(N+base))", time_worst="O(d·(N+base))",
        memory="O(N + base)", inplace=False, stable=True, comparison_based=False, domain="int",
        sizes_full=LINEAR_SIZES, sizes_quick=QUICK_LINEAR, color="#2ca02c",
    ),
}

FAMILY_RU = {"quadratic": "O(N²)", "linear": "O(N) / O(d·N)"}