"""Покадровые трассы сортировок для аниматора (N = 20–30)

Логика повторяет ядро 1:1, но щедро йелдит состояние после каждой операции.
В бенчмарках эти версии не участвуют
"""
from __future__ import annotations

from typing import Any, Iterator

Frame = dict[str, Any]


def _frame(a: list[int], roles: dict[int, str], c: int, w: int, msg: str,
           buckets: list[list[int]] | None = None, digit: int | None = None) -> Frame:
    """Снимок состояния"""
    return {"array": a.copy(), "roles": dict(roles), "comparisons": c,
            "writes": w, "message": msg, "buckets": buckets, "digit": digit}


def insertion_frames(arr: list[int]) -> Iterator[Frame]:
    """Трасса сортировки вставками"""
    a = arr.copy()
    c = w = 0
    n = len(a)
    yield _frame(a, {0: "sorted"}, c, w, "Префикс из одного элемента уже отсортирован")
    for i in range(1, n):
        cur, j = a[i], i - 1
        yield _frame(a, {i: "pivot"}, c, w, f"Берём ключ a[{i}] = {cur}")
        while j >= 0:
            c += 1
            yield _frame(a, {i: "pivot", j: "compare", j + 1: "compare"}, c, w,
                         f"Сравнение #{c}: a[{j}]={a[j]} > ключ {cur}?")
            if a[j] > cur:
                a[j + 1] = a[j]
                w += 1
                yield _frame(a, {j + 1: "write", i: "pivot"}, c, w, f"Сдвиг a[{j}] → a[{j+1}]")
                j -= 1
            else:
                break
        if j + 1 != i:
            a[j + 1] = cur
            w += 1
            yield _frame(a, {j + 1: "write"}, c, w, f"Вставка ключа {cur} в позицию {j+1}")
        yield _frame(a, {k: "sorted" for k in range(i + 1)}, c, w,
                     f"Префикс a[0..{i}] отсортирован")
    yield _frame(a, {k: "sorted" for k in range(n)}, c, w, f"Готово: {c} сравнений, {w} записей")


def selection_frames(arr: list[int]) -> Iterator[Frame]:
    """Трасса сортировки выбором"""
    a = arr.copy()
    c = w = 0
    n = len(a)
    for i in range(n - 1):
        m, mv = i, a[i]
        yield _frame(a, {i: "pivot", m: "min"}, c, w, f"Проход {i}: ищем минимум в a[{i}..{n-1}]")
        for j in range(i + 1, n):
            c += 1
            yield _frame(a, {i: "pivot", m: "min", j: "compare"}, c, w,
                         f"Сравнение #{c}: a[{j}]={a[j]} < текущий минимум {mv}?")
            if a[j] < mv:
                mv, m = a[j], j
                yield _frame(a, {i: "pivot", m: "min"}, c, w, f"Новый минимум {mv} на позиции {m}")
        if m != i:
            a[i], a[m] = a[m], a[i]
            w += 3
            yield _frame(a, {i: "write", m: "write"}, c, w, f"Обмен a[{i}] ↔ a[{m}] (3 записи)")
        yield _frame(a, {k: "sorted" for k in range(i + 1)}, c, w, f"Позиция {i} зафиксирована")
    yield _frame(a, {k: "sorted" for k in range(n)}, c, w, f"Готово: {c} сравнений, {w} записей")


def radix_frames(arr: list[int], base: int = 10) -> Iterator[Frame]:
    """Трасса LSD Radix в основании 10 (для наглядности корзин), только неотрицательные"""
    a = arr.copy()
    c = w = 0
    if any(v < 0 for v in a):
        raise ValueError("Аниматор radix работает с неотрицательными числами")
    digits = max(1, len(str(max(a) if a else 0)))
    for p in range(digits):
        buckets: list[list[int]] = [[] for _ in range(base)]
        yield _frame(a, {}, c, w, f"Проход {p+1}/{digits}: разряд 10^{p}",
                     [b.copy() for b in buckets], p)
        for i, v in enumerate(a):
            d = (v // 10 ** p) % base
            buckets[d].append(v)
            w += 1
            yield _frame(a, {i: "compare"}, c, w, f"{v}: разряд = {d} → корзина {d}",
                         [b.copy() for b in buckets], p)
        new: list[int] = []
        for b in buckets:
            new.extend(b)
        a = new
        yield _frame(a, {k: "write" for k in range(len(a))}, c, w,
                     "Сбор корзин слева направо — порядок равных сохранён (устойчивость)",
                     [b.copy() for b in buckets], p)
    yield _frame(a, {k: "sorted" for k in range(len(a))}, c, w,
                 f"Готово: {c} сравнений элементов, {w} записей", None, None)


def build_frames(algorithm: str, data: list[int]) -> list[Frame]:
    """Фабрика кадров для выбранного алгоритма"""
    if algorithm == "insertion":
        return list(insertion_frames(data))
    if algorithm == "selection":
        return list(selection_frames(data))
    if algorithm == "radix":
        return list(radix_frames(data, base=10))
    raise ValueError(algorithm)