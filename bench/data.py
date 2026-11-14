"""Генераторы входных распределений"""
from __future__ import annotations

import random

RANDOM_UPPER = 1_000_000          # фиксированный диапазон: d у radix постоянен
DUP_VALUES = (1, 2, 3, 4, 5)      # 5 уникальных значений

DISTRIBUTIONS: dict[str, str] = {
    "random": "1. Случайный (Uniform Random)",
    "sorted": "2. Отсортированный (Sorted)",
    "reversed": "3. Обратно отсортированный (Reversed)",
    "nearly": "4. Почти отсортированный (5% перемешано)",
    "duplicates": "5. Много дубликатов (3–5 значений)",
    "nearly_local": "6. Почти отсортированный (локальные обмены)",
}


def generate(n: int, kind: str, seed: int) -> list[int]:
    """Строит массив длины n заданного распределения на конкретном seed.

    Время O(N), память O(N). Вызывается ВНЕ измеряемого участка.
    """
    rng = random.Random(seed)
    if n <= 0:
        return []

    if kind == "random":
        return [rng.randrange(RANDOM_UPPER) for _ in range(n)]
    if kind == "sorted":
        return list(range(1, n + 1))
    if kind == "reversed":
        return list(range(n, 0, -1))
    if kind == "nearly":
        a = list(range(1, n + 1))
        k = max(1, n // 20)
        idxs = rng.sample(range(n), min(k, n))
        vals = [a[i] for i in idxs]
        rng.shuffle(vals)
        for i, v in zip(idxs, vals):
            a[i] = v
        return a
    if kind == "nearly_local":
        a = list(range(1, n + 1))
        for _ in range(max(1, n // 40)):
            i = rng.randrange(n - 1) if n > 1 else 0
            a[i], a[i + 1] = a[i + 1], a[i]
        return a
    if kind == "duplicates":
        return rng.choices(DUP_VALUES, k=n)
    raise ValueError(f"Неизвестное распределение: {kind}")