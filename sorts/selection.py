""" Selection Sort — неустойчивая, in-place, неадаптивная
   
    Сортирует массив выбором минимума

    Идея: на i-м шаге ищем минимум в суффиксе a[i..N-1] и ставим его на позицию i

    Сложность по времени: Θ(N^2):
        число сравнений ровно N(N-1)/2 и не зависит от порядка данных
    Сложность по памяти: O(1) дополнительной (in-place)
    Устойчивость: НЕТ. Обмен a[i] <-> a[min_idx] «перепрыгивает» через равные
        элементы: [2a, 2b, 1] -> [1, 2b, 2a]
    Число обменов: <= N-1 (на отсортированном массиве 0) — главное преимущество
        перед пузырьком при дорогих записях

    Счётчики:
        comparisons — сравнения элементов между собой (ровно N(N-1)/2)
        writes      — записи в ячейки: один обмен = 3 записи (tmp=a; a=b; b=tmp)
                      поэтому writes = 3 * swaps
"""

from __future__ import annotations

from typing import Any

from ._types import KeyFunc, SortResult


def selection_sort(arr: list[Any], key: KeyFunc | None = None) -> SortResult:
    if key is not None:
        return _selection_sort_keyed(arr, key)

    n = len(arr)
    comparisons = 0
    writes = 0
    for i in range(n - 1):
        min_idx = i
        min_val = arr[i]                 # кэшируем значение: 1 индексация вместо 2
        for j in range(i + 1, n):
            comparisons += 1
            v = arr[j]
            if v < min_val:
                min_val = v
                min_idx = j
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            writes += 3
    return arr, comparisons, writes


def _selection_sort_keyed(arr: list[Any], key: KeyFunc) -> SortResult:
    """Ветка если есть функция ключа. Время Θ(N^2), память O(N) на массив ключей."""
    keys = [key(x) for x in arr]
    n = len(arr)
    comparisons = 0
    writes = 0
    for i in range(n - 1):
        min_idx, min_val = i, keys[i]
        for j in range(i + 1, n):
            comparisons += 1
            if keys[j] < min_val:
                min_val, min_idx = keys[j], j
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            keys[i], keys[min_idx] = keys[min_idx], keys[i]
            writes += 3
    return arr, comparisons, writes