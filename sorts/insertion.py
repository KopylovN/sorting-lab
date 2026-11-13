""" Insertion Sort - устойчивая, in-place, адаптивная

    Сортирует массив вставками на месте.

    Идея: префикс a[0..i-1] всегда отсортирован; очередной элемент a[i] сдвигается влево, пока не встретит меньший либо равный элемент

    Сложность по времени:
        best  O(N)      — уже отсортированный массив (N-1 сравнений, 0 сдвигов)
        avg   O(N^2)    — ~N^2/4 инверсий
        worst O(N^2)    — обратно отсортированный массив, N(N-1)/2 инверсий
    Сложность по памяти: O(1) дополнительной (in-place), рекурсии нет
    Устойчивость: да, т.к. строгое сравнение не переставляет равные элементы

    Счётчики:
        comparisons — сравнения элементов (ключей) между собой
        writes      — записи в ячейки массива: каждый сдвиг + финальная вставка

    Аргументы:
        arr: список чисел (или объектов при заданном key). Мутируется на месте.
        key: необязательная функция ключа (аналог key в sorted, но реализован вручную).

    Выход:
        тот же самый объект arr, comparisons, writes
"""
from __future__ import annotations

from typing import Any

from ._types import KeyFunc, SortResult


def insertion_sort(arr: list[Any], key: KeyFunc | None = None) -> SortResult:
    if key is not None:
        return _insertion_sort_keyed(arr, key)

    comparisons = 0
    writes = 0
    for i in range(1, len(arr)):
        cur = arr[i]
        j = i - 1
        while j >= 0:
            comparisons += 1
            if arr[j] > cur:
                arr[j + 1] = arr[j]      # сдвиг вправо
                writes += 1
                j -= 1
            else:
                break                    # место найдено
        if j + 1 != i:                   # смотрим, был ли сдвиг в целом
            arr[j + 1] = cur
            writes += 1
    return arr, comparisons, writes


def _insertion_sort_keyed(arr: list[Any], key: KeyFunc) -> SortResult:
    """Ветка если есть функция ключа: ключи вычисляются один раз и двигаются параллельно.

    Время O(N^2) + N вызовов key, память O(N) на массив ключей.
    writes считаем только записи в сам arr (служебный массив ключей не в счёт).
    """
    keys = [key(x) for x in arr]
    comparisons = 0
    writes = 0
    for i in range(1, len(arr)):
        cur, cur_key = arr[i], keys[i]
        j = i - 1
        while j >= 0:
            comparisons += 1
            if keys[j] > cur_key:
                arr[j + 1], keys[j + 1] = arr[j], keys[j]
                writes += 1
                j -= 1
            else:
                break
        if j + 1 != i:
            arr[j + 1], keys[j + 1] = cur, cur_key
            writes += 1
    return arr, comparisons, writes
