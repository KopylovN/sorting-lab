"""Валидация результата без использования sorted()"""
from __future__ import annotations

from collections import Counter
from typing import Any


def is_sorted(a: list[Any]) -> bool:
    """Проверка неубывания за O(N)."""
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))


def same_multiset(a: list[Any], b: list[Any]) -> bool:
    """Проверка, что состав элементов сохранён"""
    return Counter(a) == Counter(b)


def correct(source: list[Any], result: list[Any]) -> bool:
    """Полная проверка корректности сортировки"""
    return is_sorted(result) and same_multiset(source, result)