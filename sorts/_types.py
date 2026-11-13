"""Псевдонимы"""

from __future__ import annotations

from typing import Any, Callable, TypeAlias

KeyFunc: TypeAlias = Callable[[Any], Any]

SortResult: TypeAlias = tuple[list[Any], int, int] #отсортированный_массив, comparisons, writes