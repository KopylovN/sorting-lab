""" LSD Radix Sort

Некомпараторный алгоритм: элементы распределяются по корзинам по значению разряда
Устойчивость каждого прохода counting sort — то, на чём держится весь алгоритм
LSD-поразрядная сортировка целых чисел (по умолчанию — по байтам, base=256)

    Сложность по времени: O(d * (N + base)), где d = число разрядов максимума
        в основании base. При фиксированном диапазоне значений d = const -> O(N)
        ВАЖНО: если значения растут вместе с N (распределения sorted/reversed)
        то d = ceil(log_base(N)) и строго говоря получается O(N log N / log base)
    Сложность по памяти: O(N + base) — out-of-place (буфер + массив счётчиков)
    Устойчивость: ДА (устойчивый counting sort на каждом разряде)
    Домен: целые числа (отрицательные поддерживаются). Для float — radix_sort_float()

    Счётчики:
        comparisons — сравнений «элемент с элементом» при распределении нет (0)
                      N-1 сравнений уходит только на поиск максимума (определение d)
                      сравнение `v < 0` — сравнение с константой, не считаем
        writes      — все записи элементов в ячейки: разбиение на neg/pos (N)
                      рассеивание по корзинам (d*N), финальная склейка (N, если есть neg)

    Аргументы:
        arr: входной список, не мутируется.
        base: основание системы счисления, степень двойки >= 2 (2, 16, 256, 65536...)
        key: функция, извлекающая целочисленный ключ

    Выход:
        новый отсортированный список, comparisons, writes
"""
from __future__ import annotations

import struct
from typing import Any, Callable

from ._types import SortResult

_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63


def radix_sort(
    arr: list[Any],
    base: int = 256,
    key: Callable[[Any], int] | None = None,
) -> SortResult:
    
    n = len(arr)
    if n < 2:
        return list(arr), 0, 0
    if base < 2 or (base & (base - 1)):
        raise ValueError("base должен быть степенью двойки >= 2")
    if key is not None:
        return _radix_sort_keyed(arr, base, key)
    if not isinstance(arr[0], int):
        raise TypeError("radix_sort работает с int")

    bits = base.bit_length() - 1
    mask = base - 1

    # Разбиение по знаку: |neg| сортируем как беззнаковые, потом разворачиваем.
    neg: list[int] = []
    pos: list[int] = []
    writes = 0
    for v in arr:
        if v < 0:
            neg.append(-v)
        else:
            pos.append(v)
        writes += 1

    pos_sorted, w, c = _lsd_nonneg(pos, bits, mask, base)
    writes += w
    comparisons = c
    if not neg:
        return pos_sorted, comparisons, writes

    neg_sorted, w, c = _lsd_nonneg(neg, bits, mask, base)
    writes += w
    comparisons += c
    out = [-v for v in reversed(neg_sorted)]
    out.extend(pos_sorted)
    writes += n
    return out, comparisons, writes


def _lsd_nonneg(
    src: list[int], bits: int, mask: int, base: int
) -> tuple[list[int], int, int]:
    """LSD для неотрицательных чисел. Возвращает (результат, writes, comparisons).

    Источник и приёмник меняются местами после каждого прохода, а значит и копирование буфера обратно не нужно.
    """
    n = len(src)
    if n < 2:
        return src, 0, 0

    max_val = max(src)                 # N-1 сравнений элементов
    comparisons = n - 1
    digits = max(1, (max_val.bit_length() + bits - 1) // bits)

    buf = [0] * n
    a, b = src, buf
    writes = 0
    for p in range(digits):
        shift = p * bits
        count = [0] * base
        for v in a:                                   # 1. гистограмма разряда
            count[(v >> shift) & mask] += 1
        total = 0
        for i in range(base):                         # 2. префиксные суммы -> старты корзин
            cnt = count[i]
            count[i] = total
            total += cnt
        for v in a:                                   # 3. устойчивое рассеивание
            d = (v >> shift) & mask
            b[count[d]] = v
            count[d] += 1
            writes += 1
        a, b = b, a
    return a, writes, comparisons


def _radix_sort_keyed(arr: list[Any], base: int, key: Callable[[Any], int]) -> SortResult:
    """Сортировка записей по целочисленному ключу с сохранением устойчивости

    Упаковываем (key, исходный_индекс) в одно целое. LSD отсортирует по ключу, а равенство ключей соотвественно разрешит индекс, следовательно, порядок равных элементов сохранится
    Время O(d*(N+base)), память O(N)
    """
    n = len(arr)
    keys = [key(x) for x in arr]
    min_key = min(keys)
    offset = -min_key if min_key < 0 else 0
    idx_bits = max(1, (n - 1).bit_length())
    idx_mask = (1 << idx_bits) - 1
    packed = [((k + offset) << idx_bits) | i for i, k in enumerate(keys)]
    sorted_packed, c, w = radix_sort(packed, base)
    out = [arr[p & idx_mask] for p in sorted_packed]
    return out, c + (n - 1), w + 2 * n


def _f2u(x: float) -> int:
    """IEEE-754 double -> монотонный беззнаковый ключ (порядок бит = порядок чисел)"""
    b = struct.unpack("<Q", struct.pack("<d", x))[0]
    return b ^ (_MASK64 if b & _SIGN64 else _SIGN64)


def _u2f(u: int) -> float:
    """Обратное преобразование к _f2u"""
    b = u ^ (_SIGN64 if u & _SIGN64 else _MASK64)
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def radix_sort_float(arr: list[float], base: int = 256) -> SortResult:
    """LSD Radix для float через побитовый монотонный ключ IEEE-754

    Время O(8*(N+256)) = O(N) (ровно 8 байтовых проходов), память O(N)
    Устойчивость: есть
    """
    n = len(arr)
    if n < 2:
        return list(arr), 0, 0
    keys = [_f2u(x) for x in arr]
    sorted_keys, w, c = _lsd_nonneg(keys, 8, 255, 256)
    out = [_u2f(k) for k in sorted_keys]
    return out, c, w + 2 * n