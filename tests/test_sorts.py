"""Автотесты: краевые случаи, счётчики, устойчивость, домен

Запуск: pytest -q или python -m tests.test_sorts
"""
from __future__ import annotations

import random

from bench.checks import correct
from bench.data import DISTRIBUTIONS, generate
from sorts import insertion_sort, radix_sort, radix_sort_float, selection_sort

SORTS = [insertion_sort, selection_sort, radix_sort]


def test_edge_cases() -> None:
    """Пустой массив, 1 элемент, все равные, два элемента, отрицательные, большие числа"""
    cases = [[], [1], [2, 1], [7] * 50, [5, 5, 1, 1, 3], [-3, -1, -7, 0, 2],
             [10 ** 20, -(10 ** 20), 0, 123456789]]
    for f in SORTS:
        for case in cases:
            src = case.copy()
            out, c, w = f(src.copy())
            assert correct(src, out), (f.__name__, case, out)
            assert c >= 0 and w >= 0


def test_random_crosscheck() -> None:
    """200 случайных массивов: все три алгоритма дают одинаковый корректный результат"""
    rng = random.Random(2026)
    for _ in range(200):
        n = rng.randrange(0, 60)
        data = [rng.randrange(-100, 100) for _ in range(n)]
        results = []
        for f in SORTS:
            out, _, _ = f(data.copy())
            assert correct(data, out), (f.__name__, data)
            results.append(out)
        assert results[0] == results[1] == results[2]


def test_all_distributions() -> None:
    """Корректность на всех 5 (+1) типах распределений"""
    for kind in DISTRIBUTIONS:
        data = generate(500, kind, seed=11)
        for f in SORTS:
            out, _, _ = f(data.copy())
            assert correct(data, out), (f.__name__, kind)


def test_inplace_and_purity() -> None:
    """In-place возвращает тот же объект; radix не мутирует вход и возвращает новый список"""
    data = [3, 1, 2]
    for f in (insertion_sort, selection_sort):
        same, _, _ = f(data.copy())
        assert isinstance(same, list)
        arr = data.copy()
        out, _, _ = f(arr)
        assert out is arr
    src = [3, 1, 2]
    out, _, _ = radix_sort(src)
    assert src == [3, 1, 2] and out is not src


def test_selection_counters() -> None:
    """Selection: ровно N(N-1)/2 сравнений на любых данных; на sorted — 0 записей"""
    for kind in DISTRIBUTIONS:
        n = 200
        data = generate(n, kind, seed=5)
        _, c, w = selection_sort(data.copy())
        assert c == n * (n - 1) // 2, kind
        assert w % 3 == 0
    _, c, w = selection_sort(list(range(1, 101)))
    assert w == 0


def test_insertion_counters() -> None:
    """Insertion: sorted → N-1 сравнений и 0 записей; reversed → N(N-1)/2 сдвигов"""
    n = 100
    _, c, w = insertion_sort(list(range(1, n + 1)))
    assert (c, w) == (n - 1, 0)
    _, c, w = insertion_sort(list(range(n, 0, -1)))
    assert c == n * (n - 1) // 2
    assert w == n * (n - 1) // 2 + (n - 1)


def test_radix_counters_and_bases() -> None:
    """Radix: 0 сравнений при распределении (N-1 на поиск max), результат не зависит от base"""
    data = generate(1000, "random", seed=3)
    ref = None
    for base in (2, 16, 256, 65536):
        out, c, w = radix_sort(data.copy(), base=base)
        assert correct(data, out)
        assert c == len(data) - 1
        ref = out if ref is None else ref
        assert out == ref


def test_adaptivity() -> None:
    """Insertion адаптивна, Selection — нет; локальные обмены дают почти линейное поведение"""
    n = 2000
    rnd = generate(n, "random", 1)
    loc = generate(n, "nearly_local", 1)
    _, c_rnd, _ = insertion_sort(rnd.copy())
    _, c_loc, _ = insertion_sort(loc.copy())
    assert c_loc < c_rnd / 20
    assert selection_sort(rnd.copy())[1] == selection_sort(loc.copy())[1]


def test_stability() -> None:
    """Insertion и Radix устойчивы, Selection — нет"""
    data = [(1, "a"), (0, "b"), (1, "c"), (0, "d")]
    key = lambda t: t[0]
    ins, _, _ = insertion_sort(data.copy(), key=key)
    rad, _, _ = radix_sort(data.copy(), key=key)
    expected = [(0, "b"), (0, "d"), (1, "a"), (1, "c")]
    assert ins == expected and rad == expected
    sel, _, _ = selection_sort([(2, "a"), (2, "b"), (1, "c")], key=key)
    assert sel != [(1, "c"), (2, "a"), (2, "b")]      # демонстрация неустойчивости


def test_key_comparator() -> None:
    """Live-задача: сортировка по остатку mod 3, при равенстве — по убыванию"""
    data = [7, 3, 8, 1, 6, 5, 4]
    key = lambda x: (x % 3, -x)
    expected = [6, 3, 7, 4, 1, 8, 5]
    assert insertion_sort(data.copy(), key=key)[0] == expected
    assert selection_sort(data.copy(), key=key)[0] == expected
    # Для radix ключ упаковывается в одно целое: старшие биты — mod 3, младшие — (MAX - x).
    rkey = lambda x: (x % 3) * 1000 + (999 - x)
    assert radix_sort(data.copy(), key=rkey)[0] == expected


def test_float_domain() -> None:
    """Insertion/Selection работают с float; radix требует int, для float есть битовая версия"""
    data = [3.5, -1.25, 0.0, 2.0, -7.75, 3.5]
    assert correct(data, insertion_sort(data.copy())[0])
    assert correct(data, selection_sort(data.copy())[0])
    try:
        radix_sort(data.copy())
    except TypeError:
        pass
    else:
        raise AssertionError("ожидался TypeError для float")
    out, _, _ = radix_sort_float(data.copy())
    assert correct(data, out)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK  {name}")
    print("\nВсе тесты пройдены")