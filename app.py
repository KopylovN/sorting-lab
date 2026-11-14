"""ИДЗ №1 — интерактивный дашборд эмпирического анализа сортировок

Запуск:  streamlit run app.py
Вариант: Insertion Sort + Selection Sort + LSD Radix Sort
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from bench.data import DISTRIBUTIONS, generate
from bench.fitting import MODELS, slope_table
from bench.memory import measure_memory_grid
from bench.runner import BenchConfig, aggregate, estimate_seconds, run_benchmark
from sorts.registry import REGISTRY
from viz import interactive as iv
from viz.steps import build_frames

AUTHOR = "Копылов Никита"
GROUP = "Б06-505"
VARIANT = "Insertion Sort · Selection Sort · LSD Radix Sort"
# ================================

ALL_DIST = "Все пять распределений (сравнение)"

st.set_page_config(page_title="ИДЗ №1 — Сортировки", page_icon="📊", layout="wide")


def show(fig, container=None, key: str | None = None) -> None:
    """Совместимый вывод plotly-графика (старые/новые версии Streamlit)"""
    target = container if container is not None else st
    try:
        target.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        target.plotly_chart(fig, use_container_width=True, key=key)


def resolve_sizes(selected: list[str], mode: str, manual: list[int]) -> dict[str, list[int]]:
    """Собирает сетку размеров с учётом ограничений O(N²) — не больше 10 000"""
    sizes: dict[str, list[int]] = {}
    for aid in selected:
        spec = REGISTRY[aid]
        if mode == "Быстрый пресет":
            s = list(spec.sizes_quick)
        elif mode == "Полная сетка":
            s = list(spec.sizes_full)
        else:
            s = [n for n in manual if spec.family == "linear" or n <= 10000]
        sizes[aid] = sorted(s)
    return sizes


# ----------------------------- САЙДБАР -----------------------------
with st.sidebar:
    st.header("Параметры эксперимента")
    st.caption(f"{AUTHOR} · {GROUP}")
    st.caption(f"Вариант: {VARIANT}")

    selected = [aid for aid, spec in REGISTRY.items()
                if st.checkbox(spec.name, value=True, key=f"cb_{aid}")]
    dist_label = st.selectbox("Тип распределения данных", list(DISTRIBUTIONS.values()) + [ALL_DIST])
    dists = list(DISTRIBUTIONS) if dist_label == ALL_DIST else \
        [k for k, v in DISTRIBUTIONS.items() if v == dist_label]
    if dist_label == ALL_DIST:
        dists = ["random", "sorted", "reversed", "nearly", "duplicates"]

    mode = st.radio("Сетка размеров", ["Быстрый пресет", "Полная сетка", "Ручной выбор"])
    manual: list[int] = []
    if mode == "Ручной выбор":
        pool = sorted(set(REGISTRY["insertion"].sizes_full + REGISTRY["radix"].sizes_full))
        manual = st.multiselect("Размеры N", pool, default=[100, 1000, 5000, 10000])

    repeats = st.slider("Число повторений (разные seed)", 5, 10, 5)
    seed = int(st.number_input("Базовый seed", value=42, step=1))
    radix_base = st.selectbox("Основание Radix Sort (base)", [16, 256, 65536], index=1)
    single_n = st.slider("N для одиночного прогона", 100, 200_000, 5000, step=100)
    validate = st.checkbox("Проверять корректность (вне таймера)", value=True)

    sizes = resolve_sizes(selected, mode, manual)
    cfg = BenchConfig(algorithms=selected, distributions=dists, sizes=sizes,
                      repeats=repeats, base_seed=seed, radix_base=radix_base,
                      validate=validate, profile="quick" if mode == "Быстрый пресет" else "full")
    est = estimate_seconds(cfg)
    st.info(f"Оценка длительности прогона: ~{est:.0f} с")
    if est > 120:
        st.warning("Долго! Уменьшите N, число повторений или возьмите быстрый пресет")

st.title("Эмпирический и теоретический анализ алгоритмов сортировки")
st.caption("ИДЗ №1 · Алгоритмы и структуры данных · ФБМФ МФТИ")

tab_run, tab_fit, tab_mem, tab_anim, tab_report = st.tabs(
    ["Замеры", "Сложность и fitting", "Память", "Визуализатор", "Отчёт"])

# ----------------------------- ТАБ 1: ЗАМЕРЫ -----------------------------
with tab_run:
    st.subheader("Серия замеров")
    st.markdown(
        "Методика: `time.perf_counter()` вокруг **одного** вызова сортировки; генерация массива "
        "и `arr.copy()` — вне таймера; каждая точка усредняется по разным seed; усы = ±σ.")
    if not selected:
        st.error("Выберите хотя бы один алгоритм в сайдбаре.")
    if st.button("Запустить серию замеров", type="primary", disabled=not selected):
        bar, label = st.progress(0.0), st.empty()
        t0 = time.perf_counter()
        raw = run_benchmark(cfg, lambda f, m: (bar.progress(f), label.text(m)))
        bar.empty(); label.empty()
        st.session_state["raw"] = raw
        st.success(f"Готово за {time.perf_counter() - t0:.1f} с. Точек: {len(raw)}. "
                   f"Корректность: {'OK' if raw['correct'].all() else 'ОШИБКА'}")

    raw: pd.DataFrame | None = st.session_state.get("raw")
    if raw is None:
        st.info("Запустите замеры — результаты сохранятся в сессии и будут доступны на всех вкладках.")
    else:
        agg = aggregate(raw)
        scale = st.radio("Масштаб осей", ["Линейный", "Log-Log"], horizontal=True)
        loglog = scale == "Log-Log"
        if len(dists) == 1:
            show(iv.time_chart(agg, dists[0], loglog), key="chart_time")
        else:
            show(iv.facet_chart(agg, loglog), key="chart_facet")

        c1, c2 = st.columns(2)
        with c1:
            show(iv.counters_chart(agg, dists[0], "comparisons_mean"), key="chart_comparisons")
        with c2:
            show(iv.counters_chart(agg, dists[0], "writes_mean"), key="chart_writes")

        st.markdown("#### Сводная статистика (μ, σ, медиана)")
        st.dataframe(agg, hide_index=True)
        with st.expander("Сырые замеры (каждый повтор)"):
            st.dataframe(raw, hide_index=True)
        st.download_button("⬇Скачать сырые данные CSV",
                           raw.to_csv(index=False).encode("utf-8"),
                           "raw_measurements.csv", "text/csv")

    with st.expander("Одиночный прогон на выбранном N"):
        if st.button("Запустить одиночный прогон", disabled=not selected):
            rows = []
            data = generate(single_n, dists[0], seed)
            for aid in selected:
                spec = REGISTRY[aid]
                params = {"base": radix_base} if aid == "radix" else {}
                copy = data.copy()
                t0 = time.perf_counter()
                out, c, w = spec.func(copy, **params)
                dt = (time.perf_counter() - t0) * 1000
                from bench.checks import correct
                rows.append({"Алгоритм": spec.name, "N": single_n, "T, мс": round(dt, 3),
                             "comparisons": c, "writes": w,
                             "in-place": spec.inplace, "устойчивая": spec.stable,
                             "корректно": correct(data, out)})
            st.dataframe(pd.DataFrame(rows), hide_index=True)

# ----------------------------- ТАБ 2: СЛОЖНОСТЬ -----------------------------
with tab_fit:
    st.subheader("Теоретическая аппроксимация и наклон log-log")
    raw = st.session_state.get("raw")
    if raw is None:
        st.info("Сначала выполните замеры на вкладке «Замеры».")
    else:
        agg = aggregate(raw)
        c1, c2, c3 = st.columns(3)
        aid = c1.selectbox("Алгоритм", sorted(agg["algorithm"].unique()),
                           format_func=lambda a: REGISTRY[a].name)
        dist = c2.selectbox("Распределение", sorted(agg["distribution"].unique()),
                            format_func=lambda d: DISTRIBUTIONS[d], key="fit_dist")
        default_model = "N²" if REGISTRY[aid].family == "quadratic" else "N"
        model = c3.selectbox("Теоретическая модель", list(MODELS),
                             index=list(MODELS).index(default_model))
        fig, c, r2, k = iv.fit_chart(agg, aid, dist, model)
        show(fig, key="chart_fit")
        m1, m2, m3 = st.columns(3)
        m1.metric("Коэффициент c МНК", f"{c:.3e}")
        m2.metric("R²", f"{r2:.4f}")
        m3.metric("Наклон log-log, k", f"{k:.3f}")
        st.caption("Ожидания по ТЗ: k ≈ 2 для O(N²); k ≈ 1…1.15 для O(N log N); k ≈ 1 для O(N)")

        st.markdown("#### Наклоны по всем комбинациям")
        st.dataframe(slope_table(agg), hide_index=True)
        st.markdown(
            "**Как читать:** степенная зависимость `T = c·N^k` в log-log становится прямой "
            "`log T = k·log N + log c`, поэтому наклон = степень. Малые N исключаются из "
            "регрессии: там доминируют накладные расходы интерпретатора")

# ----------------------------- ТАБ 3: ПАМЯТЬ -----------------------------
with tab_mem:
    st.subheader("Профилирование памяти (tracemalloc)")
    st.markdown("Копия входа создаётся **до** `tracemalloc.start()`, поэтому в пик попадает "
                "только дополнительная память алгоритма. Время здесь не меряем: "
                "tracemalloc замедляет интерпретатор в 2–4 раза")
    mem_sizes = st.multiselect("Размеры N", [1000, 5000, 10000, 50000, 100000, 300000],
                               default=[1000, 10000, 100000])
    mem_repeats = st.slider("Повторов на точку", 1, 5, 3)
    if st.button("Замерить память", disabled=not selected):
        ms = {a: [n for n in mem_sizes if REGISTRY[a].family == "linear" or n <= 10000]
              for a in selected}
        bar = st.progress(0.0)
        mem = measure_memory_grid(selected, ["random"], ms, seed=seed, repeats=mem_repeats,
                                  radix_base=radix_base,
                                  progress_cb=lambda f, m: bar.progress(f))
        bar.empty()
        st.session_state["mem"] = mem
    mem = st.session_state.get("mem")
    if mem is not None:
        show(iv.memory_chart(mem), key="chart_memory")
        st.dataframe(mem, hide_index=True)
        st.caption("In-place (Insertion/Selection) → O(1): пик единицы КБ (служебные объекты). "
                   "Radix → O(N + base): буфер + список pos ≈ 16·N байт.")

# ----------------------------- ТАБ 4: АНИМАТОР -----------------------------
with tab_anim:
    st.subheader("Пошаговая визуализация (N = 20–30)")
    c1, c2, c3, c4 = st.columns(4)
    anim_algo = c1.selectbox("Алгоритм", list(REGISTRY), format_func=lambda a: REGISTRY[a].name)
    anim_dist = c2.selectbox("Распределение", list(DISTRIBUTIONS),
                             format_func=lambda d: DISTRIBUTIONS[d], key="anim_dist")
    anim_n = c3.slider("Размер массива N", 5, 30, 20)
    anim_seed = int(c4.number_input("seed", value=7, key="anim_seed"))
    delay = st.select_slider("Задержка между кадрами, с", [0.0, 0.02, 0.05, 0.1, 0.2, 0.4], value=0.1)
    st.markdown("🟦 обычный · 🟧 сравнивается · 🟥 записывается · 🟩 зафиксирован · "
                "🟪 текущий ключ/позиция · 🔴 текущий минимум")

    if st.button("Сгенерировать массив и кадры"):
        data = generate(anim_n, anim_dist, anim_seed)
        st.session_state["frames"] = build_frames(anim_algo, data)
        st.session_state["anim_idx"] = 0
        st.session_state["anim_src"] = data

    frames = st.session_state.get("frames")
    if not frames:
        st.info("Сгенерируйте массив, чтобы получить покадровую трассу")
    else:
        last = len(frames) - 1
        idx = min(st.session_state.get("anim_idx", 0), last)
        b1, b2, b3, b4 = st.columns(4)
        if b2.button("⏭ Шаг вперёд"):
            idx = min(idx + 1, last)
        if b3.button("⏮ Шаг назад"):
            idx = max(idx - 1, 0)
        if b4.button("⟲ В начало"):
            idx = 0
        play = b1.button("▶ Проиграть")
        st.session_state["anim_idx"] = idx
        pos = st.slider("Кадр", 0, last, value=idx)
        st.session_state["anim_idx"] = pos
        chart_ph, info_ph, bucket_ph = st.empty(), st.empty(), st.empty()

        def draw(i: int) -> None:
            """Отрисовка одного кадра."""
            f = frames[i]
            show(iv.frame_chart(f, f"Кадр {i + 1}/{last + 1}"), container=chart_ph, key=f"chart_frame_{i}")
            info_ph.markdown(f"**{f['message']}**  ·  сравнений: `{f['comparisons']}`  "
                             f"·  записей: `{f['writes']}`")
            if f["buckets"] is not None:
                lines = [f"`{d}`: {', '.join(map(str, b)) if b else '—'}"
                         for d, b in enumerate(f["buckets"])]
                bucket_ph.markdown("**Корзины (разряд 10^%d):**\n\n" % f["digit"] + "\n\n".join(lines))
            else:
                bucket_ph.empty()

        if play:
            for i in range(pos, last + 1):
                draw(i)
                time.sleep(delay)
            st.session_state["anim_idx"] = last
            st.rerun()
        draw(pos)
        with st.expander("Исходный массив"):
            st.code(st.session_state.get("anim_src"))

# ----------------------------- ТАБ 5: ОТЧЁТ -----------------------------
with tab_report:
    st.subheader("Паспорт варианта")
    st.table(pd.DataFrame([{
        "Алгоритм": s.name, "Семейство": s.family, "Best": s.time_best, "Avg": s.time_avg,
        "Worst": s.time_worst, "Память": s.memory, "In-place": s.inplace,
        "Устойчивая": s.stable, "Сравнительная": s.comparison_based, "Домен": s.domain,
    } for s in REGISTRY.values()]))
    path = Path(__file__).parent / "ANALYSIS.md"
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.warning("Создайте ANALYSIS.md с аналитическими выводами")