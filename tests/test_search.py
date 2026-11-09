"""Синтетическая проверка адаптивного поиска без GPU."""
import numpy as np
import pandas as pd
import pytest

from hprobe.search import compute_curve, load_paragraphs, probe_risk


def make_synthetic_run(n=200, seed=0):
    rng = np.random.default_rng(seed)
    error = rng.integers(0, 2, n)
    records = [{"uid": i, "answer": "wrong" if error[i] else "right"} for i in range(n)]
    labels = pd.DataFrame({
        "uid": list(range(n)),
        "correct": 1 - error,
        "gold": ["right"] * n,
        "kind": ["text"] * n,
    })
    # риск выше у ошибочных ответов
    risk = rng.random(n) * 0.4 + 0.6 * error
    return records, labels, risk


def test_curve_monotonicity_and_boundaries():
    records, labels, risk = make_synthetic_run()

    def regenerate_fn(to_fix):
        # Поиск исправляет 90% ошибочных ответов.
        return [{"uid": r["uid"], "answer": f"right-{r['uid']}"} for r in to_fix]

    paragraphs = {"p1": "context"}
    thresholds = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    curve = compute_curve(records, labels, risk, paragraphs, thresholds, regenerate_fn)

    # Пороги отсортированы по убыванию.
    assert curve["threshold"].tolist() == sorted(thresholds, reverse=True)

    # Доля поиска неубывающая при движении сверху вниз.
    fractions = curve["fraction"].tolist()
    assert all(fractions[i] <= fractions[i + 1] for i in range(len(fractions) - 1))

    # Крайние точки.
    assert fractions[0] == 0.0  # threshold=1.0 — никто не ищет
    assert fractions[-1] == 1.0  # threshold=0.0 — все ищут

    # Точность растёт с ростом доли поиска (в среднем), потому что поиск исправляет ошибки.
    accs = curve["accuracy"].tolist()
    assert accs[-1] > accs[0]
    assert 0.0 <= min(accs) <= max(accs) <= 1.0


def test_questions_without_paragraphs_do_not_crash():
    records, labels, risk = make_synthetic_run(n=20)
    paragraphs = {}  # нет абзацев

    def regenerate_fn(to_fix):
        return [{"uid": r["uid"], "answer": f"right-{r['uid']}"} for r in to_fix]

    curve = compute_curve(records, labels, risk, paragraphs, [0.5], regenerate_fn)
    assert len(curve) == 1
    assert curve["n_searched"].iloc[0] > 0


def test_probe_risk_shape():
    probe = {
        "layer": np.int64(1),
        "mean": np.zeros(8, dtype=np.float32),
        "scale": np.ones(8, dtype=np.float32),
        "coef": np.zeros((1, 8), dtype=np.float32),
        "intercept": np.array([0.0], dtype=np.float32),
    }
    hidden = np.random.normal(size=(10, 3, 8)).astype(np.float16)
    risk = probe_risk(probe, hidden)
    assert risk.shape == (10,)
    assert np.allclose(risk, 0.5)


def test_load_paragraphs_missing_file(tmp_path):
    assert load_paragraphs(tmp_path) == {}
