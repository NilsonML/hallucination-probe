"""Пробы на синтетических данных: сигнал о будущей ошибке заложен только в средние слои."""
import numpy as np
import pandas as pd

from hprobe.probes import layer_sweep, summarize


def make_synthetic(n=900, layers=6, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    error = rng.integers(0, 2, n)
    hidden = rng.normal(size=(n, layers, dim)).astype(np.float16)
    hidden[:, 3, 0] += 1.5 * error  # «знание» о будущей ошибке — в слое 3
    df = pd.DataFrame({
        "split": rng.choice(["train", "val", "test"], n, p=[0.7, 0.15, 0.15]),
        "correct": 1 - error, "error": error,
        "mean_logprob": -rng.random(n) - 0.3 * error, "min_logprob": -rng.random(n),
        "first_prob": rng.random(n), "first_entropy": rng.random(n), "mean_entropy": rng.random(n),
        "consistency": rng.random(n),
    })
    return df, hidden


def test_probe_finds_informative_layer():
    df, hidden = make_synthetic()
    layers = layer_sweep(df, hidden)
    assert int(layers.loc[layers["test_auc"].idxmax(), "layer"]) == 3
    assert layers.loc[3, "test_auc"] > 0.8
    assert layers.drop(index=3)["test_auc"].max() < 0.65


def test_summary_selects_layer_on_val_and_reports_ci():
    df, hidden = make_synthetic()
    summary = summarize(df, hidden, layer_sweep(df, hidden))
    probe = summary[~summary["uses_answer"]].iloc[0]
    assert "слой 3" in probe["method"]
    assert probe["ci_low"] < probe["test_auc"] < probe["ci_high"]
    assert summary.iloc[0]["method"] == probe["method"]
