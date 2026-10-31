"""Классификаторы-пробы: предсказание ошибки модели по скрытым состояниям до ответа.

    python -m hprobe.probes --run outputs/qwen2.5-1.5b/ru

Целевая переменная: 1 — модель ответит неверно (галлюцинация или отказ), 0 — верно.
Для каждого слоя обучается логистическая регрессия (регуляризация подбирается на val),
качество считается на test. Для сравнения — базовые методы, которые используют уже
сгенерированный ответ: вероятность токенов, энтропия, самосогласованность сэмплов.

Результаты в <run>/probe/: layers.csv, summary.csv, layers.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = (0.001, 0.01, 0.1, 1.0)

# Базовые методы: чем больше значение, тем вероятнее ошибка
BASELINES = {
    "Средний log p токенов": lambda d: -d["mean_logprob"],
    "Минимальный log p токена": lambda d: -d["min_logprob"],
    "Вероятность 1-го токена": lambda d: -d["first_prob"],
    "Энтропия 1-го токена": lambda d: d["first_entropy"],
    "Средняя энтропия": lambda d: d["mean_entropy"],
    "Самосогласованность (5 сэмплов)": lambda d: -d["consistency"],
}


def load_run(run: Path) -> tuple[pd.DataFrame, np.ndarray]:
    labels = pd.read_csv(run / "labels.csv")
    hidden = np.load(run / "hidden.npy", mmap_mode="r")
    assert len(labels) == hidden.shape[0], "labels.csv и hidden.npy разной длины"
    labels["error"] = 1 - labels["correct"]
    return labels, hidden


def fit_probe(x_tr, y_tr, x_val, y_val, c_grid=C_GRID):
    """Логистическая регрессия; C выбирается по ROC-AUC на val."""
    best = None
    for c in c_grid:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
        clf.fit(x_tr, y_tr)
        auc = roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, c, clf)
    return best[2], best[1]


def bootstrap_auc(y, score, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], score[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def layer_sweep(df: pd.DataFrame, hidden: np.ndarray) -> pd.DataFrame:
    tr, va, te = (df["split"] == s for s in ("train", "val", "test"))
    y = df["error"].to_numpy()
    rows = []
    for layer in range(hidden.shape[1]):
        x = np.asarray(hidden[:, layer, :], dtype=np.float32)
        clf, c = fit_probe(x[tr], y[tr], x[va], y[va])
        p = clf.predict_proba(x[te])[:, 1]
        rows.append({"layer": layer, "C": c, "val_auc": roc_auc_score(y[va], clf.predict_proba(x[va])[:, 1]),
                     "test_auc": roc_auc_score(y[te], p), "test_pr_auc": average_precision_score(y[te], p)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, hidden: np.ndarray, layers: pd.DataFrame) -> pd.DataFrame:
    te = (df["split"] == "test").to_numpy()
    y = df["error"].to_numpy()
    rows = []
    for name, fn in BASELINES.items():
        s = fn(df).to_numpy()
        if np.isnan(s[te]).all():
            continue
        s = np.nan_to_num(s, nan=np.nanmean(s))
        lo, hi = bootstrap_auc(y[te], s[te])
        rows.append({"method": name, "uses_answer": True, "test_auc": roc_auc_score(y[te], s[te]),
                     "ci_low": lo, "ci_high": hi, "test_pr_auc": average_precision_score(y[te], s[te])})

    # Лучший слой выбирается по val, а не по test
    best = layers.loc[layers["val_auc"].idxmax()]
    layer = int(best["layer"])
    tr, va = (df["split"] == "train").to_numpy(), (df["split"] == "val").to_numpy()
    x = np.asarray(hidden[:, layer, :], dtype=np.float32)
    clf, _ = fit_probe(x[tr], y[tr], x[va], y[va])
    p = clf.predict_proba(x[te])[:, 1]
    lo, hi = bootstrap_auc(y[te], p)
    rows.append({"method": f"Проба, слой {layer} (до ответа)", "uses_answer": False,
                 "test_auc": roc_auc_score(y[te], p), "ci_low": lo, "ci_high": hi,
                 "test_pr_auc": average_precision_score(y[te], p)})
    out = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    out.insert(0, "base_error_rate", float(y[te].mean()))
    return out


def plot_layers(layers: pd.DataFrame, summary: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers["layer"], layers["test_auc"], marker="o", ms=3, label="Проба по слою (test)")
    sc = summary[summary["method"].str.startswith("Самосогласованность")]
    if len(sc):
        ax.axhline(sc["test_auc"].iloc[0], ls="--", c="gray", label="Самосогласованность, 5 сэмплов")
    lp = summary[summary["method"] == "Средний log p токенов"]
    if len(lp):
        ax.axhline(lp["test_auc"].iloc[0], ls=":", c="gray", label="Средний log p ответа")
    ax.axhline(0.5, c="lightgray", lw=0.8)
    ax.set(xlabel="Слой (0 — эмбеддинги)", ylabel="ROC-AUC предсказания ошибки", title=title)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    out = run / "probe"
    out.mkdir(exist_ok=True)

    df, hidden = load_run(run)
    print(f"{run}: {len(df)} вопросов, доля ошибок {df['error'].mean():.1%}, слоёв {hidden.shape[1]}")
    layers = layer_sweep(df, hidden)
    layers.to_csv(out / "layers.csv", index=False)
    summary = summarize(df, hidden, layers)
    summary.to_csv(out / "summary.csv", index=False)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8")) if (run / "meta.json").exists() else {}
    plot_layers(layers, summary, out / "layers.png", f"{meta.get('model', run.name)} · {meta.get('lang', '')}")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
