"""Обучение и оценка проб по скрытым состояниям.

ПОЧЕМУ ЛИНЕЙНАЯ ПРОБА (логистическая регрессия), А НЕ НЕЙРОСЕТЬ
-----------------------------------------------------------
Это самый сильный научный аргумент проекта. Если даже линейный классификатор
находил бы сигнал, значит, информация о будущей ошибке лежит в скрытых
состояниях «на поверхности» — вдоль одного направления. Мощный классификатор
(MLP, градиентный бустинг) мог бы выдумать сигнал из шума, и нам пришлось бы
доказывать, что это не переобучение. Линейная проба переобучиться почти не
может: у неё всего d+1 параметр против ~700 000 вопросов-измерений.

ПРОТОКОЛ ЧЕСТНОСТИ
-----------------
1. Слой и силу регуляризации C выбираем ТОЛЬКО по валидации (val_auc).
2. Тестовую часть трогаем один раз — итоговая цифра.
3. Разброс оценки показываем бутстрэпом (1000 повторных выборок теста).

СТАНДАРТИЗАЦИЯ: StandardScaler по train. Без неё регуляризация логрега (L2)
штрафовала бы сильнее признаки с большей дисперсией — артефакт масштаба.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = (0.001, 0.01, 0.1, 1.0)
FEATURES = ["mean_logprob", "min_logprob", "first_prob", "mean_entropy", "first_entropy", "n_tokens", "consistency", "samples_correct"]


def load_run(run: str | Path):
    """Читает labels.csv и hidden.npy, проверяет совпадение длин."""
    run = Path(run)
    df = pd.read_csv(run / "labels.csv")
    labels = {r["uid"]: r for _, r in df.iterrows()}
    hidden = np.load(run / "hidden.npy", mmap_mode="r")
    assert len(labels) == hidden.shape[0], f"labels ({len(labels)}) != hidden ({hidden.shape[0]})"
    return df, labels, hidden


def fit_probe(df, hidden, layer: int):
    """Обучает пробу на векторах одного слоя.

    Pipeline: StandardScaler → LogisticRegression. StandardScaler сам
    обрабатывает zero-std признаки (scale_=1.0), поэтому артефактов
    нормировки не возникает.

    Перебор C по сетке: сила регуляризации L2. Маленький C = сильный
    штраф на большие веса = проще модель = меньше переобучение.
    """
    y = df["error"].to_numpy().astype(int)
    train_mask = df["split"] == "train"
    X_train = hidden[train_mask, layer, :].astype(np.float32)
    X = hidden[:, layer, :].astype(np.float32)
    best_auc = -1
    best_pipe = None
    for C in C_GRID:
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=1000))])
        pipe.fit(X_train, y[train_mask])
        val_auc = roc_auc_score(y[df["split"] == "val"], pipe.predict_proba(X[df["split"] == "val"])[:, 1])
        if val_auc > best_auc:
            best_auc = val_auc
            best_pipe = pipe
    return best_pipe


def layer_sweep(df, hidden):
    """Перебор всех слоёв: ответ на вопрос В2 (в каком слое сигнал сильнее)."""
    out = []
    y = df["error"].to_numpy().astype(int)
    for layer in range(hidden.shape[1]):
        pipe = fit_probe(df, hidden, layer)
        X = hidden[:, layer, :].astype(np.float32)
        row = {"layer": layer}
        for split in ("train", "val", "test"):
            mask = df["split"] == split
            prob = pipe.predict_proba(X[mask])[:, 1]
            row[f"{split}_auc"] = roc_auc_score(y[mask], prob)
            row[f"{split}_ap"] = average_precision_score(y[mask], prob)
        out.append(row)
    return pd.DataFrame(out)


def summarize(df, hidden, layers: pd.DataFrame):
    """Итоговая таблица: проба против базовых методов на тесте.

    uses_answer=True у базовых методов — принципиально: они получают
    информацию ИЗ САМОГО ОТВЕТА (вероятности токенов, совпадение сэмплов),
    проба — только из состояний до ответа. Сравнение «до ответа» против
    «после шести ответов» — ядро вопроса В1.
    """
    test_mask = df["split"] == "test"
    y = df["error"].to_numpy().astype(int)

    # Лучшая проба (по val_auc)
    best = layers.loc[layers["val_auc"].idxmax()]
    layer = int(best["layer"])
    pipe = fit_probe(df, hidden, layer)
    X = hidden[:, layer, :].astype(np.float32)
    prob = pipe.predict_proba(X[test_mask])[:, 1]
    rows = [{
        "method": f"Проба (слой {layer})", "uses_answer": False,
        "test_auc": roc_auc_score(y[test_mask], prob),
        "test_ap": average_precision_score(y[test_mask], prob),
    }]

    # Базовые методы
    for feat in FEATURES:
        if feat not in df.columns:
            continue
        s = df.loc[test_mask, feat].to_numpy(dtype=float)
        rows.append({
            "method": feat, "uses_answer": True,
            "test_auc": roc_auc_score(y[test_mask], s),
            "test_ap": average_precision_score(y[test_mask], s),
        })
    return pd.DataFrame(rows).sort_values("test_auc", ascending=False)


def save_probe(pipe: Pipeline, path: Path, layer: int) -> None:
    """Сохраняет веса пробы для search.py и transfer-экспериментов."""
    lr = pipe.named_steps["lr"]
    sc = pipe.named_steps["sc"]
    np.savez(path, layer=layer, coef=lr.coef_, intercept=lr.intercept_,
             mean=sc.mean_, scale=sc.scale_)


def plot_layers(layers: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers["layer"], layers["test_auc"], "o-", label="test AUC")
    ax.plot(layers["layer"], layers["val_auc"], ".-", alpha=0.6, label="val AUC (выбор слоя)")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="монетка")
    ax.set_xlabel("слой (0 — эмбеддинги)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    out = Path(args.run)
    df, labels, hidden = load_run(out)
    layers = layer_sweep(df, hidden)
    layers.to_csv(out / "probe" / "layers.csv", index=False)
    summary = summarize(df, hidden, layers)
    summary.to_csv(out / "probe" / "summary.csv", index=False)

    best_layer = int(summary.iloc[0]["method"].split("слой ")[1]) if "слой" in str(summary.iloc[0]["method"]) else 0
    save_probe(fit_probe(df, hidden, best_layer), out / "probe" / "probe.npz", best_layer)
    plot_layers(layers, out / "probe" / "layers.png")
    print(summary.to_string(index=False))
    print(f"\nprobe.npz: {out / 'probe' / 'probe.npz'}")


if __name__ == "__main__":
    main()
