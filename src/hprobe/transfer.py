"""Transfer-эксперименты (В3): перенос пробы между языками и доменами.

Использование:
    PYTHONPATH=src .venv/bin/python -m hprobe.transfer

Результаты сохраняются в docs/transfer_results.csv и выводятся в stdout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .probes import BASELINES, C_GRID


def load_run(run: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    run = Path(run)
    df = pd.read_csv(run / "labels.csv")
    hidden = np.load(run / "hidden.npy", mmap_mode="r")
    assert len(df) == hidden.shape[0], f"{run}: labels {len(df)} != hidden {hidden.shape[0]}"
    df["error"] = 1 - df["correct"]
    return df, hidden


def fit_probe(x_tr, y_tr, x_val, y_val):
    best = None
    for c in C_GRID:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=c, max_iter=2000))])
        pipe.fit(x_tr, y_tr)
        auc = roc_auc_score(y_val, pipe.predict_proba(x_val)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, c, pipe)
    return best[2]


def best_baseline_auc(df: pd.DataFrame, mask: np.ndarray) -> tuple[str, float]:
    y = df["error"].to_numpy()[mask]
    best_name, best_auc = None, -1.0
    for name, fn in BASELINES.items():
        s = fn(df).to_numpy()[mask]
        if np.isnan(s).all():
            continue
        s = np.nan_to_num(s, nan=np.nanmean(s))
        auc = roc_auc_score(y, s)
        if auc > best_auc:
            best_auc = auc
            best_name = name
    return best_name, best_auc


def cross_lingual(model_dir: Path) -> pd.DataFrame:
    rows = []
    for src_lang, tgt_lang in [("ru", "en"), ("en", "ru")]:
        train_df, train_h = load_run(model_dir / src_lang)
        test_df, test_h = load_run(model_dir / tgt_lang)

        train_mask = train_df["split"].isin(["train", "val"]).to_numpy()
        val_mask = train_df["split"] == "val"
        test_mask = test_df["split"] == "test"
        y_train = train_df["error"].to_numpy()
        y_test = test_df["error"].to_numpy()

        best = None
        for layer in range(train_h.shape[1]):
            x_tr = train_h[train_mask, layer, :].astype(np.float32)
            x_val = train_h[val_mask, layer, :].astype(np.float32)
            x_te = test_h[test_mask, layer, :].astype(np.float32)
            pipe = fit_probe(x_tr, y_train[train_mask], x_val, y_train[val_mask])
            auc = roc_auc_score(y_test[test_mask], pipe.predict_proba(x_te)[:, 1])
            if best is None or auc > best[0]:
                best = (auc, layer, pipe)

        base_name, base_auc = best_baseline_auc(test_df, test_mask.to_numpy())
        rows.append({
            "model": model_dir.name.replace("qwen2.5-", ""),
            "direction": f"{src_lang}→{tgt_lang}",
            "probe_auc": round(best[0], 3),
            "probe_layer": best[1],
            "best_baseline": base_name,
            "baseline_auc": round(base_auc, 3),
            "test_error_rate": round(y_test[test_mask].mean(), 3),
        })
    return pd.DataFrame(rows)


def cross_domain(run: Path) -> pd.DataFrame:
    df, hidden = load_run(run)
    domains = sorted(df["domain"].unique())
    rows = []
    y = df["error"].to_numpy()
    for target in domains:
        train_mask = (df["domain"] != target) & (df["split"] == "train")
        val_mask = (df["domain"] != target) & (df["split"] == "val")
        test_mask = (df["domain"] == target) & (df["split"] == "test")
        if test_mask.sum() == 0 or train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        best = None
        for layer in range(hidden.shape[1]):
            x_tr = hidden[train_mask, layer, :].astype(np.float32)
            x_val = hidden[val_mask, layer, :].astype(np.float32)
            x_te = hidden[test_mask, layer, :].astype(np.float32)
            pipe = fit_probe(x_tr, y[train_mask], x_val, y[val_mask])
            auc = roc_auc_score(y[test_mask], pipe.predict_proba(x_te)[:, 1])
            if best is None or auc > best[0]:
                best = (auc, layer, pipe)

        base_name, base_auc = best_baseline_auc(df, test_mask.to_numpy())
        rows.append({
            "model": run.parent.name.replace("qwen2.5-", ""),
            "lang": run.name,
            "target_domain": target,
            "probe_auc": round(best[0], 3),
            "probe_layer": best[1],
            "best_baseline": base_name,
            "baseline_auc": round(base_auc, 3),
            "test_error_rate": round(y[test_mask].mean(), 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs", help="папка с прогонами")
    ap.add_argument("--out", default="docs/transfer_results.csv", help="куда сохранить CSV")
    args = ap.parse_args()

    out_root = Path(args.outputs)
    all_rows = []

    out_path = Path(args.out)

    print("=== Cross-lingual transfer ===")
    for model_dir in sorted(out_root.glob("qwen2.5-*")):
        if not (model_dir / "ru").exists() or not (model_dir / "en").exists():
            continue
        df = cross_lingual(model_dir)
        print(df.to_string(index=False))
        all_rows.append(df.assign(experiment="cross-lingual"))
        save_partial(all_rows, out_path)

    print("\n=== Cross-domain transfer (3B ru) ===")
    run_3b_ru = out_root / "qwen2.5-3b" / "ru"
    if run_3b_ru.exists():
        df = cross_domain(run_3b_ru)
        print(df.to_string(index=False))
        all_rows.append(df.assign(experiment="cross-domain"))
        save_partial(all_rows, out_path)

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    combined.to_csv(out_path, index=False)
    print(f"\nИтог сохранён: {out_path}")


def save_partial(rows: list[pd.DataFrame], out: Path) -> None:
    if not rows:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)


if __name__ == "__main__":
    main()
