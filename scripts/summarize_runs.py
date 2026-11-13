#!/usr/bin/env python3
"""Собирает сводную таблицу по всем готовым пробам в outputs/.

Использование:
    PYTHONPATH=src .venv/bin/python scripts/summarize_runs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _model_sort(name: str) -> tuple[int, str]:
    size = name.replace("qwen2.5-", "").replace("b", "")
    try:
        return (float(size), name)
    except ValueError:
        return (0.0, name)


def collect() -> pd.DataFrame:
    rows = []
    for run in sorted(ROOT.glob("outputs/*/*")):
        summary_path = run / "probe" / "summary.csv"
        if not summary_path.exists():
            continue
        summ = pd.read_csv(summary_path)
        probe = summ[~summ["uses_answer"]].iloc[0]
        baselines = summ[summ["uses_answer"]].sort_values("test_auc", ascending=False)
        best_base = baselines.iloc[0]
        layer = int(probe["method"].split("слой ")[1].split(" ")[0].rstrip(")"))
        rows.append({
            "model": run.parent.name,
            "lang": run.name,
            "error_rate": probe["base_error_rate"],
            "probe_layer": layer,
            "probe_auc": probe["test_auc"],
            "probe_ci_low": probe["ci_low"],
            "probe_ci_high": probe["ci_high"],
            "best_baseline": best_base["method"],
            "baseline_auc": best_base["test_auc"],
        })
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> str:
    out = []
    out.append("| Модель | Язык | Ошибок (test) | Слой пробы | Проба AUC | Лучший baseline | Baseline AUC |")
    out.append("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        out.append(
            f"| {r['model'].replace('qwen2.5-', '')} | {r['lang']} | "
            f"{r['error_rate']:.1%} | {r['probe_layer']} | "
            f"{r['probe_auc']:.3f} [{r['probe_ci_low']:.3f}–{r['probe_ci_high']:.3f}] | "
            f"{r['best_baseline']} | {r['baseline_auc']:.3f} |"
        )
    return "\n".join(out)


def main() -> int:
    df = collect()
    if df.empty:
        print("Не найдено ни одного outputs/*/probe/summary.csv", file=sys.stderr)
        return 1
    df = df.sort_values(by=["model", "lang"], key=lambda s: s.map(_model_sort) if s.name == "model" else s)
    print(format_table(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
