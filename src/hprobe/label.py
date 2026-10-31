"""Разметка прогона: правильность ответа, отказ, самосогласованность.

    python -m hprobe.label --run outputs/qwen2.5-1.5b/ru

Создаёт в папке прогона:
- labels.csv — по строке на вопрос (в порядке hidden.npy);
- manual_check.csv — 100 случайных ответов для ручной проверки разметки
  (столбец human: 1 — ответ верный, 0 — неверный). После заполнения:
  python -m hprobe.label --run ... --check  → доля совпадений с автоматической разметкой.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .labeling import answers_agree, is_correct, is_refusal


def label_run(run: Path, questions_path: Path) -> pd.DataFrame:
    questions = {}
    for line in questions_path.open(encoding="utf-8"):
        q = json.loads(line)
        questions[q["uid"]] = q
    rows = []
    for line in (run / "records.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        q = questions[r["uid"]]
        samples = r.get("samples") or []
        rows.append({
            "uid": r["uid"], "split": q["split"], "domain": q["domain"], "kind": q["answer_kind"],
            "question": r["question"], "answer": r["answer"],
            "gold": " | ".join(q["gold"][:5]),
            "correct": int(is_correct(r["answer"], q["gold"], q["answer_kind"])),
            "refused": int(is_refusal(r["answer"])),
            # доля сэмплов, совпадающих с жадным ответом (базовый метод самосогласованности)
            "consistency": (sum(answers_agree(r["answer"], s) for s in samples) / len(samples))
            if samples else float("nan"),
            "samples_correct": (sum(is_correct(s, q["gold"], q["answer_kind"]) for s in samples)
                                / len(samples)) if samples else float("nan"),
            **{k: r[k] for k in ("mean_logprob", "min_logprob", "first_prob",
                                 "mean_entropy", "first_entropy", "n_tokens")},
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--n-check", type=int, default=100)
    ap.add_argument("--check", action="store_true", help="сравнить ручную разметку с автоматической")
    args = ap.parse_args()
    run = Path(args.run)

    if args.check:
        m = pd.read_csv(run / "manual_check.csv").dropna(subset=["human"])
        agree = (m["human"].astype(int) == m["correct"]).mean()
        print(f"Проверено вручную: {len(m)}, совпадение с автоматической разметкой: {agree:.1%}")
        print(m[m["human"].astype(int) != m["correct"]][["question", "answer", "gold", "correct", "human"]]
              .to_string(max_colwidth=50))
        return

    df = label_run(run, Path(args.questions))
    df.to_csv(run / "labels.csv", index=False)
    print(f"{run}: {len(df)} ответов, верных {df.correct.mean():.1%}, отказов {df.refused.mean():.1%}")
    print(df.groupby("domain")["correct"].agg(["mean", "size"]).round(3).to_string())

    check_path = run / "manual_check.csv"
    if not check_path.exists():
        sample = df.sample(min(args.n_check, len(df)), random_state=0)
        sample = sample[["uid", "question", "answer", "gold", "correct"]].assign(human="")
        sample.to_csv(check_path, index=False)
        print(f"Для ручной проверки: {check_path}")


if __name__ == "__main__":
    main()
