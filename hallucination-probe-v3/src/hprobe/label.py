"""Разметка ответов: records.jsonl → labels.csv + manual_check.csv.

Генерация — тяжёлый GPU-шаг в Colab, а правила проверки мы улучшаем по ходу
проекта. Если бы разметка была внутри generate.py, каждая правка «а давайте
считать так» стоила бы повторного прогона модели на час. Разделив, мы
переразмечаем 2327 ответов за секунды, а GPU-прогон делаем один раз.

labels.csv — главный «интерфейсный» файл: probes.py и search.py читают
только его (+ hidden.npy).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from .labeling import answers_agree, is_correct, is_refusal

FEATURES = ["mean_logprob", "min_logprob", "first_prob", "mean_entropy", "first_entropy", "n_tokens"]


def find_questions(path: str | Path = "data/questions.jsonl") -> dict[int, dict]:
    return {q["uid"]: q for q in (json.loads(line) for line in Path(path).open(encoding="utf-8"))}


def label_run(run: str | Path, questions_path: str | Path = "data/questions.jsonl",
              check: bool = False) -> pd.DataFrame:
    """Создаёт labels.csv и manual_check.csv из records.jsonl."""
    run = Path(run)
    qmap = find_questions(questions_path)
    records = [json.loads(line) for line in (run / "records.jsonl").open(encoding="utf-8") if line.strip()]

    rows = []
    for r in records:
        q = qmap[r["uid"]]
        correct = int(is_correct(r["answer"], q["gold"], q["answer_kind"]))
        samples = r.get("samples", [])
        # Самосогласованность: доля сэмплов, совпадающих с жадным ответом.
        consistency = (sum(answers_agree(s, r["answer"]) for s in samples) / len(samples)) if samples else 0.0
        # Доля верных среди сэмплов (отдельная полезная метрика).
        samples_correct = (sum(is_correct(s, q["gold"], q["answer_kind"]) for s in samples) / len(samples)) if samples else 0.0
        rows.append({
            "uid": r["uid"], "domain": q["domain"], "split": q["split"],
            "answer_kind": q["answer_kind"],
            "correct": correct, "error": 1 - correct,
            "refusal": int(is_refusal(r["answer"])),
            "consistency": round(consistency, 4),
            "samples_correct": round(samples_correct, 4),
            **{f: r.get(f) for f in FEATURES},
        })
    df = pd.DataFrame(rows)
    df.to_csv(run / "labels.csv", index=False)

    # 100 случайных ответов на ручную проверку. Фиксированный seed — чтобы
    # выборка не «прыгала» между перезапусками, иначе человеку пришлось бы
    # проверять заново уже проверенное.
    rng = random.Random(42)
    pick = rng.sample(range(len(records)), min(100, len(records)))
    manual = pd.DataFrame([{
        "uid": records[i]["uid"], "question": records[i]["question"],
        "answer": records[i]["answer"],
        "gold": " | ".join(qmap[records[i]["uid"]]["gold"]),
        "correct": int(df.iloc[i]["correct"]), "human": "",
    } for i in pick])
    manual.to_csv(run / "manual_check.csv", index=False)

    print(f"{run}: размечено {len(df)} ответов, верных {df['correct'].mean():.1%}, отказов {df['refusal'].mean():.1%}")
    if check:
        filled = manual[manual["human"].astype(str).str.strip() != ""]
        if len(filled) == 0:
            print("manual_check.csv ещё не заполнен: поставьте 1/0 в колонку human и запустите снова с --check")
        else:
            human = filled["human"].astype(str).str.strip().astype(int)
            acc = (human == filled["correct"]).mean()
            print(f"совпадение автоматики с человеком на {len(filled)} примерах: {acc:.1%} — это число идёт в тезисы")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--check", action="store_true",
                    help="сверить заполненный manual_check.csv с автоматикой")
    args = ap.parse_args()
    label_run(args.run, args.questions, check=args.check)
