"""Адаптивный поиск: модель ищет в Википедии ТОЛЬКО когда проба предсказывает
ошибку. Ответ на вопрос В5: есть ли практическая польза от пробы.

СХЕМА
-----
вопрос -> риск ошибки (проба по hidden.npy) -> риск >= порога?
    нет -> отвечаем как обычно (поиск не тратится)
    да  -> в промпт добавляем абзац Википедии из RuBQ, отвечаем заново

Меняя порог, получаем кривую «точность ответов против доли вопросов, где
понадобился поиск». Границы: доля 0.0 = никогда не искать (точность
исходной модели), доля 1.0 = искать всегда (потолок качества).

ПОЧЕМУ АБЗАЦ ИЗ RuBQ, А НЕ ЖИВОЙ ПОИСК ПО ВИКИПЕДИИ
----------------------------------------------------
Это контролируемая постановка: абзац гарантированно содержит ответ
(paragraph_ids), поэтому измеряем вклад ОТБОРА вопросов, а не качество
ретривера. Живой поиск — следующий шаг после конференции.

ЧЕСТНОСТЬ: кривая строится ТОЛЬКО на тестовой части; пробы и порог не
переобучаются под тест — порог вообще не обучается, он параметр кривой.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .generate import load_model
from .labeling import is_correct
from .prompts import build_search_prompt

THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
              0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_probe(path: Path) -> dict:
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def probe_risk(vectors: np.ndarray, pack: dict) -> np.ndarray:
    """Вероятность ошибки по векторам слоя: (N, d) -> (N,)."""
    z = ((vectors.astype(np.float64) - pack["mean"]) / pack["scale"]) @ pack["coef"].T + pack["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def compute_curve(df, risk, correct_search) -> pd.DataFrame:
    """Точность и доля поиска для каждого порога.

    Пороги сортируются по УБЫВАНИЮ: сначала высокие (мало поиска),
    потом низкие (много поиска, точность ближе к ctx).
    """
    out = []
    base_correct = df["correct"].to_numpy(dtype=float)
    for t in sorted(THRESHOLDS, reverse=True):
        flag = risk >= t
        acc = float(np.where(flag, correct_search, base_correct).mean())
        out.append({"threshold": round(t, 2), "accuracy": round(acc, 4),
                    "search_fraction": round(flag.mean(), 4)})
    return pd.DataFrame(out)


def regenerate_batch(tok, model, batch, lang):
    """Перегенерация с абзацем Википедии."""
    prompts = [build_search_prompt(tok, q[f"question_{lang}"],
                                   [q.get("search_context", "")], lang) for q in batch]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    gen = model.generate(**enc, do_sample=False, pad_token_id=tok.pad_token_id,
                         max_new_tokens=24)
    texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [t.strip() for t in texts]


def add_search_context(questions, paragraphs):
    """Добавляет абзацы Википедии к вопросам, у которых они есть.

    Вопросы без paragraph_ids не могут участвовать в поиске:
    для них считаем correct_search = correct_base (поиск невозможен).
    """
    for q in questions:
        ids = q.get("paragraph_ids", [])
        q["has_context"] = bool(ids)
        q["search_context"] = "\n".join(paragraphs.get(str(i), "") for i in ids if str(i) in paragraphs)
    return questions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--paragraphs", default="data/raw/RuBQ_2.0_paragraphs.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()

    out = Path(args.run)
    df = pd.read_csv(out / "labels.csv")
    hidden = np.load(out / "hidden.npy", mmap_mode="r")
    probe = load_probe(out / "probe" / "probe.npz")

    qkey = f"question_{df['lang'].iloc[0] if 'lang' in df.columns else 'ru'}"
    questions = [json.loads(line) for line in open(args.questions, encoding="utf-8")]
    qmap = {q["uid"]: q for q in questions}
    paragraphs = json.loads(Path(args.paragraphs).read_text(encoding="utf-8"))
    questions = add_search_context(questions, paragraphs)

    uids = df["uid"].tolist()
    risk = probe_risk(hidden[:, int(probe["layer"]), :].astype(np.float64), probe)
    risk_map = {uids[i]: risk[i] for i in range(len(uids))}

    # Вопросы с контекстом и риском
    todo = [q for q in questions if q["uid"] in qmap and q["has_context"] and risk_map[q["uid"]] >= 0.0]
    print(f"Всего: {len(questions)}, перегенерация: {len(todo)}")

    tok, model = load_model(args.model, args.dtype, args.device)
    answers_search = {}
    for start in tqdm(range(0, len(todo), args.batch_size), desc="search"):
        batch = todo[start: start + args.batch_size]
        for q, a in zip(batch, regenerate_batch(tok, model, batch, qkey.split("_")[-1])):
            answers_search[q["uid"]] = int(is_correct(a, qmap[q["uid"]]["gold"], qmap[q["uid"]]["answer_kind"]))

    # Для вопросов без контекста или не вошедших в todo: поиск невозможен
    correct_search = np.array([
        answers_search.get(uid, int(df.loc[df["uid"] == uid, "correct"].iloc[0]))
        for uid in uids
    ])

    curve = compute_curve(df, risk, correct_search)
    curve.to_csv(out / "search" / "curve.csv", index=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve["search_fraction"], curve["accuracy"], "o-", label="адаптивный поиск")
    base_acc = df["correct"].mean()
    search_acc = correct_search.mean()
    ax.axhline(base_acc, color="gray", ls="--", lw=1, label=f"без поиска ({base_acc:.1%})")
    ax.axhline(search_acc, color="green", ls="--", lw=1, label=f"поиск всем ({search_acc:.1%})")
    ax.set_xlabel("доля вопросов с поиском")
    ax.set_ylabel("точность")
    ax.set_xlim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "search" / "curve.png", dpi=150)
    plt.close(fig)
    print(curve.to_string(index=False))
    print(f"\n{out / 'search' / 'curve.png'}")


if __name__ == "__main__":
    main()
