"""Подготовка вопросов RuBQ 2.0.

Из исходных JSON собирается таблица вопросов, на которые есть ответ:
  uid, question_ru, question_en, gold (все допустимые формулировки ответа),
  answer_kind (entity / year / number / text), domain (тематика), split.

Источник: RuBQ 2.0, https://github.com/vladislavneon/RuBQ (лицензия CC BY-SA 4.0).
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

RAW_FILES = ("RuBQ_2.0_dev.json", "RuBQ_2.0_test.json")

# Тематика вопроса определяется по свойству Wikidata в SPARQL-запросе.
# Берётся первое свойство из списка вопроса, которое есть в словаре.
DOMAINS: dict[str, list[str]] = {
    "geography": ["P17", "P36", "P131", "P206", "P403", "P1376", "P30", "P2046", "P1082",
                  "P885", "P159", "P47", "P610", "P2044", "P2043", "P4614", "P421", "P38"],
    "culture": ["P50", "P57", "P170", "P86", "P175", "P161", "P58", "P84", "P136", "P1303",
                "P577", "P272", "P495", "P674", "P407", "P676", "P87", "P162", "P264", "P123"],
    "people": ["P19", "P20", "P569", "P570", "P26", "P40", "P22", "P25", "P106", "P69", "P119",
               "P27", "P1559", "P39", "P166", "P3373", "P735", "P734", "P1477", "P742", "P108"],
    "history": ["P571", "P580", "P582", "P585", "P112", "P61", "P793", "P607", "P1365",
                "P1366", "P576", "P35", "P6", "P488", "P710", "P575", "P97", "P53", "P140"],
    "sport": ["P641", "P118", "P413", "P54", "P1346", "P286", "P115"],
    "science": ["P2101", "P2102", "P178", "P2067", "P1086", "P246", "P2054", "P2048", "P2101",
                "P2386", "P2583", "P1215", "P2120"],
}
DOMAINS["geography"] += ["P276", "P706", "P37", "P2044"]
DOMAINS["culture"] += ["P2437", "P412", "P135", "P1113", "P149", "P180", "P410"]
PROP2DOMAIN = {p: d for d, props in DOMAINS.items() for p in props}

DATE_RE = re.compile(r"^-?\d{1,4}-\d{2}-\d{2}T")
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def domain_of(question_props: list[str]) -> str:
    for p in question_props:
        d = PROP2DOMAIN.get(p.split(":")[-1])
        if d:
            return d
    return "other"


def _year(value: str) -> str:
    return str(int(value[: value.index("-", 1)]))


def gold_of(q: dict) -> tuple[list[str], str]:
    """Все допустимые формулировки ответа и тип ответа."""
    golds: list[str] = []
    kinds = Counter()
    for a in q["answers"]:
        v = str(a.get("value", ""))
        if a["type"] == "uri":
            kinds["entity"] += 1
            names = a.get("wd_names") or {}
            golds += [a.get("label") or ""] + names.get("ru", []) + names.get("en", []) + a.get("wp_names", [])
        elif DATE_RE.match(v):
            kinds["year"] += 1
            golds.append(_year(v))
        elif NUM_RE.match(v):
            kinds["number"] += 1
            golds.append(v)
        else:
            kinds["text"] += 1
            golds.append(v)
    # Ответ разметчиков RuBQ: иногда точнее Wikidata (например, «пять этажей»)
    if q.get("answer_text"):
        golds.append(q["answer_text"])
    seen, uniq = set(), []
    for g in golds:
        g = g.strip()
        if g and g.lower() not in seen:
            seen.add(g.lower())
            uniq.append(g)
    return uniq, kinds.most_common(1)[0][0]


def load_rubq(raw_dir: str | Path, max_answers: int = 5) -> list[dict]:
    raw_dir = Path(raw_dir)
    items = []
    for name in RAW_FILES:
        items += json.loads((raw_dir / name).read_text(encoding="utf-8"))
    rows = []
    for q in items:
        # Без ответа (тег no-answer) и вопросы-перечисления с >5 ответами не годятся
        # для однозначной проверки короткого ответа.
        if not q["answers"] or len(q["answers"]) > max_answers:
            continue
        if "count" in (q.get("tags") or []):
            continue
        gold, kind = gold_of(q)
        rows.append({
            "uid": q["uid"],
            "question_ru": q["question_text"].strip(),
            "question_en": (q.get("question_eng") or "").strip(),
            "gold": gold,
            "answer_kind": kind,
            "domain": domain_of(q["question_props"]),
            "tags": q.get("tags") or [],
            "paragraph_ids": (q.get("paragraphs_uids") or {}).get("with_answer", []),
        })
    rows.sort(key=lambda r: r["uid"])
    return rows


def assign_splits(rows: list[dict], seed: int = 42, val: float = 0.15, test: float = 0.15) -> None:
    """Случайное разбиение, стратифицированное по тематике. Один uid = одна часть
    для обоих языков, чтобы один и тот же факт не попал в train и test."""
    rng = random.Random(seed)
    by_domain: dict[str, list[dict]] = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)
    for group in by_domain.values():
        rng.shuffle(group)
        n = len(group)
        n_test, n_val = round(n * test), round(n * val)
        for i, r in enumerate(group):
            r["split"] = "test" if i < n_test else "val" if i < n_test + n_val else "train"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="RuBQ 2.0 → data/questions.jsonl")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/questions.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_rubq(args.raw)
    assign_splits(rows, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Вопросов: {len(rows)} → {args.out}")
    for key in ("split", "domain", "answer_kind"):
        print(f"  {key}: {dict(Counter(r[key] for r in rows).most_common())}")


if __name__ == "__main__":
    main()
