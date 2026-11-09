"""Адаптивный поиск (В5): модель обращается к Википедии только при высоком риске ошибки.

    python -m hprobe.search \
        --run outputs/qwen2.5-1.5b/ru \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --questions data/questions.jsonl \
        --raw data/raw

Работает как потолок: абзацы берутся из RuBQ (гарантированно содержат ответ).
Сохраняет в <run>/search/curve.csv и curve.png:
- threshold      порог риска
- fraction       доля вопросов, для которых включён поиск
- accuracy       точность при данном пороге
- searched_acc   точность на вопросах с поиском
- baseline_acc   точность без поиска (threshold = 1.0)
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


def load_probe(path: Path) -> dict:
    """Загружает веса пробы из npz, сохранённые probes.py."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def probe_risk(probe: dict, hidden: np.ndarray) -> np.ndarray:
    """Вероятность ошибки для каждого вопроса по лучшему слою."""
    layer = int(probe["layer"])
    x = np.asarray(hidden[:, layer, :], dtype=np.float32)
    z = (x - probe["mean"]) / probe["scale"]
    logit = z @ probe["coef"].T + probe["intercept"]
    return 1 / (1 + np.exp(-logit.ravel()))


def load_paragraphs(raw_dir: Path) -> dict[str, str]:
    """uid абзаца -> текст."""
    path = raw_dir / "RuBQ_2.0_paragraphs.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {p["uid"]: p["text"] for p in data}


def compute_curve(
    records: list[dict],
    labels: pd.DataFrame,
    risk: np.ndarray,
    paragraph_map: dict[str, str],
    thresholds: list[float],
    regenerate_fn,
    lang: str = "ru",
) -> pd.DataFrame:
    """Кривая точность / доля вопросов с поиском.

    regenerate_fn(records_to_fix) -> list[dict] с ключом 'answer' (новый ответ).
    """
    uids = [r["uid"] for r in records]
    uid2record = {r["uid"]: r for r in records}
    uid2label = dict(zip(labels["uid"].tolist(), labels.to_dict("records")))

    # Пороги в порядке убывания: от «почти никто не ищет» до «почти все ищут».
    thresholds = sorted(set(thresholds), reverse=True)
    rows = []

    for thr in thresholds:
        mask = risk >= thr
        to_fix = [records[i] for i in range(len(records)) if mask[i]]
        fixed = {item["uid"]: item for item in regenerate_fn(to_fix)} if to_fix else {}

        correct, total, searched_correct, searched_total = 0, 0, 0, 0
        for i, uid in enumerate(uids):
            label = uid2label.get(uid)
            if label is None:
                continue
            total += 1
            if mask[i] and uid in fixed:
                answer = fixed[uid]["answer"]
                searched_total += 1
            else:
                answer = uid2record[uid]["answer"]
            if is_correct(answer, label["gold"].split(" | "), label["kind"]):
                correct += 1
                if mask[i] and uid in fixed:
                    searched_correct += 1

        rows.append({
            "threshold": thr,
            "fraction": searched_total / total if total else 0.0,
            "accuracy": correct / total if total else 0.0,
            "searched_acc": searched_correct / searched_total if searched_total else float("nan"),
            "n_searched": searched_total,
            "n_total": total,
        })

    return pd.DataFrame(rows)


def regenerate_batch(
    tok, model, to_fix: list[dict], paragraph_map: dict[str, str],
    questions: dict[int, dict], lang: str, max_new_tokens: int
) -> list[dict]:
    """Перегенерирует ответы с добавлением абзацев RuBQ."""
    from .generate import clean_answer

    prompts = []
    for r in to_fix:
        q = questions[r["uid"]]
        pids = q.get("paragraph_ids") or []
        paragraphs = [paragraph_map.get(pid, "") for pid in pids if pid in paragraph_map]
        if not paragraphs:
            paragraphs = [""]  # вопросы без абзацев не падают
        prompts.append(build_search_prompt(tok, q[f"question_{lang}"], paragraphs, lang))

    out = []
    for r, prompt in tqdm(zip(to_fix, prompts), total=len(to_fix), desc="search", leave=False):
        enc = tok(prompt, return_tensors="pt", padding=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
        text = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append({"uid": r["uid"], "answer": clean_answer(text)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--lang", choices=["ru", "en"], default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--thresholds", type=float, nargs="+", default=None,
                    help="пороги риска (по умолчанию 0.0, 0.05, ..., 0.95)")
    args = ap.parse_args()

    run = Path(args.run)
    out = run / "search"
    out.mkdir(exist_ok=True)

    meta = json.loads((run / "meta.json").read_text(encoding="utf-8")) if (run / "meta.json").exists() else {}
    lang = args.lang or meta.get("lang", "ru")

    records = [json.loads(l) for l in (run / "records.jsonl").open(encoding="utf-8")]
    labels = pd.read_csv(run / "labels.csv")
    hidden = np.load(run / "hidden.npy", mmap_mode="r")
    probe = load_probe(run / "probe" / "probe.npz")

    questions = {}
    for line in Path(args.questions).open(encoding="utf-8"):
        q = json.loads(line)
        questions[q["uid"]] = q

    paragraph_map = load_paragraphs(Path(args.raw))
    risk = probe_risk(probe, hidden)

    thresholds = args.thresholds if args.thresholds is not None else list(np.arange(0.0, 1.0, 0.05))

    tok, model = load_model(args.model)

    def regenerate_fn(to_fix):
        return regenerate_batch(tok, model, to_fix, paragraph_map, questions, lang, args.max_new_tokens)

    curve = compute_curve(records, labels, risk, paragraph_map, thresholds, regenerate_fn, lang)
    curve.to_csv(out / "curve.csv", index=False)
    plot_curve(curve, out / "curve.png", run.name)
    print(curve.round(3).to_string(index=False))


def plot_curve(curve: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(curve["fraction"], curve["accuracy"], marker="o", ms=3)
    ax.set(xlabel="Доля вопросов с поиском", ylabel="Точность",
           title=f"Адаптивный поиск · {title}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
