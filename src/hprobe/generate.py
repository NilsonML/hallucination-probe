"""Прогон языковой модели по вопросам.

Для каждого вопроса сохраняется:
- жадный ответ модели и признаки неуверенности по нему (логарифмы вероятностей токенов,
  энтропия) — базовые методы для сравнения;
- N ответов, сгенерированных с сэмплированием, — для метода самосогласованности;
- скрытые состояния ВСЕХ слоёв на последнем токене запроса, то есть до того,
  как модель начала отвечать. По ним обучаются классификаторы-пробы.

Результаты пишутся по частям, поэтому прерванный прогон в Colab можно продолжить
той же командой: готовые вопросы пропускаются.

Пример:
    python -m hprobe.generate --model Qwen/Qwen2.5-1.5B-Instruct --lang ru \
        --out outputs/qwen2.5-1.5b/ru
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import build_prompt, clean_answer


def load_model(name: str, dtype: str = "auto", device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {"auto": (torch.float16 if device == "cuda" else torch.float32),
                   "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch_dtype).to(device).eval()
    return tok, model


def _token_stats(scores: tuple[torch.Tensor, ...], seqs: torch.Tensor, eos_ids: set[int]) -> list[dict]:
    """Признаки неуверенности по жадному ответу (по каждому примеру в батче)."""
    logits = torch.stack(scores, dim=1).float()            # [B, T, V]
    logp = torch.log_softmax(logits, dim=-1)
    chosen = logp.gather(-1, seqs.unsqueeze(-1)).squeeze(-1)  # [B, T]
    entropy = -(logp.exp() * logp).sum(-1)                   # [B, T]
    out = []
    for b in range(seqs.shape[0]):
        n = seqs.shape[1]
        for t in range(seqs.shape[1]):
            if int(seqs[b, t]) in eos_ids:
                n = t
                break
        n = max(n, 1)
        lp, ent = chosen[b, :n], entropy[b, :n]
        out.append({
            "mean_logprob": float(lp.mean()),
            "min_logprob": float(lp.min()),
            "first_prob": float(lp[0].exp()),
            "mean_entropy": float(ent.mean()),
            "first_entropy": float(ent[0]),
            "n_tokens": int(n),
        })
    return out


@torch.no_grad()
def run_batch(tok, model, prompts: list[str], n_samples: int, max_new_tokens: int,
              temperature: float, top_p: float):
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_len = enc["input_ids"].shape[1]
    cfg_eos = model.generation_config.eos_token_id
    eos_ids = set(cfg_eos if isinstance(cfg_eos, list) else [cfg_eos])
    eos_ids |= {tok.eos_token_id, tok.pad_token_id}
    eos_ids.discard(None)

    greedy = model.generate(
        **enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
        output_scores=True, output_hidden_states=True, return_dict_in_generate=True,
        temperature=None, top_p=None, top_k=None)
    # hidden_states[0] — проход по запросу: кортеж (эмбеддинги + каждый слой) [B, T, D].
    # Левый паддинг → позиция -1 = последний токен запроса у всех примеров.
    hidden = torch.stack([h[:, -1, :] for h in greedy.hidden_states[0]], dim=1)  # [B, L+1, D]
    gen = greedy.sequences[:, prompt_len:]
    answers = [clean_answer(t) for t in tok.batch_decode(gen, skip_special_tokens=True)]
    stats = _token_stats(greedy.scores, gen, eos_ids)

    samples: list[list[str]] = [[] for _ in prompts]
    if n_samples > 0:
        sampled = model.generate(
            **enc, do_sample=True, temperature=temperature, top_p=top_p, top_k=0,
            num_return_sequences=n_samples, max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(sampled[:, prompt_len:], skip_special_tokens=True)
        for i, t in enumerate(texts):
            samples[i // n_samples].append(clean_answer(t))

    return answers, samples, stats, hidden.to(torch.float16).cpu().numpy()


def done_uids(out: Path) -> set[int]:
    path = out / "records.jsonl"
    if not path.exists():
        return set()
    return {json.loads(line)["uid"] for line in path.open(encoding="utf-8")}


def next_chunk_index(out: Path) -> int:
    """Следующий свободный номер чанка — восстанавливаем из records.jsonl, а не из glob."""
    path = out / "records.jsonl"
    if not path.exists():
        return 0
    max_chunk = "0"
    for line in path.open(encoding="utf-8"):
        chunk = json.loads(line).get("chunk")
        if chunk is not None and chunk > max_chunk:
            max_chunk = chunk
    return int(max_chunk) + 1


def merge_chunks(out: Path) -> None:
    """Собирает части в hidden.npy в порядке records.jsonl."""
    records = [json.loads(line) for line in (out / "records.jsonl").open(encoding="utf-8")]
    chunks = {p.stem: np.load(p) for p in sorted((out / "chunks").glob("*.npy"))}
    rows = [chunks[r["chunk"]][r["row"]] for r in records]
    np.save(out / "hidden.npy", np.stack(rows))
    print(f"hidden.npy: {len(rows)} × {rows[0].shape} → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--lang", choices=["ru", "en"], default="ru")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=0, help="только первые N вопросов (проверка)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)

    qkey = f"question_{args.lang}"
    questions = [json.loads(line) for line in open(args.questions, encoding="utf-8")]
    questions = [q for q in questions if q[qkey]]
    if args.limit:
        questions = questions[: args.limit]
    done = done_uids(out)
    todo = [q for q in questions if q["uid"] not in done]
    print(f"Вопросов: {len(questions)}, уже готово: {len(done)}, осталось: {len(todo)}")

    tok, model = load_model(args.model, args.dtype)
    meta = {"model": args.model, "lang": args.lang, "n_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size, "position": "last_prompt_token",
            "args": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Не сортируем по длине: иначе нарушается порядок questions/records/hidden/labels.
    prompts = {q["uid"]: build_prompt(tok, q[qkey], args.lang) for q in todo}
    n_chunks = next_chunk_index(out)

    with (out / "records.jsonl").open("a", encoding="utf-8") as f:
        for start in tqdm(range(0, len(todo), args.batch_size), desc=f"{args.lang}"):
            batch = todo[start: start + args.batch_size]
            answers, samples, stats, hidden = run_batch(
                tok, model, [prompts[q["uid"]] for q in batch], args.n_samples,
                args.max_new_tokens, args.temperature, args.top_p)
            chunk = f"{n_chunks:05d}"
            np.save(out / "chunks" / f"{chunk}.npy", hidden)
            n_chunks += 1
            for i, q in enumerate(batch):
                f.write(json.dumps({
                    "uid": q["uid"], "lang": args.lang, "question": q[qkey],
                    "answer": answers[i], "samples": samples[i], **stats[i],
                    "chunk": chunk, "row": i,
                }, ensure_ascii=False) + "\n")
            f.flush()

    merge_chunks(out)


if __name__ == "__main__":
    main()
