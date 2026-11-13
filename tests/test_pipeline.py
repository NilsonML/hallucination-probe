"""Сквозная проверка генерации и разметки на крошечной случайной модели (без интернета).

Модель отвечает бессмыслицей, но проверяется сама механика: батчи с левым паддингом,
извлечение скрытых состояний, продолжение прерванного прогона, сборка hidden.npy.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def make_tiny_model(path: Path, texts: list[str]) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=600, special_tokens=special,
                                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = CHATML
    tok.save_pretrained(path)

    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(path)


def test_generate_and_label(tmp_path):
    questions = ROOT / "data" / "questions.jsonl"
    qs = [json.loads(l) for l in questions.open(encoding="utf-8")]
    model_dir = tmp_path / "tiny"
    make_tiny_model(model_dir, [q["question_ru"] for q in qs] + [q["question_en"] for q in qs])

    out = tmp_path / "run"
    cmd = [sys.executable, "-m", "hprobe.generate", "--model", str(model_dir), "--lang", "ru",
           "--out", str(out), "--batch-size", "4", "--n-samples", "3", "--max-new-tokens", "6"]
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    # первый прогон «прерывается» на 10 вопросах, второй продолжает до 14
    subprocess.run(cmd + ["--limit", "10"], check=True, cwd=ROOT, env=env)
    subprocess.run(cmd + ["--limit", "14"], check=True, cwd=ROOT, env=env)

    records = [json.loads(l) for l in (out / "records.jsonl").open(encoding="utf-8")]
    assert len(records) == 14 and len({r["uid"] for r in records}) == 14
    assert all(len(r["samples"]) == 3 for r in records)

    hidden = np.load(out / "hidden.npy")
    assert hidden.shape == (14, 4, 32)  # 3 слоя + эмбеддинги, hidden_size 32
    assert np.isfinite(hidden.astype(np.float32)).all()

    # Скрытое состояние не должно зависеть от соседей по батчу (проверка левого паддинга)
    from hprobe.generate import load_model
    from hprobe.prompts import build_prompt
    tok, model = load_model(str(model_dir), "fp32", "cpu")
    r = records[5]
    enc = tok([build_prompt(tok, r["question"], "ru")], return_tensors="pt")
    enc.pop("token_type_ids", None)
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    single = torch.stack([h[0, -1] for h in hs]).numpy()
    assert np.allclose(single, hidden[5].astype(np.float32), atol=2e-2)

    subprocess.run([sys.executable, "-m", "hprobe.label", "--run", str(out)], check=True, cwd=ROOT, env=env)
    labels = pd.read_csv(out / "labels.csv")
    assert len(labels) == 14 and set(labels["correct"]) <= {0, 1}
    assert (out / "manual_check.csv").exists()
