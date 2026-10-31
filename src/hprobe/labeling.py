"""Автоматическая проверка правильности короткого ответа модели.

Правила:
- entity / text: ответ верен, если совпадает (по основам слов) хотя бы с одной
  допустимой формулировкой: все значимые слова формулировки есть в ответе,
  либо все значимые слова ответа входят в формулировку («Гайдай» ⊂ «Леонид Иович Гайдай»).
- year: в ответе есть тот же год.
- number: в ответе есть число, отличающееся от эталона не более чем на 2 %.
- Отказ («не знаю») — отдельный флаг, правильным не считается.

Качество разметки проверяется вручную на случайной выборке (scripts/manual_check.py).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

_RU = snowballstemmer.stemmer("russian")
_EN = snowballstemmer.stemmer("english")

STOP = set("""и в во на с со к по из от до о об у за для при the of a an and in on at to
был была были год году года лет г гг это является""".split())

NUM_WORDS = {
    "ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11,
    "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15,
    "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
    "семьдесят": 70, "восемьдесят": 80, "девяносто": 90, "сто": 100,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20, "hundred": 100,
}

REFUSAL_RE = re.compile(
    r"не знаю|не могу (?:ответить|сказать)|нет (?:данных|информации)|неизвестно|"
    r"i don'?t know|i'?m not sure|unknown|cannot answer|can'?t answer", re.I)

NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower().replace("ё", "е"))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")  # ударения
    text = re.sub(r"[^\w\s.,-]", " ", text)
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    if re.fullmatch(r"[a-z]+", word):
        return _EN.stemWord(word)
    return _RU.stemWord(word)


def content_stems(text: str) -> frozenset[str]:
    words = re.findall(r"\w+", normalize(text))
    return frozenset(_stem(w) for w in words if w not in STOP and (len(w) > 1 or w.isdigit()))


def extract_numbers(text: str) -> list[float]:
    t = normalize(text)
    # «17 100 000» → «17100000»
    t = re.sub(r"(?<=\d) (?=\d{3}\b)", "", t)
    nums = [float(m.replace(",", ".")) for m in NUM_RE.findall(t)]
    nums += [float(NUM_WORDS[w]) for w in re.findall(r"\w+", t) if w in NUM_WORDS]
    return nums


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_RE.search(answer))


def _numbers_match(answer: str, golds: list[str], rel_tol: float) -> bool:
    gold_nums = [n for g in golds for n in extract_numbers(g)]
    for a in extract_numbers(answer):
        for g in gold_nums:
            if g == a or (abs(g) > 0 and abs(a - g) / abs(g) <= rel_tol):
                return True
    return False


def _entity_match(answer: str, golds: list[str]) -> bool:
    ans = content_stems(answer)
    if not ans:
        return False
    for g in golds:
        gs = content_stems(g)
        if gs and (gs <= ans or ans <= gs):
            return True
    return False


def is_correct(answer: str, golds: list[str], kind: str, rel_tol: float = 0.02) -> bool:
    if not answer.strip() or is_refusal(answer):
        return False
    if kind == "year":
        years = {g for g in golds if re.fullmatch(r"-?\d{1,4}", g.strip())}
        found = set(re.findall(r"\b\d{3,4}\b", answer))
        if years & found:
            return True
        return _entity_match(answer, [g for g in golds if g not in years])
    if kind == "number":
        return _numbers_match(answer, golds, rel_tol) or _entity_match(answer, golds)
    return _entity_match(answer, golds) or _numbers_match(answer, golds, 0.0)


def answers_agree(a: str, b: str) -> bool:
    """Совпадают ли по смыслу два ответа модели (для самосогласованности)."""
    if is_refusal(a) or is_refusal(b):
        return is_refusal(a) and is_refusal(b)
    na, nb = extract_numbers(a), extract_numbers(b)
    if na and nb:
        return bool(set(na) & set(nb))
    sa, sb = content_stems(a), content_stems(b)
    return bool(sa) and bool(sb) and (sa <= sb or sb <= sa)
