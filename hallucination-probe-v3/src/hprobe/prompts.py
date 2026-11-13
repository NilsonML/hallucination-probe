"""Шаблоны запросов к модели.

Промпт — единственное «соприкосновение» с моделью. Если промпт меняется
в разных местах, прогоны становятся несравнимы. Здесь промпт собирается
в одном месте.

ChatML — родной формат Qwen2.5-Instruct. Модель на нём обучалась
следовать инструкциям, поэтому «неуверенность» в её скрытых состояниях
отражает именно этот сценарий.
"""

# ChatML-шаблон, используемый в тестах (test_pipeline.py)
CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

_SYSTEM = {
    "ru": "Ты отвечаешь на фактические вопросы. Отвечай максимально коротко.",
    "en": "You answer factual questions. Answer as briefly as possible.",
}

_FEWSHOT = {
    "ru": [
        ("Какая столица Франции?", "Париж"),
        ("Кто написал роман «Война и мир»?", "Лев Толстой"),
    ],
    "en": [
        ("What is the capital of France?", "Paris"),
        ("Who wrote the novel 'War and Peace'?", "Leo Tolstoy"),
    ],
}


def _messages(question: str, lang: str) -> list[dict]:
    """Собирает диалог для apply_chat_template."""
    msgs = [{"role": "system", "content": _SYSTEM[lang]}]
    for q, a in _FEWSHOT[lang]:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": question})
    return msgs


def build_prompt(tok, question: str, lang: str) -> str:
    """Промпт для обычного ответа (generate.py)."""
    return tok.apply_chat_template(
        _messages(question, lang), tokenize=False, add_generation_prompt=True
    )


def build_search_prompt(tok, question: str, paragraphs: list[str], lang: str) -> str:
    """Промпт с абзацем Википедии для адаптивного поиска (search.py).

    Абзац ставим ПОСЛЕ вопроса: модель сильнее опирается на информацию,
    расположенную ближе к позиции генерации ответа.
    """
    context = "\n\n".join(p for p in paragraphs if p)
    wrapped = f"{question}\n\nИспользуй информацию: {context}" if lang == "ru" else               f"{question}\n\nUse this information: {context}"
    return tok.apply_chat_template(
        _messages(wrapped, lang), tokenize=False, add_generation_prompt=True
    )


def clean_answer(text: str) -> str:
    """Убирает лишние пробелы и переносы из ответа модели."""
    return " ".join(text.split())
