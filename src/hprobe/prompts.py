"""Шаблоны запроса к модели. Несколько примеров в диалоге задают формат короткого ответа,
чтобы ответ можно было автоматически сравнить с эталоном."""

SYSTEM = {
    "ru": "Ты отвечаешь на вопросы викторины. Отвечай кратко: только ответ, без пояснений.",
    "en": "You answer trivia questions. Answer briefly: only the answer, no explanations.",
}

FEW_SHOT = {
    "ru": [
        ("Какая столица Франции?", "Париж"),
        ("Кто написал роман «Война и мир»?", "Лев Толстой"),
        ("В каком году человек впервые полетел в космос?", "1961"),
    ],
    "en": [
        ("What is the capital of France?", "Paris"),
        ("Who wrote the novel War and Peace?", "Leo Tolstoy"),
        ("In what year did a human first fly into space?", "1961"),
    ],
}


def build_messages(question: str, lang: str = "ru") -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM[lang]}]
    for q, a in FEW_SHOT[lang]:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": question})
    return messages


def build_prompt(tokenizer, question: str, lang: str = "ru") -> str:
    return tokenizer.apply_chat_template(
        build_messages(question, lang), tokenize=False, add_generation_prompt=True)


def build_search_prompt(tokenizer, question: str, paragraphs: list[str], lang: str = "ru") -> str:
    """Промпт с добавлением релевантных абзацев Википедии перед вопросом."""
    context = "\n\n".join(paragraphs)
    user = f"Используй контекст:\n{context}\n\nВопрос: {question}" if lang == "ru" \
        else f"Use the context:\n{context}\n\nQuestion: {question}"
    messages = [{"role": "system", "content": SYSTEM[lang]}]
    for q, a in FEW_SHOT[lang]:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def clean_answer(text: str) -> str:
    """Первая непустая строка ответа без лишних символов."""
    for line in text.strip().splitlines():
        line = line.strip().strip("*").strip()
        if line:
            return line
    return ""
