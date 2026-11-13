"""hprobe — пробы на скрытых состояниях LLM для предсказания галлюцинаций.

Пакет устроен как конвейер:
    data.py      RuBQ 2.0 → data/questions.jsonl        (CPU, один раз)
    generate.py  вопросы  → ответы + скрытые состояния   (GPU, Colab)
    label.py     ответы   → labels.csv (верно/ошибка)    (CPU)
    probes.py    состояния → ROC-AUC по слоям            (CPU)
    search.py    проба + Википедия → адаптивный поиск    (GPU, В5)
"""
