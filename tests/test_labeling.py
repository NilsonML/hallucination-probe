from hprobe.labeling import answers_agree, extract_numbers, is_correct, is_refusal

STOWE = ["Гарриет Бичер-Стоу", "Бичер-Стоу Гарриет", "Harriet Beecher Stowe"]


def test_entity_full_and_partial_name():
    assert is_correct("Гарриет Бичер-Стоу", STOWE, "entity")
    assert is_correct("Бичер-Стоу", STOWE, "entity")
    assert is_correct("Роман написала Гарриет Бичер-Стоу.", STOWE, "entity")
    assert is_correct("Harriet Beecher Stowe", STOWE, "entity")
    assert not is_correct("Марк Твен", STOWE, "entity")


def test_entity_inflection_and_yo():
    assert is_correct("Во Франции", ["Франция"], "entity")
    assert is_correct("Пётр Первый", ["Петр Первый"], "entity")
    assert not is_correct("Москва", ["Московская область"], "entity")


def test_year():
    assert is_correct("В 1508 году", ["1508"], "year")
    assert not is_correct("1509", ["1508"], "year")


def test_number_tolerance_and_words():
    assert is_correct("17 100 000 км²", ["17125191"], "number")
    assert not is_correct("16 000 000 км²", ["17125191"], "number")
    assert is_correct("5", ["7", "пять"], "number")
    assert is_correct("семь", ["7"], "number")
    assert extract_numbers("5,45 мм") == [5.45]


def test_refusal_is_not_correct():
    assert is_refusal("Я не знаю.")
    assert not is_correct("Не знаю", ["Франция"], "entity")
    assert not is_correct("", ["Франция"], "entity")


def test_answers_agree():
    assert answers_agree("Бухарест", "Столица — Бухарест")
    assert not answers_agree("Бухарест", "Будапешт")
    assert answers_agree("1508", "в 1508 году")
