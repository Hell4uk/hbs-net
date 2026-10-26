"""
Built-in synthetic Russian corpus used to train the BPE tokenizer when no
external corpus file is available. It reuses the same templates as the
contradiction dataset generator so the vocabulary covers the training data.
"""

from __future__ import annotations

import random

_ENTITIES = [
    "трава", "небо", "море", "лес", "дом", "кот", "собака", "река",
    "гора", "цветок", "камень", "дерево", "птица", "рыба", "облако",
    "звезда", "луна", "солнце", "ветер", "дорога", "машина", "книга",
    "стол", "стул", "окно", "дверь", "город", "село", "поле", "озеро",
]

_ATTRIBUTES = {
    "цвет": [
        "зелёная", "синяя", "красная", "жёлтая", "белая", "чёрная",
        "серая", "оранжевая", "фиолетовая", "коричневая",
    ],
    "размер": ["большая", "маленькая", "огромная", "крошечная", "средняя"],
    "состояние": ["мокрая", "сухая", "тёплая", "холодная", "чистая", "грязная"],
}

_TEMPLATES = [
    "{entity} была {attribute}.",
    "{entity} оказалась {attribute}.",
    "Свидетель утверждает: {entity} {attribute}.",
    "По словам очевидца, {entity} стала {attribute}.",
    "Известно, что {entity} всегда была {attribute}.",
    "В отчёте сказано, что {entity} {attribute}.",
    "Раньше {entity} была {attribute}.",
    "Сейчас {entity} {attribute}.",
]

_QUESTIONS = [
    "Какая {entity}?",
    "Какого цвета {entity}?",
    "Какого размера {entity}?",
    "В каком состоянии {entity}?",
    "Опиши {entity}.",
    "Что известно о {entity}?",
]

_ANSWER_PREFIXES = [
    "{entity} {attribute}.",
    "{entity} была {attribute}.",
    "Ответ: {entity} {attribute}.",
    "Известно, что {entity} {attribute}.",
]

_GENERIC = [
    "Модель обрабатывает последовательность свидетельств и обновляет своё состояние.",
    "Противоречие между двумя утверждениями повышает энергию системы.",
    "Синтез объединяет тезис и антитезис в новое согласованное знание.",
    "Диалектический гейт регулирует обновление внутреннего состояния модели.",
    "Состояние системы стремится к минимуму энергии при согласованности данных.",
    "Каждое новое свидетельство изменяет картину мира модели.",
    "Внимание позволяет точно извлекать информацию из длинного контекста.",
    "Рекуррентное состояние сжимает историю последовательности в вектор.",
    "Языковая модель предсказывает следующий токен на основе контекста.",
    "Обучение на противоречиях снижает количество галлюцинаций.",
    "Эти два утверждения не могут быть одновременно истинными.",
    "Модель должна указать на наличие противоречия, если оно есть.",
    "Новые данные могут опровергать предыдущие выводы.",
    "Селективность позволяет модели игнорировать нерелевантную информацию.",
    "Гибридная архитектура сочетает линейную сложность и точность извлечения.",
    "Токенайзер разбивает текст на последовательность субслов.",
    "Обучение продолжается до сходимости функции потерь.",
    "Валидационный набор используется для мониторинга качества.",
    "Кросс-энтропия измеряет расхождение предсказаний и факта.",
    "Индекс противоречия принимает значения от нуля до единицы.",
]


def build_corpus(n_sentences: int = 4000, seed: int = 7) -> list[str]:
    """Generate a deterministic synthetic Russian corpus for BPE training."""
    rng = random.Random(seed)
    sentences: list[str] = []
    for _ in range(n_sentences):
        kind = rng.random()
        if kind < 0.55:
            entity = rng.choice(_ENTITIES)
            attribute = rng.choice(
                rng.choice(list(_ATTRIBUTES.values()))
            )
            sentences.append(
                rng.choice(_TEMPLATES).format(entity=entity, attribute=attribute)
            )
        elif kind < 0.75:
            entity = rng.choice(_ENTITIES)
            sentences.append(rng.choice(_QUESTIONS).format(entity=entity))
        elif kind < 0.9:
            entity = rng.choice(_ENTITIES)
            attribute = rng.choice(rng.choice(list(_ATTRIBUTES.values())))
            sentences.append(
                rng.choice(_ANSWER_PREFIXES).format(entity=entity, attribute=attribute)
            )
        else:
            sentences.append(rng.choice(_GENERIC))
    return sentences
