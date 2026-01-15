# CharLM

Символьная MLM-модель для коррекции OCR-ошибок (Transformer Encoder).

## Структура модуля

```
CharLM/
├── __init__.py    # Экспорт API
├── config.py      # Дефолтная конфигурация
├── model.py       # CharTransformerMLM
├── train.py       # Функция train(config)
└── utils.py       # Вспомогательные функции
```

## Использование

```python
from CharLM import train, DEFAULT_CONFIG

# Дефолтное обучение
model, vocab = train()

# Кастомная конфигурация
config = {
    "epochs_a": 10,
    "epochs_b": 5,
    "lr_a": 5e-4,
    "split_prob_a": 0.2,  # вероятность split-примеров
}
model, vocab = train(config)
```

## Особенности

- **Двухстадийное обучение**: pretrain на лексиконе + finetune на контексте
- **Split-примеры**: имитация разрыва слова пробелом (OCR-ошибка)
- **Span masking**: маскирование непрерывных участков
- **Логирование**: в консоль и файл `train_log.txt`
- **Без argparse**: вся конфигурация через словарь

## Конфигурация

См. `config.py` для полного списка параметров. Основные:

| Параметр | Описание | Дефолт |
|----------|----------|--------|
| `lexicon_path` | Путь к лексикону | `all_words.txt` |
| `text_path` | Путь к текстам | `extracted_texts_cleaned.txt` |
| `max_len` | Макс. длина последовательности | 64 |
| `epochs_a` | Эпохи Stage A | 30 |
| `epochs_b` | Эпохи Stage B | 24 |
| `split_prob_a` | Вер-ть split в Stage A | 0.15 |
| `split_prob_b` | Вер-ть split в Stage B | 0.15 |
