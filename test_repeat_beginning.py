"""Тест для демонстрации аугментаций REPEAT BEGINNING и REPEAT ENDING"""

from CharTransformerMLM.vocab import CharVocab
from CharTransformerMLM.model import EditVocab
from CharTransformerMLM.dataset import CharOCREditDataset
from CharTransformerMLM.utils.collate import collate_edit

# Инициализация
vocab = CharVocab("data/charset.txt")
edit_vocab = EditVocab(vocab)

dataset = CharOCREditDataset(
    text_path="data/extracted_texts_cleaned.txt",
    vocab=vocab,
    edit_vocab=edit_vocab,
    pairs_csv_path=None,  # без real OCR
    words_path=None,
    max_len=128,
    max_words=8,
)


def decode_ids(ids, vocab):
    """Преобразует ID в текст с разделением по словам"""
    result = []
    current_word = []
    for id in ids:
        if id == vocab.eow:
            if current_word:
                result.append("".join(current_word))
            current_word = []
        else:
            current_word.append(vocab.id_to_token.get(id, "?"))
    if current_word:
        result.append("".join(current_word))
    return result


print("=" * 80)
print("REPEAT BEGINNING - примеры (2 варианта)")
print("=" * 80)

for i in range(10):
    x, y = dataset.make_repeat_beginning()
    x_ids = x.tolist()
    y_ids = y.tolist()

    noisy_words = decode_ids(x_ids, vocab)
    # Восстанавливаем target применяя edit операции
    target_ids = []
    for xid, yid in zip(x_ids, y_ids):
        if edit_vocab.id_to_op[yid] == "COPY":
            target_ids.append(xid)
        elif edit_vocab.id_to_op[yid] == "DELETE":
            continue
        elif edit_vocab.id_to_op[yid].startswith("REPLACE_"):
            ch = edit_vocab.id_to_op[yid].replace("REPLACE_", "")
            target_ids.append(vocab.token_to_id[ch])
        elif edit_vocab.id_to_op[yid].startswith("INSERT_"):
            ch = edit_vocab.id_to_op[yid].replace("INSERT_", "")
            target_ids.append(vocab.token_to_id[ch])
            target_ids.append(xid)

    target_words = decode_ids(target_ids, vocab)

    # Сравниваем
    if len(noisy_words) > len(target_words):
        variant = "отдельное слово"
    elif any(len(n) > len(t) for n, t in zip(noisy_words, target_words) if n and t):
        variant = "внутри слова"
    else:
        continue

    print(f"\n--- Пример {i + 1} ({variant}) ---")
    print(f"NOISY:  {' | '.join(noisy_words)}")
    print(f"TARGET: {' | '.join(target_words)}")

print("\n" + "=" * 80)
print("REPEAT ENDING - примеры (2 варианта)")
print("=" * 80)

for i in range(10):
    x, y = dataset.make_repeat_ending()
    x_ids = x.tolist()
    y_ids = y.tolist()

    noisy_words = decode_ids(x_ids, vocab)
    target_ids = []
    for xid, yid in zip(x_ids, y_ids):
        if edit_vocab.id_to_op[yid] == "COPY":
            target_ids.append(xid)
        elif edit_vocab.id_to_op[yid] == "DELETE":
            continue
        elif edit_vocab.id_to_op[yid].startswith("REPLACE_"):
            ch = edit_vocab.id_to_op[yid].replace("REPLACE_", "")
            target_ids.append(vocab.token_to_id[ch])
        elif edit_vocab.id_to_op[yid].startswith("INSERT_"):
            ch = edit_vocab.id_to_op[yid].replace("INSERT_", "")
            target_ids.append(vocab.token_to_id[ch])
            target_ids.append(xid)

    target_words = decode_ids(target_ids, vocab)

    # Сравниваем
    if len(noisy_words) > len(target_words):
        variant = "отдельное слово"
    elif any(len(n) > len(t) for n, t in zip(noisy_words, target_words) if n and t):
        variant = "внутри слова"
    else:
        continue

    print(f"\n--- Пример {i + 1} ({variant}) ---")
    print(f"NOISY:  {' | '.join(noisy_words)}")
    print(f"TARGET: {' | '.join(target_words)}")
