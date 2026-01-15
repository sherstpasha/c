"""Извлечение текстов из parquet и reports_json."""

import pandas as pd
import json
import re
import os

# Загружаем существующие данные
print("Загрузка существующих данных...")
with open('extracted_texts_cleaned.txt', 'r', encoding='utf-8') as f:
    old_text = f.read()
old_lines_count = len(old_text.strip().split('\n')) if old_text.strip() else 0

with open('all_words.txt', 'r', encoding='utf-8') as f:
    old_words = set(line.strip() for line in f if line.strip())
old_words_count = len(old_words)

print(f'Исходный extracted_texts_cleaned.txt: {old_lines_count} строк')
print(f'Исходный all_words.txt: {old_words_count} слов')

# 1. Извлекаем из parquet
print("\nИзвлечение из parquet...")
df = pd.read_parquet('0000 (1).parquet')
parquet_texts = df['russian_prereform'].tolist()
print(f'Из parquet: {len(parquet_texts)} текстов')

# 2. Извлекаем из reports_json
print("\nИзвлечение из reports_json...")
json_dir = 'reports_json'
json_texts = []
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
for i, fname in enumerate(json_files):
    if i % 1000 == 0:
        print(f"  Обработано {i}/{len(json_files)} файлов...")
    try:
        with open(os.path.join(json_dir, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'text_data' in data:
            for key, val in data['text_data'].items():
                if isinstance(val, str):
                    json_texts.append(val)
    except:
        pass
print(f'Из reports_json: {len(json_texts)} текстов')

# Объединяем все тексты
all_new_texts = parquet_texts + json_texts

# Очистка текста
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print("\nОчистка текстов...")
cleaned_texts = [clean_text(t) for t in all_new_texts if t and clean_text(t)]
print(f'После очистки: {len(cleaned_texts)} текстов')

# Добавляем к существующему файлу
print("\nДобавление в extracted_texts_cleaned.txt...")
with open('extracted_texts_cleaned.txt', 'a', encoding='utf-8') as f:
    for t in cleaned_texts:
        f.write(t + '\n')

print(f'Добавлено строк: {len(cleaned_texts)}')
print(f'Итого строк: {old_lines_count + len(cleaned_texts)}')

# Извлекаем слова
print("\nИзвлечение слов...")
word_pattern = re.compile(r'[А-Яа-яЁёІіѢѣѲѳЪъѴѵѪѫѬѭѮѯѰѱѠѡЅѕ]+', re.UNICODE)
new_words = set()
for t in cleaned_texts:
    words = word_pattern.findall(t)
    for w in words:
        if len(w) >= 2:
            new_words.add(w)

truly_new_words = new_words - old_words
print(f'Найдено уникальных слов: {len(new_words)}')
print(f'Новых слов (не было в all_words.txt): {len(truly_new_words)}')

# Добавляем новые слова
print("\nДобавление в all_words.txt...")
with open('all_words.txt', 'a', encoding='utf-8') as f:
    for w in sorted(truly_new_words):
        f.write(w + '\n')

print(f'\n=== ИТОГО ===')
print(f'extracted_texts_cleaned.txt: {old_lines_count} -> {old_lines_count + len(cleaned_texts)} строк (+{len(cleaned_texts)})')
print(f'all_words.txt: {old_words_count} -> {old_words_count + len(truly_new_words)} слов (+{len(truly_new_words)})')
print('Готово!')
