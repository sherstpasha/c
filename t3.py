from collections import defaultdict, Counter

DICT_PATH = "all_words_with_gt.txt"

MIN_STEM_LEN = 3
MAX_END_LEN = 6
MIN_STEM_FREQ = 10  # порог "настоящего" stem

with open(DICT_PATH, encoding="utf-8") as f:
    words = [w.strip().lower() for w in f if w.strip()]

stem_freq = Counter()
stem2endings = defaultdict(Counter)

for w in words:
    L = len(w)
    for k in range(1, min(MAX_END_LEN, L)):
        stem = w[:-k]
        ending = w[-k:]

        if len(stem) < MIN_STEM_LEN:
            continue

        stem_freq[stem] += 1
        stem2endings[stem][ending] += 1

# фильтруем мусор
stem2endings = {
    stem: endings
    for stem, endings in stem2endings.items()
    if stem_freq[stem] >= MIN_STEM_FREQ
}

print(f"Total stems: {len(stem2endings)}")

# топ stem по числу окончаний
for stem, endings in sorted(
    stem2endings.items(), key=lambda x: len(x[1]), reverse=True
)[:20]:
    print(stem, "→", list(endings.most_common(10)))

ending_freq = Counter()

for endings in stem2endings.values():
    for e, c in endings.items():
        ending_freq[e] += c

for e, c in ending_freq.most_common(50):
    print(e, c)
