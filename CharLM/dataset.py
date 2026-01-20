import random
import torch
from torch.utils.data import Dataset

from .utils import encode_str, choose_spans


class LexiconMLMDataset(Dataset):
    """
    Stage A: Lexicon MLM
    Span-masked masked language modeling на отдельных словах.
    """

    def __init__(self, words: list[str], c2i: dict, cfg: dict):
        self.words = words
        self.c2i = c2i
        self.cfg = cfg
        self.mask_id = c2i["<MASK>"]

        self.max_len = cfg["max_len"]
        self.mask_prob = cfg["mask_prob"]
        self.span_min = cfg["span_min"]
        self.span_max = cfg["span_max"]
        self.num_spans_min = cfg["num_spans_min"]
        self.num_spans_max = cfg["num_spans_max"]

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, idx: int):
        word = self.words[idx]

        ids = encode_str(word, self.c2i, self.max_len)
        word_len = min(len(word), self.max_len)

        mask_positions = choose_spans(
            word_len,
            self.span_min,
            self.span_max,
            self.num_spans_min,
            self.num_spans_max,
        )

        x = ids.copy()
        y = [-100] * self.max_len

        for pos in mask_positions:
            y[pos] = ids[pos]
            if random.random() < self.mask_prob:
                x[pos] = self.mask_id

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


class OCRMLMDataset(Dataset):

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        c2i: dict,
        max_len: int,
        mask_id: int,
        min_errors: int = 1,
    ):
        self.pairs = pairs
        self.c2i = c2i
        self.max_len = max_len
        self.mask_id = mask_id
        self.min_errors = min_errors

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inc, cor = self.pairs[idx]

        inc = inc[: self.max_len]
        cor = cor[: self.max_len]

        x = encode_str(inc, self.c2i, self.max_len)
        y = [-100] * self.max_len

        errors = 0
        for i, (ci, cc) in enumerate(zip(inc, cor)):
            if ci != cc:
                y[i] = self.c2i.get(cc, self.c2i["<UNK>"])
                x[i] = self.mask_id
                errors += 1

        if errors < self.min_errors:
            return self.__getitem__((idx + 1) % len(self))

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


class MixedMLMDataset(Dataset):
    def __init__(self, datasets: list[Dataset], sampling_probs: list[float]):
        assert len(datasets) == len(sampling_probs)
        self.datasets = datasets
        self.probs = sampling_probs

        self.lengths = [len(ds) for ds in datasets]
        self.total_len = sum(self.lengths)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        ds = random.choices(self.datasets, weights=self.probs, k=1)[0]
        j = random.randint(0, len(ds) - 1)
        return ds[j]
