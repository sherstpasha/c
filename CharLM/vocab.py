from typing import List


class CharVocab:
    """
    Символьный словарь для char-level MLM (Stage A).

    Индексы фиксированы:
    0 -> <PAD>
    1 -> <MASK>
    2 -> <UNK>
    далее -> символы из charset.txt (кроме <...>)
    """

    PAD_TOKEN = "<PAD>"
    MASK_TOKEN = "<MASK>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, charset_path: str):
        self.token_to_id = {}
        self.id_to_token = {}

        tokens = [
            self.PAD_TOKEN,
            self.MASK_TOKEN,
            self.UNK_TOKEN,
        ]

        with open(charset_path, encoding="utf-8") as f:
            for line in f:
                tok = line.rstrip("\n\r")
                if not tok:
                    continue

                if tok.startswith("<") and tok.endswith(">"):
                    continue

                # защита от дубликатов
                if tok in tokens:
                    continue

                tokens.append(tok)

        for idx, tok in enumerate(tokens):
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok

        self.pad = self.token_to_id[self.PAD_TOKEN]
        self.mask = self.token_to_id[self.MASK_TOKEN]
        self.unk = self.token_to_id[self.UNK_TOKEN]

    def encode(self, text: str) -> List[int]:
        """
        Кодировать строку в список индексов.
        """
        return [self.token_to_id.get(ch, self.unk) for ch in text]

    def decode(self, ids: List[int]) -> str:
        """
        Декодировать список индексов в строку (без PAD).
        """
        chars = []
        for i in ids:
            if i == self.pad:
                continue
            chars.append(self.id_to_token.get(i, self.UNK_TOKEN))
        return "".join(chars)

    def __len__(self) -> int:
        return len(self.token_to_id)
