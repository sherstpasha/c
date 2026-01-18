from typing import List


class CharVocab:
    def __init__(self, charset_path: str):
        self.tokens = []
        with open(charset_path, encoding="utf-8") as f:
            for line in f:
                tok = line.rstrip("\n")
                if tok:
                    self.tokens.append(tok)

        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        self.pad = self.token_to_id["<PAD>"]
        self.sos = self.token_to_id.get("<SOS>")
        self.eos = self.token_to_id.get("<EOS>")
        self.eow = self.token_to_id.get("<EOW>")
        self.sep = self.token_to_id.get("<SEP>")

        if self.eow is None:
            raise ValueError("В charset.txt должен быть <EOW>")

        self.special_ids = {
            i
            for i in [self.pad, self.sos, self.eos, self.eow, self.sep]
            if i is not None
        }

        self.mlm_replace_ids = [
            i for i in self.token_to_id.values() if i not in self.special_ids
        ]

        for ch in [" ", "-"]:
            if (
                ch in self.token_to_id
                and self.token_to_id[ch] not in self.mlm_replace_ids
            ):
                self.mlm_replace_ids.append(self.token_to_id[ch])

        self.digit = self.token_to_id.get("<DIGIT>")
        if self.digit is None:
            raise ValueError("В charset.txt должен быть <DIGIT>")

    def encode(self, text: str) -> List[int]:
        ids = []
        for ch in text:
            if ch.isdigit():
                ids.append(self.digit)
            elif ch in self.token_to_id:
                ids.append(self.token_to_id[ch])
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_token[i] for i in ids)
