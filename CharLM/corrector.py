# corrector.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import faiss

from vocab import CharVocab
from model import CharTransformerMLM


# ============================================================
# Edit distance (CER / WER)
# ============================================================


def _levenshtein(a: List, b: List) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def cer(pred_text: str, gt_text: str) -> float:
    a = list(pred_text)
    b = list(gt_text)
    return _levenshtein(a, b) / max(1, len(b))


def wer(pred_tokens: List[str], gt_tokens: List[str]) -> float:
    return _levenshtein(pred_tokens, gt_tokens) / max(1, len(gt_tokens))


# ============================================================
# CSV loader
# ============================================================


@dataclass
class PairRow:
    image: str
    incorrect: str
    correct: str


def load_pairs_csv(path: str) -> List[PairRow]:
    rows: List[PairRow] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                PairRow(
                    image=r.get("image", ""),
                    incorrect=(r.get("incorrect", "") or "").strip(),
                    correct=(r.get("correct", "") or "").strip(),
                )
            )
    return rows


# ============================================================
# Token extraction (word inside raw token)
# ============================================================

# буквенная “ядровая” часть: русские + дореформенные буквы + ё
# ВАЖНО: не включаем цифры/тире/точки
WORD_CORE_RE = re.compile(r"[А-Яа-яЁёѢѣѲѳІіѴѵ]+", re.UNICODE)


def split_token_keep_punct(token: str) -> Optional[Tuple[str, str, str]]:
    """
    Находит ПЕРВУЮ буквенную подстроку внутри token и возвращает:
      (prefix, core_word, suffix)

    Примеры:
      "Губернаторъ," -> ("", "Губернаторъ", ",")
      "Подписалъ:"   -> ("", "Подписалъ", ":")
      "\"благопосично,\"" -> ("\"", "благопосично", ",\"")
      "1883" -> None
      "1884 года" -> ("1884 ", "года", "")
    """
    m = WORD_CORE_RE.search(token)
    if not m:
        return None
    a, b = m.span()
    return token[:a], token[a:b], token[b:]


def merge_token(prefix: str, core: str, suffix: str) -> str:
    return f"{prefix}{core}{suffix}"


# ============================================================
# Corrector
# ============================================================


class FaissRerankCorrector:
    """
    Corrector over lexicon with FAISS + reranker.

    IMPORTANT:
      - Corrects only the first "word core" inside a raw token,
        keeping punctuation/numbers/spaces around it intact.
      - Skips tokens with no letters.
    """

    def __init__(
        self,
        model: CharTransformerMLM,
        vocab: CharVocab,
        lexicon_words: List[str],
        device: torch.device,
        max_len: int,
        top_k: int = 10,
        sim_threshold: float = 0.75,
        faiss_batch_size: int = 4096,
    ):
        self.model = model
        self.vocab = vocab
        self.lexicon_words = [w for w in lexicon_words if w]
        self.device = device
        self.max_len = int(max_len)
        self.top_k = int(top_k)
        self.sim_threshold = float(sim_threshold)
        self.faiss_batch_size = int(faiss_batch_size)

        self.model.eval()

        self.embs = self._build_lexicon_embeddings()
        self.index = self._build_faiss_index(self.embs)

    # ----------------------------
    # Embeddings / FAISS
    # ----------------------------

    @torch.no_grad()
    def _encode_words_to_embs(self, words: List[str]) -> torch.Tensor:
        enc = [self.vocab.encode(w)[: self.max_len] for w in words]
        if not enc:
            d = int(self.model.emb.embedding_dim)
            return torch.empty((0, d), dtype=torch.float32)

        T = max(len(x) for x in enc)
        x = torch.full(
            (len(enc), T), self.vocab.pad, dtype=torch.long, device=self.device
        )
        for i, ids in enumerate(enc):
            if ids:
                x[i, : len(ids)] = torch.tensor(ids, device=self.device)

        h = self.model.encode(x)  # [B, T, D]
        mask = (x != self.vocab.pad).float()  # [B, T]
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        emb = (h * mask.unsqueeze(-1)).sum(dim=1) / denom  # masked mean
        return emb.cpu()

    @torch.no_grad()
    def _build_lexicon_embeddings(self) -> torch.Tensor:
        out = []
        for i in range(0, len(self.lexicon_words), self.faiss_batch_size):
            out.append(
                self._encode_words_to_embs(
                    self.lexicon_words[i : i + self.faiss_batch_size]
                )
            )
        if not out:
            d = int(self.model.emb.embedding_dim)
            return torch.empty((0, d), dtype=torch.float32)
        return torch.cat(out, dim=0)

    @staticmethod
    def _build_faiss_index(embs: torch.Tensor) -> faiss.Index:
        x = embs.numpy().astype("float32")
        faiss.normalize_L2(x)
        index = faiss.IndexFlatIP(x.shape[1])
        index.add(x)
        return index

    @torch.no_grad()
    def retrieve(self, word: str) -> Tuple[List[str], List[float]]:
        emb = self._encode_words_to_embs([word]).numpy().astype("float32")
        faiss.normalize_L2(emb)
        sims, idxs = self.index.search(emb, self.top_k)
        return [self.lexicon_words[i] for i in idxs[0]], sims[0].tolist()

    # ----------------------------
    # Rerank
    # ----------------------------

    def _make_seq(self, left: str, cand: str, right: str) -> torch.Tensor:
        # контекст на уровне "слов", но мы кодим как строку
        seq = " ".join([x for x in [left, cand, right] if x])
        ids = self.vocab.encode(seq)[: self.max_len]
        out = torch.full((self.max_len,), self.vocab.pad, dtype=torch.long)
        out[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    @torch.no_grad()
    def rerank_best(
        self,
        left: str,
        right: str,
        cands: List[str],
        sims: List[float],
    ) -> Tuple[str, float, float]:
        if not cands:
            return "", float("-inf"), 0.0

        x = torch.stack([self._make_seq(left, c, right) for c in cands]).to(self.device)
        scores = self.model.score(x).detach().cpu().numpy()
        i = int(scores.argmax())
        return cands[i], float(scores[i]), float(sims[i])

    # ----------------------------
    # Public API
    # ----------------------------

    @torch.no_grad()
    def correct_token(
        self,
        raw_token: str,
        left_token: str = "",
        right_token: str = "",
    ) -> Tuple[str, Dict]:
        """
        Исправляет raw_token, сохраняя пунктуацию/цифры вокруг.
        Возвращает (new_token, debug).
        """
        split = split_token_keep_punct(raw_token)
        if split is None:
            # нечего исправлять
            return raw_token, {
                "eligible": False,
                "raw": raw_token,
                "reason": "no_letters",
            }

        pre, core, suf = split

        # контекст тоже берём как raw, но для rerank лучше вытаскивать core:
        l_split = split_token_keep_punct(left_token)
        r_split = split_token_keep_punct(right_token)
        left_core = l_split[1] if l_split else ""
        right_core = r_split[1] if r_split else ""

        cands, sims = self.retrieve(core)
        best, score, sim = self.rerank_best(left_core, right_core, cands, sims)

        # применяем только если:
        #  - лучший кандидат отличается от core
        #  - и проходит по порогу similarity
        changed_core = (best != core) and (sim >= self.sim_threshold)
        new_core = best if changed_core else core
        out = merge_token(pre, new_core, suf)

        return out, {
            "eligible": True,
            "raw": raw_token,
            "core": core,
            "best": best,
            "sim": sim,
            "score": score,
            "changed_core": changed_core,
            "out": out,
        }

    @torch.no_grad()
    def evaluate_on_pairs_csv(
        self,
        pairs_csv: str,
        max_items: Optional[int] = None,
    ) -> Dict[str, float]:
        rows = load_pairs_csv(pairs_csv)

        if max_items is not None:
            rows = rows[:max_items]

        pred_tokens: List[str] = []
        gt_tokens: List[str] = [r.correct for r in rows]

        total_items = len(rows)
        eligible = 0
        replacements = 0
        useful_repl = 0

        for i, r in enumerate(rows):
            left = rows[i - 1].incorrect if i > 0 else ""
            right = rows[i + 1].incorrect if i + 1 < len(rows) else ""

            pred, dbg = self.correct_token(r.incorrect, left, right)
            pred_tokens.append(pred)

            if dbg.get("eligible", False):
                eligible += 1

            # replacement = ИТОГОВЫЙ ТОКЕН реально изменился
            if pred != r.incorrect:
                replacements += 1
                if pred == r.correct:
                    useful_repl += 1

        return {
            "total_items": len(rows),
            # сколько раз модель вообще решила заменить
            "changed": replacements,
            # сколько замен совпали с GT
            "improved": useful_repl,
            # ключевая метрика
            "efficiency": useful_repl / replacements if replacements else 0.0,
            # quality metrics
            "cer": cer(" ".join(pred_tokens), " ".join(gt_tokens)),
            "wer": wer(pred_tokens, gt_tokens),
        }
