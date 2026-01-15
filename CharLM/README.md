# CharLM

Character-level Embedding Model for OCR Correction using Metric Learning.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      CharLM Model                            │
├──────────────────────────────────────────────────────────────┤
│  Input: [B, T] character indices                             │
│                    ↓                                         │
│  ┌────────────────────────────────────┐                      │
│  │     Transformer Encoder (Shared)   │                      │
│  │  - Pre-LN, GELU activation         │                      │
│  │  - Learnable positional embeddings │                      │
│  │  - 6 layers, 8 heads (default)     │                      │
│  └────────────────────────────────────┘                      │
│                    ↓                                         │
│         Hidden states [B, T, D]                              │
│              ↓              ↓                                │
│  ┌─────────────────┐  ┌─────────────────┐                    │
│  │  Embedding Head │  │    MLM Head     │                    │
│  │     (MAIN)      │  │     (AUX)       │                    │
│  │                 │  │                 │                    │
│  │ Mean Pooling    │  │ Linear → Vocab  │                    │
│  │ Linear          │  │                 │                    │
│  │ LayerNorm       │  │                 │                    │
│  │ L2 Normalize    │  │                 │                    │
│  └─────────────────┘  └─────────────────┘                    │
│         ↓                    ↓                               │
│  Embedding [B, D]     Logits [B, T, V]                       │
└──────────────────────────────────────────────────────────────┘
```

## Training Strategy

### Stage A: Lexicon Embedding Pretraining
- **Input**: Clean words from lexicon
- **Anchor**: Clean or OCR-corrupted word
- **Positive**: Clean word
- **Negatives**: Random lexicon words + corrupted versions
- **Loss**: `metric_weight * metric_loss + mlm_weight * mlm_loss`
- **Objective**: Shape embedding space

### Stage B: Context-Aware Embedding
- **Input**: Windows of 1-3 words
- **Anchor**: Window with OCR noise in context
- **Positive**: Clean center word
- **Negatives**: In-batch negatives
- **Loss**: Metric + optional MLM

### Stage C: Supervised OCR Pairs (Main)
- **Input**: (incorrect → correct) pairs
- **Anchor**: Incorrect word (OCR error)
- **Positive**: Correct word
- **Negatives**: Hard negatives mined from lexicon
- **Loss**: `metric_loss + λ * mlm_loss`
- **Evaluation**: Recall@1, Recall@10

## Usage

```python
from CharLM import train, CharLM, DEFAULT_CONFIG

# Train with default config
model, (c2i, i2c, chars), exp_dir = train()

# Train with custom config
config = {
    "exp_dir": "exp_metric",
    "epochs_a": 20,
    "epochs_c": 30,
    "metric_loss_type": "infonce",  # or "triplet"
    "temperature": 0.07,
    "hard_neg_k": 15,
}
model, vocab, exp_dir = train(config)

# Use model for encoding
import torch
from CharLM import encode_batch

words = ["слово", "текст", "пример"]
input_ids = encode_batch(words, c2i, max_len=64).to(device)
embeddings = model.encode_words(input_ids)  # [3, 256], L2-normalized

# Similarity search
query_emb = model.encode_words(query_ids)
sim = torch.mm(query_emb, embeddings.T)  # cosine similarity
```

## Model API

```python
model = CharLM(vocab_size, emb_size=256, embed_dim=256, ...)

# Main methods
embeddings = model.encode_words(x)      # [B, D] L2-normalized
logits = model.forward_mlm(x)           # [B, T, V]
embeddings, logits = model(x)           # both

# Parameter groups (for separate LR)
encoder_params = model.get_encoder_params()
head_params = model.get_head_params()
```

## Configuration

Key parameters in `config.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `emb_size` | Encoder hidden size | 256 |
| `embed_dim` | Output embedding size | 256 |
| `n_layers` | Transformer layers | 6 |
| `metric_loss_type` | "infonce" or "triplet" | "infonce" |
| `temperature` | InfoNCE temperature | 0.07 |
| `triplet_margin` | Triplet loss margin | 0.3 |
| `hard_neg_k` | Hard negatives per sample | 10 |
| `lr_encoder_*` | Encoder learning rate | varies |
| `lr_embed_head_*` | Head learning rate | varies |

## OCR Augmentation

Applied to anchor words (not positives):
- Character substitution (visually similar)
- Character deletion
- Character insertion
- Adjacent character swap
- Space insertion
- Character duplication

Configurable via `p_char_*` parameters.

## Output Files

```
exp_dir/
├── model_a.pt      # Best Stage A checkpoint
├── model_b.pt      # Best Stage B checkpoint
├── model_c.pt      # Best Stage C checkpoint
├── model.pt        # Final model
├── vocab.json      # Character vocabulary
├── config.json     # Training config
└── train.log       # Training log
```
