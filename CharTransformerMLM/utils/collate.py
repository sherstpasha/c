import torch


def collate_denoise(batch, pad_idx: int):
    xs, ys = zip(*batch)

    max_len = max(x.size(0) for x in xs)

    B = len(xs)
    x_pad = torch.full((B, max_len), pad_idx, dtype=torch.long)
    y_pad = torch.full((B, max_len), -100, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        L = x.size(0)
        x_pad[i, :L] = x
        y_pad[i, :L] = y

    attn_mask = x_pad == pad_idx

    return {
        "x": x_pad,
        "y": y_pad,
        "attn_mask": attn_mask,
    }
