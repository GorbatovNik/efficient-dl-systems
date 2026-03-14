from enum import Enum
from functools import partial
import time
import statistics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer import PositionalEncoding, generate_square_subsequent_mask
from dataset import (
    MAX_LENGTH,
    BrainDataset,
    BigBrainDataset,
    UltraBigBrainDataset,
    UltraDuperBigBrainDataset,
    UltraBigBrainBatchSampler,
    collate_fn,
    collate_fn_packed,
)


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4


class GPT2Model(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 1024, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        if input_ids.dim() == 2 and input_ids.shape[0] != input_ids.shape[1]:
            x = input_ids.transpose(0, 1)
        else:
            x = input_ids.transpose(0, 1)

        seq_len = x.shape[0]

        x = self.embedding(x) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)

        causal_mask = generate_square_subsequent_mask(seq_len).to(x.device)

        if attention_mask is not None:
            attn_mask = self._build_packed_mask(attention_mask, seq_len, x.device)
            combined_mask = causal_mask.unsqueeze(0) + attn_mask
            nhead = 8
            combined_mask = combined_mask.repeat(nhead, 1, 1)
            out = self.decoder(x, x, tgt_mask=combined_mask)
        else:
            out = self.decoder(x, x, tgt_mask=causal_mask)

        return self.output_proj(out)

    @staticmethod
    def _build_packed_mask(segment_ids, seq_len, device):
        seg_i = segment_ids.unsqueeze(2)
        seg_j = segment_ids.unsqueeze(1)
        same_segment = (seg_i == seg_j) & (seg_i >= 0) & (seg_j >= 0)
        mask = torch.where(same_segment, 0.0, float("-inf"))
        return mask


def get_gpt2_model(vocab_size: int = 30522) -> nn.Module:
    return GPT2Model(vocab_size=vocab_size, d_model=1024, nhead=8)


def run_epoch(
    data_mode: DataMode,
    data_path: str,
    batch_size: int = 32,
    k: int = 1,
    packing_mode: str = "basic",
    device: str = "cuda",
    num_warmup: int = 5,
) -> dict:
    if data_mode == DataMode.BRAIN:
        dataset = BrainDataset(data_path)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, max_length=MAX_LENGTH),
            num_workers=4,
            pin_memory=True,
        )
    elif data_mode == DataMode.BIG_BRAIN:
        dataset = BigBrainDataset(data_path)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, max_length=None),
            num_workers=4,
            pin_memory=True,
        )
    elif data_mode == DataMode.ULTRA_BIG_BRAIN:
        dataset = UltraBigBrainDataset(data_path)
        sampler = UltraBigBrainBatchSampler(dataset, batch_size=batch_size, k=k)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=partial(collate_fn, max_length=None),
            num_workers=4,
            pin_memory=True,
        )
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        dataset = UltraDuperBigBrainDataset(data_path, packing_mode=packing_mode)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_packed,
            num_workers=4,
            pin_memory=True,
        )
    else:
        raise ValueError(f"Unknown data mode: {data_mode}")

    vocab_size = dataset.vocab_size if hasattr(dataset, "vocab_size") else 30522
    model = get_gpt2_model(vocab_size=vocab_size).to(device)
    model.eval()

    batch_times = []
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
                input_ids, attention_mask = batch_data
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                model(input_ids[:, :-1], attention_mask[:, :-1])
                if device == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
            else:
                input_ids, targets = batch_data
                input_ids = input_ids.to(device)

                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                model(input_ids)
                if device == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()

            if i >= num_warmup:
                batch_times.append(end - start)

    if not batch_times:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "num_batches": 0}

    return {
        "min": min(batch_times),
        "max": max(batch_times),
        "mean": statistics.mean(batch_times),
        "median": statistics.median(batch_times),
        "num_batches": len(batch_times),
    }


if __name__ == "__main__":
    import sys
    import pandas as pd

    data_path = sys.argv[1] if len(sys.argv) > 1 else "wikitext-103-raw/wiki.train.raw"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32

    results = []

    print("Running BRAIN mode...")
    r = run_epoch(DataMode.BRAIN, data_path, batch_size=batch_size, device=device)
    results.append({"mode": "BRAIN", **r})

    print("Running BIG_BRAIN mode...")
    r = run_epoch(DataMode.BIG_BRAIN, data_path, batch_size=batch_size, device=device)
    results.append({"mode": "BIG_BRAIN", **r})

    for k_val in [1, 5, 10, 20, 50]:
        print(f"Running ULTRA_BIG_BRAIN mode (k={k_val})...")
        r = run_epoch(
            DataMode.ULTRA_BIG_BRAIN, data_path, batch_size=batch_size, k=k_val, device=device
        )
        results.append({"mode": f"ULTRA_BIG_BRAIN (k={k_val})", **r})

    for pack_mode in ["basic", "ffd", "obfd"]:
        print(f"Running ULTRA_DUPER_BIG_BRAIN mode ({pack_mode})...")
        r = run_epoch(
            DataMode.ULTRA_DUPER_BIG_BRAIN,
            data_path,
            batch_size=batch_size,
            packing_mode=pack_mode,
            device=device,
        )
        results.append({"mode": f"ULTRA_DUPER_BIG_BRAIN ({pack_mode})", **r})

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80)
    print(df.to_string(index=False))
