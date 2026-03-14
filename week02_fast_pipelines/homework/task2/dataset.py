from typing import Optional
from collections import defaultdict
import random

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
from transformers import AutoTokenizer

MAX_LENGTH = 640
PAD_TOKEN_ID = 0


def _load_and_tokenize(data_path: str, max_length: int = MAX_LENGTH):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    sequences = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
            if len(token_ids) == 0 or len(token_ids) > max_length:
                continue
            sequences.append(token_ids)
    return sequences, tokenizer.vocab_size


class BrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.sequences, self.vocab_size = _load_and_tokenize(data_path, max_length)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        padded = seq + [PAD_TOKEN_ID] * (self.max_length - len(seq))
        input_ids = torch.tensor(padded[:-1], dtype=torch.long)
        targets = torch.tensor(padded[1:], dtype=torch.long)
        return input_ids, targets


class BigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.sequences, self.vocab_size = _load_and_tokenize(data_path, max_length)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long)


class UltraBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.sequences, self.vocab_size = _load_and_tokenize(data_path, max_length)
        self.lengths = [len(s) for s in self.sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long)


class UltraDuperBigBrainDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_length: int = MAX_LENGTH,
        packing_mode: str = "basic",
    ):
        self.max_length = max_length
        sequences, self.vocab_size = _load_and_tokenize(data_path, max_length)

        if packing_mode == "basic":
            self.packed = self._basic_packing(sequences, max_length)
        elif packing_mode == "ffd":
            self.packed = self._ffd_packing(sequences, max_length)
        elif packing_mode == "obfd":
            self.packed = self._obfd_packing(sequences, max_length)
        else:
            raise ValueError(f"Unknown packing mode: {packing_mode}")

    def __len__(self):
        return len(self.packed)

    def __getitem__(self, idx: int):
        input_ids, attention_mask = self.packed[idx]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    @staticmethod
    def _basic_packing(sequences, max_length):
        packed = []
        current_ids = []
        current_seq_id = 0
        current_mask = []

        for seq in sequences:
            if len(current_ids) + len(seq) <= max_length:
                current_mask.extend([current_seq_id] * len(seq))
                current_ids.extend(seq)
                current_seq_id += 1
            else:
                if current_ids:
                    pad_len = max_length - len(current_ids)
                    current_mask.extend([-1] * pad_len)
                    current_ids.extend([PAD_TOKEN_ID] * pad_len)
                    packed.append((current_ids, current_mask))
                current_ids = list(seq)
                current_mask = [0] * len(seq)
                current_seq_id = 1

        if current_ids:
            pad_len = max_length - len(current_ids)
            current_mask.extend([-1] * pad_len)
            current_ids.extend([PAD_TOKEN_ID] * pad_len)
            packed.append((current_ids, current_mask))

        return packed

    @staticmethod
    def _ffd_packing(sequences, max_length):
        sorted_seqs = sorted(
            [(len(s), s) for s in sequences if len(s) <= max_length],
            key=lambda x: x[0],
            reverse=True,
        )

        bins_ids = []
        bins_masks = []
        bins_remaining = []
        bins_seq_count = []

        for seq_len, seq in sorted_seqs:
            placed = False
            for i in range(len(bins_ids)):
                if bins_remaining[i] >= seq_len:
                    bins_masks[i].extend([bins_seq_count[i]] * seq_len)
                    bins_ids[i].extend(seq)
                    bins_remaining[i] -= seq_len
                    bins_seq_count[i] += 1
                    placed = True
                    break
            if not placed:
                bins_ids.append(list(seq))
                bins_masks.append([0] * seq_len)
                bins_remaining.append(max_length - seq_len)
                bins_seq_count.append(1)

        packed = []
        for ids, mask in zip(bins_ids, bins_masks):
            pad_len = max_length - len(ids)
            mask.extend([-1] * pad_len)
            ids.extend([PAD_TOKEN_ID] * pad_len)
            packed.append((ids, mask))

        return packed

    @staticmethod
    def _obfd_packing(sequences, max_length):
        sorted_seqs = sorted(
            [(len(s), s) for s in sequences if len(s) <= max_length],
            key=lambda x: x[0],
            reverse=True,
        )

        bins_ids = []
        bins_masks = []
        bins_remaining = []
        bins_seq_count = []

        size = max_length + 1
        tree_size = 1
        while tree_size < size:
            tree_size *= 2

        tree = [False] * (2 * tree_size)
        capacity_bins = defaultdict(list)

        def update(pos):
            pos += tree_size
            tree[pos] = len(capacity_bins[pos - tree_size]) > 0
            pos >>= 1
            while pos >= 1:
                tree[pos] = tree[2 * pos] or tree[2 * pos + 1]
                pos >>= 1

        def query_min_ge(lo):
            return _query(1, 0, tree_size, lo, size)

        def _query(node, node_lo, node_hi, lo, hi):
            if not tree[node]:
                return -1
            if node_lo >= hi or node_hi <= lo:
                return -1
            if node_hi - node_lo == 1:
                return node_lo
            mid = (node_lo + node_hi) // 2
            left = _query(2 * node, node_lo, mid, lo, hi)
            if left != -1:
                return left
            return _query(2 * node + 1, mid, node_hi, lo, hi)

        for seq_len, seq in sorted_seqs:
            cap = query_min_ge(seq_len)
            if cap != -1 and capacity_bins[cap]:
                bin_idx = capacity_bins[cap].pop()
                if not capacity_bins[cap]:
                    update(cap)
                new_cap = cap - seq_len
                bins_masks[bin_idx].extend([bins_seq_count[bin_idx]] * seq_len)
                bins_ids[bin_idx].extend(seq)
                bins_remaining[bin_idx] = new_cap
                bins_seq_count[bin_idx] += 1
                capacity_bins[new_cap].append(bin_idx)
                update(new_cap)
            else:
                bin_idx = len(bins_ids)
                bins_ids.append(list(seq))
                bins_masks.append([0] * seq_len)
                bins_remaining.append(max_length - seq_len)
                bins_seq_count.append(1)
                new_cap = max_length - seq_len
                capacity_bins[new_cap].append(bin_idx)
                update(new_cap)

        packed = []
        for ids, mask in zip(bins_ids, bins_masks):
            pad_len = max_length - len(ids)
            mask.extend([-1] * pad_len)
            ids.extend([PAD_TOKEN_ID] * pad_len)
            packed.append((ids, mask))

        return packed


def collate_fn(
    batch: list, max_length: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_length is not None:
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        return inputs, targets

    lengths = [len(seq) for seq in batch]
    max_len = max(lengths)
    input_ids = []
    targets = []
    for seq in batch:
        pad_len = max_len - len(seq)
        padded = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        input_ids.append(padded[:-1])
        targets.append(padded[1:])
    return torch.stack(input_ids), torch.stack(targets)


def collate_fn_packed(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_masks = torch.stack([item[1] for item in batch])
    return input_ids, attention_masks


class UltraBigBrainBatchSampler(Sampler):
    def __init__(self, dataset: UltraBigBrainDataset, batch_size: int, k: int = 1):
        self.batch_size = batch_size
        self.k = k
        self.length_to_indices = defaultdict(list)
        for idx, length in enumerate(dataset.lengths):
            self.length_to_indices[length].append(idx)

        sorted_lengths = sorted(self.length_to_indices.keys())
        self.buckets = []
        if sorted_lengths:
            current_bucket = []
            bucket_min = sorted_lengths[0]
            for length in sorted_lengths:
                if length - bucket_min <= k:
                    current_bucket.extend(self.length_to_indices[length])
                else:
                    if current_bucket:
                        self.buckets.append(current_bucket)
                    current_bucket = list(self.length_to_indices[length])
                    bucket_min = length
            if current_bucket:
                self.buckets.append(current_bucket)

        self._num_batches = sum(
            (len(bucket) + batch_size - 1) // batch_size for bucket in self.buckets
        )

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        all_batches = []
        for bucket in self.buckets:
            indices = list(bucket)
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                all_batches.append(indices[i : i + self.batch_size])
        random.shuffle(all_batches)
        yield from all_batches
