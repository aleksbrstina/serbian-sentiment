"""PyTorch Dataset and batching for the BiLSTM."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch.utils.data import Dataset

from sentiment.models.neural.tokenizer import PAD_ID, Vocabulary, tokenize

Truncation = Literal["head", "tail", "head_tail"]


def truncate(ids: list[int], max_len: int, strategy: Truncation = "head_tail") -> list[int]:
    """Shorten a token-id sequence to at most ``max_len`` ids.

    Args:
        ids: Token ids.
        max_len: Maximum length.
        strategy: ``"head"`` keeps the beginning, ``"tail"`` the end, and
            ``"head_tail"`` the first and last ``max_len / 2`` ids. Reviews often
            open with context and close with a verdict, so head+tail keeps both.

    Returns:
        Truncated ids (unchanged if already short enough).
    """
    if len(ids) <= max_len:
        return ids
    if strategy == "head":
        return ids[:max_len]
    if strategy == "tail":
        return ids[-max_len:]
    head = max_len // 2
    return ids[:head] + ids[len(ids) - (max_len - head) :]


class ReviewDataset(Dataset):
    """Reviews encoded once to id tensors; items are ``(ids, label_id)``."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[str] | None,
        vocab: Vocabulary,
        label_to_id: dict[str, int],
        max_len: int = 512,
        truncation: Truncation = "head_tail",
    ) -> None:
        """Tokenize, encode and truncate all texts up front.

        Args:
            texts: Input texts.
            labels: Gold labels, or None for inference (label id -1).
            vocab: Vocabulary used for encoding.
            label_to_id: Mapping label -> class index.
            max_len: Maximum sequence length in tokens.
            truncation: Truncation strategy, see :func:`truncate`.
        """
        self.items: list[torch.Tensor] = []
        for text in texts:
            ids = truncate(vocab.encode(tokenize(text)), max_len, truncation) or [PAD_ID]
            self.items.append(torch.tensor(ids, dtype=torch.long))
        if labels is None:
            self.labels = torch.full((len(texts),), -1, dtype=torch.long)
        else:
            self.labels = torch.tensor([label_to_id[lab] for lab in labels], dtype=torch.long)

    def __len__(self) -> int:
        """Number of examples."""
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(token_ids, label_id)`` for one example."""
        return self.items[index], self.labels[index]


def collate(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch to its longest sequence.

    Returns:
        ``(ids [B, T], lengths [B], labels [B])``.
    """
    seqs, labels = zip(*batch, strict=True)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    ids = torch.nn.utils.rnn.pad_sequence(list(seqs), batch_first=True, padding_value=PAD_ID)
    return ids, lengths, torch.stack(list(labels))
