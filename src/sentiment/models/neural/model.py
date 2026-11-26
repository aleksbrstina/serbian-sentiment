"""BiLSTM sentiment classifier."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sentiment.models.neural.tokenizer import PAD_ID

Pooling = Literal["last", "max", "mean", "max_mean"]


class BiLSTMClassifier(nn.Module):
    """Embedding -> dropout -> bidirectional LSTM -> pooling -> dropout -> linear.

    Pooling over time matters for long reviews: the final hidden state of an LSTM
    run over ~500 tokens is dominated by the last few sentences, while max pooling
    lets a strong cue anywhere in the review reach the classifier.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 300,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
        embedding_dropout: float = 0.2,
        pooling: Pooling = "max_mean",
        pretrained: np.ndarray | None = None,
        freeze_embeddings: bool = True,
    ) -> None:
        """Build the network.

        Args:
            vocab_size: Number of vocabulary entries (including specials).
            num_classes: Number of output classes.
            embedding_dim: Embedding size (300 for fastText).
            hidden_size: LSTM hidden size per direction.
            num_layers: Stacked LSTM layers.
            dropout: Dropout before the output layer (and between LSTM layers).
            embedding_dropout: Dropout applied to embeddings.
            pooling: How to turn the sequence of hidden states into one vector.
            pretrained: Optional ``(vocab_size, embedding_dim)`` initial weights.
            freeze_embeddings: Keep embedding weights fixed during training.
        """
        super().__init__()
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        if pretrained is not None:
            if pretrained.shape != (vocab_size, embedding_dim):
                raise ValueError(
                    f"Pretrained shape {pretrained.shape} != {(vocab_size, embedding_dim)}"
                )
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained))
        self.embedding.weight.requires_grad = not freeze_embeddings
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        pooled = 2 * hidden_size * (2 if pooling == "max_mean" else 1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(pooled, num_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            ids: Padded token ids, shape ``[B, T]``.
            lengths: True sequence lengths, shape ``[B]``.

        Returns:
            Logits of shape ``[B, num_classes]``.
        """
        emb = self.embedding_dropout(self.embedding(ids))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)

        if self.pooling == "last":
            # Final forward state and final backward state of the top layer.
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T', 2H]
            mask = (torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]).to(
                out.device
            )
            mask = mask.unsqueeze(-1)
            max_pool = out.masked_fill(~mask, float("-inf")).max(dim=1).values
            mean_pool = (out * mask).sum(dim=1) / lengths[:, None].to(out.dtype)
            pooled = {
                "max": max_pool,
                "mean": mean_pool,
                "max_mean": torch.cat([max_pool, mean_pool], dim=1),
            }[self.pooling]
        return self.output(self.dropout(pooled))

    def count_parameters(self) -> dict[str, int]:
        """Number of trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
