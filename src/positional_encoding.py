"""
Sinusoidal positional encoding for Transformer models.

Self-attention does not know token order by default.
This module injects order information by adding a deterministic
position-dependent vector to each token embedding.

Input shape:  (batch_size, seq_len, d_model)
Output shape: (batch_size, seq_len, d_model)
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        base: float = 10000.0,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if max_len <= 0:
            raise ValueError("max_len must be positive.")

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        dimensions = torch.arange(0, d_model, 2, dtype=torch.float32)

        frequency_scale = torch.exp(-math.log(base) * dimensions / d_model)

        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequency_scale)
        encoding[:, 1::2] = torch.cos(
            positions * frequency_scale[: encoding[:, 1::2].shape[1]]
        )

        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        if token_embeddings.ndim != 3:
            raise ValueError(
                "Expected token_embeddings shape: "
                "(batch_size, seq_len, d_model)"
            )

        _, seq_len, d_model = token_embeddings.shape

        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}.")

        if seq_len > self.max_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_len={self.max_len}.")

        positional_slice = self.encoding[:, :seq_len, :]
        return self.dropout(token_embeddings + positional_slice)


if __name__ == "__main__":
    x = torch.zeros(2, 6, 8)

    pos_enc = SinusoidalPositionalEncoding(
        d_model=8,
        max_len=100,
        dropout=0.0,
    )

    y = pos_enc(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("Position 0 :", y[0, 0])