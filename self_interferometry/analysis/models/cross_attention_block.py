#!/usr/bin/env python3

import torch
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention


class CrossAttentionBlock(nn.Module):
    """Cross attention block with banded masking for multi-channel time series.

    This block applies cross-attention between channels using a banded mask that
    restricts attention to a local temporal window (atten_len samples). This allows
    channels to exchange information while maintaining temporal locality.
    """

    def __init__(
        self,
        n_channels: int,
        embed_dim: int,
        atten_len: int,
        num_heads: int,
        dropout: float,
        chunk_size: int = 2048,
    ):
        """Initialize cross attention block.

        Args:
            n_channels: Number of input channels
            embed_dim: Embedding dimension (channels per feature)
            atten_len: Band length for attention mask (temporal window size)
            num_heads: Number of attention heads
            dropout: Dropout probability
            chunk_size: Size of chunks to process (reduces memory from O(L²) to O(chunk_size²))
        """
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.atten_len = atten_len
        self.num_heads = num_heads
        self.chunk_size = chunk_size

        # Multi-head attention expects embed_dim to be divisible by num_heads
        assert embed_dim % num_heads == 0, (
            f'embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})'
        )

        # Project each channel independently to query, key, value
        self.query_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.key_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.value_proj = nn.Conv1d(embed_dim, embed_dim, 1)

        # Output projection
        self.out_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def create_banded_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create banded attention mask.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Boolean mask of shape [seq_len, seq_len] where True allows attention
        """
        # Create a causal mask (lower triangular)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=-self.atten_len + 1
        )
        mask = torch.tril(mask, diagonal=self.atten_len - 1)

        # Convert to boolean (1 = attend, 0 = mask out)
        # For scaled_dot_product_attention, we need True where we want to attend
        return mask.bool()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through cross attention with chunked processing.

        Processes the sequence in overlapping chunks to reduce memory from O(L²) to O(chunk_size²).
        Each chunk overlaps by atten_len on each side to ensure correct banded attention.

        Args:
            x: Input tensor of shape [batch_size, embed_dim, seq_len]

        Returns:
            Output tensor of same shape as input
        """
        batch_size, embed_dim, seq_len = x.shape
        head_dim = embed_dim // self.num_heads

        # Project to Q, K, V once for the entire sequence
        Q = self.query_proj(x)  # [B, embed_dim, seq_len]
        K = self.key_proj(x)  # [B, embed_dim, seq_len]
        V = self.value_proj(x)  # [B, embed_dim, seq_len]

        # Reshape for multi-head attention
        # [B, embed_dim, seq_len] -> [B, num_heads, seq_len, head_dim]
        Q = Q.reshape(batch_size, self.num_heads, head_dim, seq_len).transpose(2, 3)
        K = K.reshape(batch_size, self.num_heads, head_dim, seq_len).transpose(2, 3)
        V = V.reshape(batch_size, self.num_heads, head_dim, seq_len).transpose(2, 3)

        # Process in chunks if sequence is longer than chunk_size
        if seq_len <= self.chunk_size:
            # Process entire sequence at once
            attn_mask = self.create_banded_mask(seq_len, x.device)
            attn_output = scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            # Chunked processing with overlap
            attn_output = torch.zeros_like(Q)
            padding = self.atten_len  # Overlap on each side

            # Process chunks with overlap
            for start_idx in range(0, seq_len, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, seq_len)

                # Add padding for context (but don't go beyond sequence boundaries)
                chunk_start = max(0, start_idx - padding)
                chunk_end = min(seq_len, end_idx + padding)
                chunk_len = chunk_end - chunk_start

                # Extract chunk with context
                Q_chunk = Q[:, :, chunk_start:chunk_end, :]
                K_chunk = K[:, :, chunk_start:chunk_end, :]
                V_chunk = V[:, :, chunk_start:chunk_end, :]

                # Create mask for this chunk
                chunk_mask = self.create_banded_mask(chunk_len, x.device)

                # Apply attention
                chunk_output = scaled_dot_product_attention(
                    Q_chunk,
                    K_chunk,
                    V_chunk,
                    attn_mask=chunk_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )

                # Extract the valid region (remove padding) and place in output
                valid_start = start_idx - chunk_start
                valid_end = valid_start + (end_idx - start_idx)
                attn_output[:, :, start_idx:end_idx, :] = chunk_output[
                    :, :, valid_start:valid_end, :
                ]

        # Reshape back to [B, embed_dim, seq_len]
        attn_output = attn_output.transpose(2, 3).reshape(
            batch_size, embed_dim, seq_len
        )

        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        # Residual connection
        return x + output
