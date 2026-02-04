#!/usr/bin/env python3

import logging
from dataclasses import dataclass

from torch import Tensor, nn

# Conditional import for newer PyTorch versions
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    ATTENTION_BACKEND_AVAILABLE = True
except ImportError:
    ATTENTION_BACKEND_AVAILABLE = False
    SDPBackend = None
    sdpa_kernel = None

from .cross_attention_block import CrossAttentionBlock
from .temporal_block import TemporalBlock

logger = logging.getLogger(__name__)


@dataclass
class StemTCANConfig:
    # Common parameters (required by training interface)
    input_size: int  # Length of input sequence
    output_size: int  # Length of output sequence (same as input for our case)
    in_channels: int  # Number of input channels
    activation: str
    norm: str  # 'layer' or 'batch'
    dropout: float

    # StemTCAN specific parameters
    kernel_size: int  # Kernel size for all temporal blocks
    n_stem_blocks: int  # Number of TemporalBlocks in the stem (default: 2)
    n_post_attention_blocks: int  # Number of TemporalBlocks after attention
    stem_out_channels: int  # Output channels for stem blocks
    post_attention_channels: list[int]  # Output channels for post-attention blocks
    atten_len: int  # Band length for cross attention (samples along time dimension)
    atten_heads: int  # Number of attention heads
    atten_chunk_size: int  # Chunk size for memory-efficient attention (default: 2048)
    dilation_base: int  # Base for dilation in stem blocks
    stride: int  # Stride for all layers


class StemTCAN(nn.Module):
    """Stem-fusion Temporal Convolutional Attention Network.

    Architecture:
    1. Stem: Two TemporalBlocks to learn local features
    2. Cross Attention: Banded cross attention for channel fusion
    3. Post-attention: Configurable number of TemporalBlocks
    4. Output: 1x1 convolution for regression
    """

    def __init__(self, config: StemTCANConfig):
        super().__init__()
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.in_channels = config.in_channels
        self.n_stem_blocks = config.n_stem_blocks
        self.n_post_attention_blocks = config.n_post_attention_blocks
        self.dilation_base = config.dilation_base
        self.stride = config.stride

        # Validate configuration
        assert len(config.post_attention_channels) == config.n_post_attention_blocks, (
            f'post_attention_channels length ({len(config.post_attention_channels)}) '
            f'must match n_post_attention_blocks ({config.n_post_attention_blocks})'
        )

        # Lifting layer (input projection)
        self.lifting = nn.Conv1d(config.in_channels, config.stem_out_channels, 1)

        # Stem: n_stem_blocks TemporalBlocks for local feature extraction
        self.stem_blocks = nn.ModuleList([])
        for i in range(self.n_stem_blocks):
            dilation_size = self.dilation_base**i
            in_channels = (
                config.stem_out_channels
            )  # All stem blocks have same channel count
            out_channels = config.stem_out_channels

            # Calculate padding to maintain sequence length
            padding = (config.kernel_size - self.stride) * dilation_size

            self.stem_blocks.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    config.kernel_size,
                    stride=self.stride,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=config.dropout,
                    activation=config.activation,
                    norm=config.norm,
                    input_length=config.input_size,
                )
            )

        # Cross attention block for channel fusion
        self.cross_attention = CrossAttentionBlock(
            n_channels=config.in_channels,
            embed_dim=config.stem_out_channels,
            atten_len=config.atten_len,
            num_heads=config.atten_heads,
            dropout=config.dropout,
            chunk_size=config.atten_chunk_size,
        )

        # Post-attention TemporalBlocks
        self.post_attention_blocks = nn.ModuleList([])
        for i in range(self.n_post_attention_blocks):
            dilation_size = self.dilation_base**i
            in_channels = (
                config.stem_out_channels
                if i == 0
                else config.post_attention_channels[i - 1]
            )
            out_channels = config.post_attention_channels[i]

            # Calculate padding to maintain sequence length
            padding = (config.kernel_size - self.stride) * dilation_size

            self.post_attention_blocks.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    config.kernel_size,
                    stride=self.stride,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=config.dropout,
                    activation=config.activation,
                    norm=config.norm,
                    input_length=config.input_size,
                )
            )

        # Output layer: 1x1 convolution for regression
        self.output_layer = nn.Conv1d(config.post_attention_channels[-1], 1, 1)

        # Log model info
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f'Number of parameters in StemTCAN: {total_params:,}')

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through StemTCAN.

        Args:
            x: Input tensor of shape [batch_size, in_channels, sequence_length]

        Returns:
            Tensor of shape [batch_size, 1, sequence_length] containing predictions
        """
        # Lifting: project input to stem feature dimension
        x = self.lifting(x)  # [B, stem_out_channels, seq_len]

        # Stem: process through stem blocks for local feature extraction
        for stem_block in self.stem_blocks:
            x = stem_block(x)  # [B, stem_out_channels, seq_len]

        # Cross attention: fuse channel information
        x = self.cross_attention(x)  # [B, stem_out_channels, seq_len]

        # Post-attention: process through additional temporal blocks
        for post_block in self.post_attention_blocks:
            x = post_block(x)  # [B, post_attention_channels[i], seq_len]

        # Output: 1x1 convolution for regression
        output = self.output_layer(x)  # [B, 1, seq_len]

        return output
