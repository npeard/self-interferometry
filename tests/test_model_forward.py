#!/usr/bin/env python3
"""Sanity tests for forward pass of TCN and UTCN models."""

import torch
import pytest

from self_interferometry.analysis.models.tcn import TCN, TCNConfig
from self_interferometry.analysis.models.utcn import UTCN, UTCNConfig

SEQUENCE_LENGTH = 256
IN_CHANNELS = 3
BATCH_SIZE = 2


@pytest.fixture
def tcn_model():
    config = TCNConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        activation='GELU',
        use_layer_norm=True,
        use_weight_norm=False,
        kernel_size=7,
        temporal_channels=[16, 32, 64, 32, 16],
        dilation_base=2,
    )
    return TCN(config).eval()


@pytest.fixture
def utcn_model():
    config = UTCNConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        activation='GELU',
        use_layer_norm=True,
        use_weight_norm=False,
        kernel_size=7,
        temporal_channels=[64, 32, 16, 8, 16, 32, 64],
        temporal_dilations=[1, 2, 4, 8, 4, 2, 1],
        horizontal_skips_map=None,
        horizontal_skip='linear',
    )
    return UTCN(config).eval()


class TestTCNForward:
    def test_output_shape(self, tcn_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = tcn_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH), (
            f'Expected shape ({BATCH_SIZE}, 1, {SEQUENCE_LENGTH}), got {out.shape}'
        )

    def test_output_is_finite(self, tcn_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = tcn_model(x)
        assert torch.isfinite(out).all(), 'TCN output contains non-finite values'

    def test_zero_input(self, tcn_model):
        x = torch.zeros(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = tcn_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH)
        assert torch.isfinite(out).all()


class TestUTCNForward:
    def test_output_shape(self, utcn_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = utcn_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH), (
            f'Expected shape ({BATCH_SIZE}, 1, {SEQUENCE_LENGTH}), got {out.shape}'
        )

    def test_output_is_finite(self, utcn_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = utcn_model(x)
        assert torch.isfinite(out).all(), 'UTCN output contains non-finite values'

    def test_zero_input(self, utcn_model):
        x = torch.zeros(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = utcn_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH)
        assert torch.isfinite(out).all()
