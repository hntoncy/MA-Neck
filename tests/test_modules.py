import torch

from ma_neck import MGCA, MSACA, SAE


def test_sae_shape():
    x = torch.randn(2, 64, 32, 32)
    y = SAE(64)(x)
    assert y.shape == x.shape


def test_mgca_shapes():
    x1 = torch.randn(2, 64, 32, 32)
    x2 = torch.randn(2, 64, 32, 32)
    y1, y2 = MGCA(64)(x1, x2)
    assert y1.shape == x1.shape
    assert y2.shape == x2.shape


def test_msaca_concat_shape():
    x1 = torch.randn(2, 64, 32, 32)
    x2 = torch.randn(2, 64, 32, 32)
    y = MSACA(64)([x1, x2])
    assert y.shape == (2, 128, 32, 32)
