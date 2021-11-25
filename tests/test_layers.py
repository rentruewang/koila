import torch
from torch import Tensor
from torch.nn import (
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    Dropout,
    LayerNorm,
    LeakyReLU,
    Linear,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    ReLU,
    Sigmoid,
    Softmax,
)

import koila
from koila import LazyTensor

from . import common


def test_linear_layer() -> None:
    arr = torch.randn(7, 11, 13)
    la = koila.lazy(arr)
    layer = Linear(13, 17)

    out = layer(arr)
    assert out.shape == (7, 11, 17)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert lo.shape == (7, 11, 17)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    common.assert_isclose(lo.run(), out)


def test_batchnorm_layers() -> None:
    # 1D
    arr = torch.randn(3, 5, 7)
    la = koila.lazy(arr)
    layer = BatchNorm1d(5)

    out = layer(arr)
    assert out.shape == (3, 5, 7)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert lo.shape == (3, 5, 7)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    common.assert_isclose(lo.run(), out)

    # 2D
    arr = torch.randn(3, 5, 7, 11)
    la = koila.lazy(arr)
    layer = BatchNorm2d(5)

    out = layer(arr)
    assert out.shape == (3, 5, 7, 11)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert lo.shape == (3, 5, 7, 11)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    common.assert_isclose(lo.run(), out)

    # 3D
    arr = torch.randn(3, 5, 7, 11, 13)
    la = koila.lazy(arr)
    layer = BatchNorm3d(5)

    out = layer(arr)
    assert out.shape == (3, 5, 7, 11, 13)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert lo.shape == (3, 5, 7, 11, 13)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    common.assert_isclose(lo.run(), out)


def test_layernorm_layers() -> None:
    # 1D
    arr = torch.randn(3, 5, 7)
    la = koila.lazy(arr)
    layer = LayerNorm([5, 7])

    out = layer(arr)
    assert out.shape == (3, 5, 7)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert lo.shape == (3, 5, 7)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    common.assert_isclose(lo.run(), out)


def test_dropout_layer() -> None:
    arr = torch.randn(7, 11)
    la = koila.lazy(arr)
    layer = Dropout(p=0.5)

    out = layer(arr)
    assert out.shape == (7, 11)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert lo.shape == (7, 11)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)


def test_relu_layer() -> None:
    arr = torch.randn(7, 11)
    la = koila.lazy(arr)
    layer = ReLU()

    out = layer(arr)
    assert out.shape == (7, 11)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert lo.shape == (7, 11)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    common.assert_isclose(lo.run(), out)


def test_leaky_relu_layer() -> None:
    arr = torch.randn(7, 11)
    la = koila.lazy(arr)
    layer = LeakyReLU(negative_slope=0.3)

    out = layer(arr)
    assert out.shape == (7, 11)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert lo.shape == (7, 11)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    common.assert_isclose(lo.run(), out)


def test_sigmoid_layer() -> None:
    arr = torch.randn(7, 11)
    la = koila.lazy(arr)
    layer = Sigmoid()

    out = layer(arr)
    assert out.shape == (7, 11)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert lo.shape == (7, 11)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    common.assert_isclose(lo.run(), out)


def test_softmax_layer() -> None:
    arr = torch.randn(7, 11)
    la = koila.lazy(arr)
    layer = Softmax(dim=-1)

    out = layer(arr)
    assert out.shape == (7, 11)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert lo.shape == (7, 11)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    common.assert_isclose(lo.run(), out)


def test_conv_layer() -> None:
    # 1D
    arr = torch.randn(7, 11, 13)
    la = koila.lazy(arr)
    layer = Conv1d(11, 17, kernel_size=3, stride=2)

    out = layer(arr)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    assert lo.shape == out.shape
    common.assert_isclose(lo.run(), out)

    # 2D
    arr = torch.randn(7, 11, 13, 14)
    la = koila.lazy(arr)
    layer = Conv2d(11, 17, kernel_size=3, stride=2)

    out = layer(arr)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    assert lo.shape == out.shape
    common.assert_isclose(lo.run(), out)

    # 3D
    arr = torch.randn(7, 11, 13, 14, 15)
    la = koila.lazy(arr)
    layer = Conv3d(11, 17, kernel_size=3, stride=2)

    out = layer(arr)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    assert lo.shape == out.shape
    common.assert_isclose(lo.run(), out)


def test_convtranspose_layer() -> None:
    # 1D
    arr = torch.randn(7, 11, 13)
    la = koila.lazy(arr)
    layer = ConvTranspose1d(11, 17, kernel_size=3, stride=2)

    out = layer(arr)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    assert lo.shape == out.shape
    common.assert_isclose(lo.run(), out)

    # 2D
    arr = torch.randn(7, 11, 13, 14)
    la = koila.lazy(arr)
    layer = ConvTranspose2d(11, 17, kernel_size=3, stride=2)

    out = layer(arr)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    assert lo.shape == out.shape
    common.assert_isclose(lo.run(), out)

    # 3D
    arr = torch.randn(7, 11, 13, 14, 15)
    la = koila.lazy(arr)
    layer = ConvTranspose3d(11, 17, kernel_size=3, stride=2)

    out = layer(arr)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    assert lo.shape == out.shape
    common.assert_isclose(lo.run(), out)


def test_maxpool_layer() -> None:
    # 1D
    arr = torch.randn(7, 11, 13)
    la = koila.lazy(arr)
    layer = MaxPool1d(kernel_size=3, stride=2)

    out = layer(arr)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    assert lo.shape == out.shape
    common.assert_isclose(lo.run(), out)

    # 2D
    arr = torch.randn(7, 11, 13, 14)
    la = koila.lazy(arr)
    layer = MaxPool2d(kernel_size=3, stride=2)

    out = layer(arr)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    assert lo.shape == out.shape
    common.assert_isclose(lo.run(), out)

    # 3D
    arr = torch.randn(7, 11, 13, 14, 15)
    la = koila.lazy(arr)
    layer = MaxPool3d(kernel_size=3, stride=2)

    out = layer(arr)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    assert lo.shape == out.shape
    common.assert_isclose(lo.run(), out)


def test_avgpool_layer() -> None:
    # 1D
    arr = torch.randn(7, 11, 13)
    la = koila.lazy(arr)
    layer = AvgPool1d(kernel_size=3, stride=2)

    out = layer(arr)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    assert lo.shape == out.shape
    common.assert_isclose(lo.run(), out)

    # 2D
    arr = torch.randn(7, 11, 13, 14)
    la = koila.lazy(arr)
    layer = AvgPool2d(kernel_size=3, stride=2)

    out = layer(arr)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    assert lo.shape == out.shape
    common.assert_isclose(lo.run(), out)

    # 3D
    arr = torch.randn(7, 11, 13, 14, 15)
    la = koila.lazy(arr)
    layer = AvgPool3d(kernel_size=3, stride=2)

    out = layer(arr)
    assert not isinstance(out, LazyTensor)
    assert isinstance(out, Tensor)

    assert isinstance(la, LazyTensor)
    lo = layer(la)
    assert not isinstance(lo, Tensor)
    assert isinstance(lo, LazyTensor)
    assert lo.shape == out.shape
    common.assert_isclose(lo.run(), out)
