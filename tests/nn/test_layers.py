# # Copyright (c) AIoWay Authors - All Rights Reserved

# import pytest
# import torch
# from pytest import FixtureRequest
# from torch import Tensor
# from torch.nn import Conv2d as TorchConv2d

# from aioway.attrs import Attr
# from aioway.modules import Conv2d, Identity, Linear


# @pytest.fixture
# def linear():
#     return Linear(in_features=3, out_features=5, bias=True)


# @pytest.fixture
# def identity():
#     return Identity()


# @pytest.fixture
# def linear_input():
#     return torch.randn(7, 3)


# @pytest.fixture
# def linear_attr(linear_input: Tensor):
#     return Attr.from_tensor(linear_input)


# @pytest.fixture
# def conv2d_input():
#     return torch.randn(3, 5, 7, 11)


# @pytest.fixture(params=[1, 2, 3])
# def dilation(request: FixtureRequest):
#     return request.param


# @pytest.fixture(params=[0, 1, 2, 3])
# def padding(request: FixtureRequest):
#     return request.param


# @pytest.fixture(params=[1, 2, 3])
# def stride(request: FixtureRequest):
#     return request.param


# @pytest.fixture(params=[1, 2, 3])
# def kernel_size(request: FixtureRequest):
#     return request.param


# @pytest.fixture
# def conv2d(dilation: int, padding: int, stride: int, kernel_size: int):
#     return Conv2d(
#         in_channels=5,
#         out_channels=13,
#         kernel_size=kernel_size,
#         dilation=dilation,
#         padding=padding,
#         stride=stride,
#     )


# @pytest.fixture
# def nn_conv2d(
#     conv2d: Conv2d, dilation: int, padding: int, stride: int, kernel_size: int
# ):
#     return conv2d.MODULE_TYPE(
#         in_channels=5,
#         out_channels=13,
#         kernel_size=kernel_size,
#         dilation=dilation,
#         padding=padding,
#         stride=stride,
#     )


# def test_linear(linear: Linear, linear_input: Tensor):
#     result = linear.forward(linear_input)
#     assert isinstance(result, Tensor)
#     assert result.shape == (7, 5)


# def test_linear_preview(linear: Linear, linear_attr: Attr):
#     result = linear.preview(linear_attr)
#     assert isinstance(result, Attr)
#     assert result.shape == (7, 5)


# def test_identity(identity: identity, linear_input: Tensor):
#     result = identity.forward(linear_input)
#     assert isinstance(result, Tensor)
#     assert result.shape == (7, 3)


# def test_identity_preview(identity: identity, linear_attr: Attr):
#     result = identity.preview(linear_attr)
#     assert isinstance(result, Attr)
#     assert result.shape == (7, 3)


# def test_conv2d_forward(conv2d: Conv2d, nn_conv2d: TorchConv2d, conv2d_input: Tensor):
#     ours = conv2d.forward(conv2d_input)
#     theirs = nn_conv2d.forward(conv2d_input)

#     assert ours.shape == theirs.shape


# def test_conv2d_preview(conv2d: Conv2d, nn_conv2d: TorchConv2d, conv2d_input: Tensor):
#     ours = conv2d.preview(conv2d_input)
#     theirs = nn_conv2d.forward(conv2d_input)

#     assert ours.shape == theirs.shape
