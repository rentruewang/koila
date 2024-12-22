# Copyright (c) RenChu Wang - All Rights Reserved

import torch
from torch import Tensor
from torch.nn import Flatten, Linear, Module, ReLU, Sequential

from koila import BatchInfo, LazyTensor

from . import common


def test_torch_tutorial() -> None:
    "Testing the model taken from pytorch's tutorial."

    class NeuralNetwork(Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = Flatten()
            self.linear_relu_stack = Sequential(
                Linear(28 * 28, 512),
                ReLU(),
                Linear(512, 512),
                ReLU(),
                Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    input = torch.randn(9, 28, 28)
    nn = NeuralNetwork()

    output = nn(input)
    assert output.shape == (9, 10)
    assert isinstance(output, Tensor)
    assert not isinstance(output, LazyTensor)

    lazy_input = LazyTensor(input, batch=0)
    assert lazy_input.batch() == BatchInfo(0, 9)
    nn = NeuralNetwork()

    lazy_output = nn(lazy_input)
    assert lazy_output.shape == (9, 10)
    assert not isinstance(lazy_output, Tensor)
    assert isinstance(lazy_output, LazyTensor)

    assert lazy_input.run((3, 6)).size() == (3, 28, 28)
    common.assert_isclose(lazy_input.run((3, 6)), input[3:6])
    tbout = lazy_output.run((3, 6))
    assert tbout.shape == (3, 10)
    assert isinstance(tbout, Tensor)
    assert not isinstance(tbout, LazyTensor)
    common.assert_isclose(tbout, nn(input[3:6]))

    assert lazy_output.batch() == BatchInfo(0, 9)
