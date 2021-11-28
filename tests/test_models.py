import torch
from torch import Tensor
from torch.nn import Flatten, Linear, Module, ReLU, Sequential

from koila import LazyTensor


def test_nn() -> None:
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

    t = torch.randn(3, 28, 28)
    nn = NeuralNetwork()

    out = nn(t)
    assert out.shape == (3, 10)
    assert isinstance(out, Tensor)
    assert not isinstance(out, LazyTensor)

    lt = LazyTensor(torch.randn(3, 28, 28))
    nn = NeuralNetwork()

    lout = nn(lt)
    assert lout.shape == (3, 10)
    assert not isinstance(lout, Tensor)
    assert isinstance(lout, LazyTensor)
