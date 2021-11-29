import torch
from torch import Tensor
from torch.nn import Flatten, Linear, Module, ReLU, Sequential

from koila import BatchInfo, LazyTensor


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

    t = torch.randn(9, 28, 28)
    nn = NeuralNetwork()

    out = nn(t)
    assert out.shape == (9, 10)
    assert isinstance(out, Tensor)
    assert not isinstance(out, LazyTensor)

    lt = LazyTensor(t, batch=0)
    assert lt.batch() == BatchInfo(0, 9)
    nn = NeuralNetwork()

    lout = nn(lt)
    assert lout.shape == (9, 10)
    assert not isinstance(lout, Tensor)
    assert isinstance(lout, LazyTensor)

    assert lt.take_batch(3, 6).size() == (3, 28, 28)
    tbout = lout.take_batch(3, 6)
    assert tbout.shape == (3, 10)
    assert isinstance(tbout, Tensor)
    assert not isinstance(tbout, LazyTensor)

    assert lout.batch() == BatchInfo(0, 9)
