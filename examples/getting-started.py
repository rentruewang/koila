from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Flatten, Linear, Module, ReLU, Sequential

from koila import LazyTensor, lazy

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.DEBUG)

# It works on cpu and cuda.
DEVICE = "cpu"
BATCH = 9

# Create a simple neural network, like in pytorch's tutorial.
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


def tensor_clone(grad: Tensor | None) -> Tensor:
    """
    Clones a tensor. If the gradient is None, raise a ValueError.

    Returns
    -------

    A Tensor that is cloned from the input.
    """

    if grad is None:
        raise ValueError

    return grad.detach().clone()


# Create the model.
network = NeuralNetwork().to(DEVICE)

# The original input.
input = torch.randn(BATCH, 28, 28).to(DEVICE)

# The original label.
label = torch.randint(0, 10, [BATCH])

# The loss function.
loss_fn = CrossEntropyLoss()

# Calculate output and loss.
out = network(input)
loss = loss_fn(out, label)
assert isinstance(loss, Tensor), type(loss)

# Resets the gradients to zero.
network.zero_grad()

# Backward pass.
loss.backward()

# Detach and cloneing the gradients.
grads = [tensor_clone(p.grad) for p in network.parameters()]

# Now, wrap the input in a lazy tensor.
# Specify the dimension for batch in order to know which dimension to parallelize.
# You can now never worry about out of memory errors,
# because it's automatically handled for you.
lazy_input = lazy(input, batch=0)

# Using the same operations.
lazy_out = network(lazy_input)

# Using the same operations
# The output would automatically be a LazyTensor
# but don't worry, no code modification is needed.
# When backward is called, the LazyTensors would be automatically evaluated.
lazy_loss = loss_fn(lazy_out, label)
# assert isinstance(lazy_loss, LazyTensor), type(lazy_loss)
network.zero_grad()
lazy_loss.backward()

# Would yield the same results.
lazy_grads = [tensor_clone(p.grad) for p in network.parameters()]

# The outputs are the same.
assert torch.allclose(out, lazy_out)

# The gradients are also the same.
assert all(
    [torch.allclose(grad, lazy_grad) for (grad, lazy_grad) in zip(grads, lazy_grads)]
)
