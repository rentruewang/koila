import torch
from torch.nn import CrossEntropyLoss, Flatten, Linear, Module, ReLU, Sequential

from koila import lazy

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


# Create the model
nn = NeuralNetwork().to(DEVICE)

# The original input
t = torch.randn(BATCH, 28, 28).to(DEVICE)

# The original label
label = torch.randint(0, 10, [BATCH])

# The loss function
loss_fn = CrossEntropyLoss()

# Calculate losses
out = nn(t)
loss = loss_fn(out, label)

# Backward pass
nn.zero_grad()
loss.backward()

# Detach and cloneing the gradients.
grads = [p.detach().clone() for p in nn.parameters()]

# Now, wrap the input in a lazy tensor.
# Specify the dimension for batch in order to know which dimension to parallelize.
# You can now never worry about out of memory errors,
# because it's automatically handled for you.
lt = lazy(t, batch=0)

# Using the same operations.
out_lazy = nn(lt)
loss = loss_fn(out_lazy, label)
nn.zero_grad()
loss.backward()

# Would yield the same results.
my_grads = [p.detach().clone() for p in nn.parameters()]

# The outputs are the same
assert torch.allclose(out, out_lazy.torch())
# The gradients are also the same.
assert all([m.allclose(g) for (m, g) in zip(grads, my_grads)])
