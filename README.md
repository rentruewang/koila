# 🐨 Koila

> Koila solves `CUDA error: out of memory error` painlessly.
> Fix it with just one line of code, and forget it.

![Type Checking](https://github.com/rentruewang/koila/actions/workflows/typecheck.yaml/badge.svg)
![Formatting](https://github.com/rentruewang/koila/actions/workflows/format.yaml/badge.svg)
![Unit testing](https://github.com/rentruewang/koila/actions/workflows/unittest.yaml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Never%20worry%20about%20out%20of%20memory%20errors%20again&url=https://github.com/rentruewang/koila&hashtags=pytorch,outofmemory)

![Koila](./assets/koila.png)

## 🚀 Features

- 🙅 Prevents `CUDA error: out of memory error` with one single line of code.

- 🦥 Lazily evaluates pytorch code to save computing power.

- ✂️ Automatically splits along the batch dimension to more GPU friendly numbers (2's powers) to speed up the execution.

- 🤏 Minimal API (wrapping all inputs will be enough).

## 🤔 Why Koila?

Ever encountered `RuntimeError: CUDA error: out of memory`?
We all love `PyTorch` because of its speed, efficiency, and transparency, but that means it doesn't do extra things. Things like preventing a very common error that has been bothering many users since [2017](https://github.com/pytorch/pytorch/issues/958#issuecomment-285090162).

This library aims to prevent that by being a light-weight wrapper over native `PyTorch`. When a tensor is wrapped, the library **automatically computes the amount of remaining GPU memory and uses the right batch size**, saving everyone from having to manually finetune the batch size whenever a model is used.

Also, the library automatically uses the right batch size to GPU. Did you know that using bigger batches doesn't always speed up processing? It's handled automatically in this library too.

Because `Koila` code is `PyTorch` code, as it runs `PyTorch` under the hood, you can use both together without worrying compatibility.

Oh, and all that in 1 line of code! 😊

## ⬇️ Installation

`Koila` is available on [PyPI](https://pypi.org/project/koila/). To install, run the following command.

```bash
pip install koila
```

## 🏃 Getting started

The usage is dead simple. For example, you have the following `PyTorch` code (copied from `PyTorch`'s [tutorial](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html))

Define the input, label, and model:

```python
# A batch of MNIST image
input = torch.randn(8, 28, 28)

# A batch of labels
label = torch.randn(0, 10, [8])

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
```

Define the loss function, calculate output and losses.

```python
loss_fn = CrossEntropyLoss()

# Calculate losses
out = nn(t)
loss = loss_fn(out, label)

# Backward pass
nn.zero_grad()
loss.backward()
```

Ok. How to adapt the code to use `Koila`'s features?

You change this line of code:

```python
# Wrap the input tensor.
# If a batch argument is provided, that dimension of the tensor would be treated as the batch.
# In this case, the first dimension (dim=0) is used as batch's dimension.
input = lazy(torch.randn(8, 28, 28), batch=0)
```

Done. You will not run out of memory again.

See `examples/getting-started.py` for the full example.

## 🏋️ How does it work under the hood?

`CUDA error: out of memory` generally happens in forward pass, because temporary variables will need to be saved in memory.

`Koila` is a thin wrapper around `PyTorch`. It is inspired by TensorFlow's static/lazy evaluation. By building the graph first, and run the model only when necessarily, the model has access to all the information necessarily to determine how much resources is really need to compute the model.

In terms of memory usage, only **shapes of temporary variables are required to calculate the memory usage of those variables used in the model**. For example, `+` takes in two tensors with equal sizes, and outputs a tensor with a size equal to the input size, and `log` takes in one tensor, and outputs another tensor with the same shape. Broadcasting makes it a little more complicated than that, but the general ideas are the same. By tracking all these shapes, one could easily tell how much memory is used in a forward pass. And select the optimal batch size accordingly.

## 🐌 It sounds slow. Is it?

**NO**. Indeed, calculating shapes and computing the size and memory usage sound like a lot of work. However, keep in mind that even a gigantic model like GPT-3, which has 96 layers, has only a few hundred nodes in its computing graph. Because `Koila`'s algorithms run in linear time, any modern computer will be able to handle a graph like this instantly.

Most of the computing is spent on computing individual tensors, and transferring tensors across devices. And bear in mind that those checks happen in vanilla `PyTorch` anyways. So no, not slow at all.

## 🔊 How to pronounce koila?

This project was originally named _koala_, the laziest species in the world, and this project is about lazy evaluation of tensors. However, as that name is taken on [PyPI](https://pypi.org/project/koala/), I had no choice but to use another name. `Koila` is a word made up by me, pronounced similarly to _voila_ (It's a French word), so sounds like koala.

## ⭐ Give me a star!

If you like what you see, please consider giving this a star (★)!

## 🏗️ Why did I build this?

Batch size search is not new. In fact, the mighty popular [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) has it. So why did I go through the trouble and build this project?

PyTorch Lightning's batch size search is deeply integrated in its own ecosystem. You have to use its `DataLoader`, subclass from their models, and train your models accordingly. While it works well with supervised learning tasks, it's really painful to use in a reinforcement learning task, where interacting with the environment is a must.

In comparison, because `Koila` is a super lightweight PyTorch wrapper, it works when PyTorch works, thus providing maximum flexibility and minimal changes to existing code.

## 📝 Todos

- 🧩 Provide an extensible API to write custom functions for the users.
- 😌 Simplify internal workings even further. (Especially interaction between `Tensor`s and `LazyTensor`s).
- 🍪 Work with multiple GPUs.

## 🚧 Warning

The code works on many cases, but it's still a work in progress. This is not (yet) a fully `PyTorch` compatible library due to limited time.

## 🥰 Contributing

We take openness and inclusiveness very seriously. We have adopted the following [Code of Conduct](./CODE_OF_CONDUCT.md).
