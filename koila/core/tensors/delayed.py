import torch
from torch import Tensor as TorchTensor

from koila.interfaces import TensorLike
from typing import Protocol


class DelayedTensor(TensorLike, TorchTensor):
    def __init__(self):
        pass
