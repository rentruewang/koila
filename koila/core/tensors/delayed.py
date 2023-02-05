from torch import Tensor as TorchTensor

from koila.interfaces import TensorLike


class DelayedTensor(TensorLike, TorchTensor):
    def __init__(self):
        pass
