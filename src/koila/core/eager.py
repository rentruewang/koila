from torch import Tensor as TorchTensor

from koila.interfaces import TensorLike


class EagerTensor(TensorLike, TorchTensor):
    def __init__(self):
        pass
