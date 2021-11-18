from typing import Protocol

from torch import Tensor as TorchTensor


class Runnable(Protocol):
    def run(self) -> TorchTensor:
        ...
