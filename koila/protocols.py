from typing import Protocol, runtime_checkable

from torch import Tensor as TorchTensor


@runtime_checkable
class Runnable(Protocol):
    def run(self) -> TorchTensor:
        ...
