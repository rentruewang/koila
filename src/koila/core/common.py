from __future__ import annotations

from typing import Protocol

from torch import Tensor as TorchTensor

from koila.interfaces import TensorLike


class Template(TensorLike, Protocol):
    _data: TensorLike | TorchTensor

    def __str__(self):
        pass
