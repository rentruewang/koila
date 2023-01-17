from __future__ import annotations

import torch
from torch import Tensor as TorchTensor

from koila.interfaces import TensorLike
from typing import Protocol


class Template(TensorLike, Protocol):
    _data: TensorLike | TorchTensor

    def __str__(self):
        pass
