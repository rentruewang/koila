from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import overload

import torch
from torch import Tensor


@dataclass
class LazyOperation:
    function: Callable[..., Tensor]
    args: tuple[LazyTensor | Tensor | int | float, ...]
    kwargs: dict[str, LazyTensor | Tensor | int | float]

    @overload
    @staticmethod
    def evaluate(value: LazyTensor | Tensor) -> Tensor:
        ...

    @overload
    @staticmethod
    def evaluate(value: int) -> int:
        ...

    @overload
    @staticmethod
    def evaluate(value: float) -> float:
        ...

    @staticmethod
    def evaluate(value: LazyTensor | Tensor | int | float) -> Tensor | int | float:
        if isinstance(value, LazyTensor):
            return value.run()
        return value

    def run(self) -> Tensor:
        print("run")
        args = [self.evaluate(arg) for arg in self.args]
        kwargs = {k: self.evaluate(w) for (k, w) in self.kwargs.items()}
        return self.function(*args, **kwargs)


@dataclass(init=False)
class LazyTensor:
    data: Tensor | LazyOperation
    retain_forward: bool

    def __init__(
        self, data: Tensor | LazyOperation, retain_forward: bool = False
    ) -> None:
        self.data = data
        self.retain_forward = retain_forward

    @classmethod
    def ensure_lazy(cls, tensor: LazyTensor | Tensor) -> LazyTensor:
        if isinstance(tensor, Tensor):
            return cls(tensor)
        return tensor

    def __add__(self, other: LazyTensor | Tensor) -> LazyTensor:
        return self.add(other)

    def __radd__(self, other: LazyTensor | Tensor) -> LazyTensor:
        return self.ensure_lazy(other).add(self)

    def add(self, other: LazyTensor | Tensor) -> LazyTensor:
        tensor = self.ensure_lazy(other)
        return LazyTensor(LazyOperation(torch.add, (self, tensor), {}))

    def run(self) -> Tensor:
        data = self.data
        if isinstance(data, Tensor):
            return data

        tensor = data.run()
        if not self.retain_forward:
            self.data = tensor
        return tensor
