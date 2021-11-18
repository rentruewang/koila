from __future__ import annotations

import dataclasses as dc
import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, overload

import torch
from torch import Tensor as TorchTensor

from .protocols import Runnable


@dataclass
class Function(Runnable):
    function: Callable[..., TorchTensor]
    args: Tuple[Tensor | TorchTensor | int | float, ...]
    kwargs: Dict[str, Tensor | TorchTensor | int | float]

    @overload
    @staticmethod
    def evaluate(value: Tensor | TorchTensor) -> TorchTensor:
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
    def evaluate(
        value: Tensor | TorchTensor | int | float,
    ) -> TorchTensor | int | float:
        if isinstance(value, Tensor):
            return value.run()
        return value

    def run(self) -> TorchTensor:
        print("run")
        args = [self.evaluate(arg) for arg in self.args]
        kwargs = {k: self.evaluate(w) for (k, w) in self.kwargs.items()}
        return self.function(*args, **kwargs)


@dataclass(init=False)
class Tensor(Runnable):
    data: TorchTensor | Function
    retain_forward: bool

    def __init__(
        self, data: TorchTensor | Function, retain_forward: bool = False
    ) -> None:
        self.data = data
        self.retain_forward = retain_forward

    def __getattr__(self, name: str) -> Callable[..., Any]:
        func = self._lookup_global_function(name)
        return functools.partial(func, self)

    def __add__(self, other: Tensor | TorchTensor) -> Tensor:
        return _add(self, other)

    def __radd__(self, other: Tensor | TorchTensor) -> Tensor:
        return _add(other, self)

    def add(self, other: Tensor | TorchTensor) -> Tensor:
        return _add(self, other)

    def run(self) -> TorchTensor:
        data = self.data
        if isinstance(data, TorchTensor):
            return data

        tensor = data.run()
        if not self.retain_forward:
            self.data = tensor
        return tensor


def _lazy(tensor: Tensor | TorchTensor) -> Tensor:
    if isinstance(tensor, TorchTensor):
        return Tensor(tensor)
    return tensor


def _add(a: Tensor | TorchTensor, b: Tensor | TorchTensor) -> Tensor:
    a = _lazy(a)
    b = _lazy(b)
    func = Function(torch.add, (a, b), {})
    return Tensor(func)
