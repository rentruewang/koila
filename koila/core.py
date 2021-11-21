from __future__ import annotations

import functools
import typing
from typing import Any, Callable, Dict, Tuple, Type
from unittest.mock import Mock

import torch
from torch import Tensor

from .protocols import Lazy, LazyFunction, Runnable


class LazyTensor(Lazy[Tensor], Mock):
    def __init__(self, data: Tensor | Runnable[Tensor] | Lazy[Tensor]) -> None:
        super().__init__(data)

    def __add__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.add(self, other))

    def __radd__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.add(other, self))

    def __sub__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.sub(self, other))

    def __rsub__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.sub(other, self))

    def __mul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.mul(self, other))

    def __rmul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.mul(other, self))

    def __truediv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.div(self, other))

    def __rtruediv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.div(other, self))

    def __floordiv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.trunc(self / other))

    def __rfloordiv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.trunc(other / self))

    def __pow__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.pow(self, other))

    def __rpow__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.pow(other, self))

    def __mod__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.remainder(self, other))

    def __rmod__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.remainder(other, self))

    def __matmul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.matmul(self, other))

    def __rmatmul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return typing.cast(LazyTensor, torch.matmul(other, self))

    def __getattr__(self, name: str) -> LazyFunction:
        func = getattr(torch, name)
        method = getattr(Tensor, name)
        wrapper = functools.wraps(method)
        partial = functools.partial(func, self)
        return LazyFunction(wrapper(partial))

    def __bool__(self) -> bool:
        return bool(self.item())

    def __int__(self) -> int:
        return int(self.item())

    def __float__(self) -> float:
        return float(self.item())

    def item(self) -> bool | int | float:
        return self.run().item()

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Tensor],
        types: Tuple[Type[Any], ...],
        args: Tuple[Tensor | LazyTensor, ...] = (),
        kwargs: Dict[str, Tensor | LazyTensor] | None = None,
    ) -> LazyTensor:
        if kwargs is None:
            kwargs = {}

        if not all(issubclass(typ, (LazyTensor, Tensor, int, float)) for typ in types):
            return NotImplemented

        return LazyTensor(LazyFunction(func)(*args, **kwargs))
