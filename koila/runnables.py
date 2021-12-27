from __future__ import annotations

import functools
import operator
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    NamedTuple,
    Protocol,
    Tuple,
    TypeVar,
    overload,
    runtime_checkable,
)

from torch import Tensor
from torch import device as Device
from torch import dtype as DType

E = TypeVar("E")
T = TypeVar("T", covariant=True)
V = TypeVar("V", contravariant=True)


@runtime_checkable
class Runnable(Protocol[T]):
    @abstractmethod
    def run(self) -> T:
        ...
