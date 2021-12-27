from __future__ import annotations

from abc import abstractmethod
from typing import (
    Protocol,
    TypeVar,
    runtime_checkable,
)


E = TypeVar("E")
T = TypeVar("T", covariant=True)
V = TypeVar("V", contravariant=True)


@runtime_checkable
class Runnable(Protocol[T]):
    @abstractmethod
    def run(self) -> T:
        ...
