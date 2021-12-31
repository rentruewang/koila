from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T", covariant=True)


@runtime_checkable
class Runnable(Protocol[T]):
    @abstractmethod
    def run(self) -> T:
        ...
