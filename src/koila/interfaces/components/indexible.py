from __future__ import annotations

import abc
from typing import Protocol, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Indexible(Protocol[K, V]):
    """
    An `Indexible` container acts like a dictionary, that can be looked up.
    A list is indexible (with its key type limited to integers.)
    """

    @abc.abstractmethod
    def __getitem__(self, index: K) -> V:
        "The [] (getter) operator."

        ...

    @abc.abstractmethod
    def __setitem__(self, index: K, value: V) -> None:
        "The [] (setter) operator."

        ...
