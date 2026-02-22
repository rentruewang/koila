# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from abc import ABC
from collections.abc import KeysView
from typing import Protocol, Self

__all__ = ["Table", "Column", "Indexible"]


@typing.runtime_checkable
class Indexible[I](Protocol):
    @typing.overload
    def idx(self, idx: int) -> I: ...

    @typing.overload
    def idx(self, idx: slice | list[int]) -> Self: ...


class Column(ABC):
    """
    A column type for the ``Table`` type,
    which is built from ``Column``s and ``str`` keys.

    See ``Table`` for more details.
    """


class Table[C: Column](ABC):
    """
    A tabular type that acts like a table (can be ``Frame``, ``Stream``, ``Chunk`` etc).

    A ``Table`` should support the following functions:

    1. ``column(key: str, /) -> Collumn``.
        Getting the individual column.
    2. ``select(*keys: str) -> Self``.
        Getting a couple of columns should return the same ``Table``.
    3. ``keys() -> KeysView[str]``
    """

    @abc.abstractmethod
    def keys(self) -> KeysView[str]:
        """
        A ``KeysView`` object.
        """

        ...

    @abc.abstractmethod
    def column(self, key: str, /) -> C:
        """
        Get the column from the ``Tabular`` object.
        A ``KeyError`` is raised if the column is not present.

        Essentially this is the ``Mapping.__getitem__`` method,
        but a normal method to simplify implementation.

        Args:
            key: The column name.

        Returns:
            The column instance.

        Raises:
            KeyError: If the column is not present.
        """

        ...

    def get(self, key: str, /) -> C | None:
        "This is the ``Mapping.get`` method."

        if key not in self.keys():
            return None

        return self.column(key)

    @abc.abstractmethod
    def select(self, *keys: str) -> Self:
        """
        Select multiple columns from the ``Tabular`` object.

        If a key is missing, a ``KeyError`` is raised.

        Returns:
            A ``Tabular`` that wraps the result.
        """
