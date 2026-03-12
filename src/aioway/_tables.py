# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from abc import ABC
from collections.abc import KeysView
from typing import Protocol, Self

__all__ = ["Table", "Indexible"]


@typing.runtime_checkable
class Indexible[I](Protocol):
    @typing.overload
    def idx(self, idx: int) -> I: ...

    @typing.overload
    def idx(self, idx: slice | list[int]) -> Self: ...


class Table[C](ABC):
    """
    A tabular type that acts like a table (can be ``Frame``, ``Stream``, ``Chunk`` etc).

    A ``Table`` should support the following functions:

    1. ``column(key: str, /) -> C``.
        Getting the individual column.
    2. ``select(*keys: str) -> Self``.
        Getting a couple of columns should return the same ``Table``.
    3. ``keys() -> KeysView[str]``
    """

    @typing.overload
    def __getitem__(self, key: str, /) -> C: ...

    @typing.overload
    def __getitem__(self, key: list[str], /) -> Self: ...

    def __getitem__(self, key, /):
        match key:
            case str():
                return self.column(key)
            case list() if all(isinstance(i, str) for i in key):
                return self.select(*key)

        raise TypeError(
            "The default implemenetation of `Table.__getitem__` "
            f"does not know how to handle {key=}. "
            "It only supports `key` of type `str` and `list[str]`."
        )

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

    @typing.overload
    def get(self, key: str, /) -> C | None: ...

    @typing.overload
    def get(self, key: str, /, default: C) -> C: ...

    @typing.overload
    def get[T](self, key: str, /, default: T) -> C | T: ...

    @typing.no_type_check
    def get(self, key, /, default):
        "This is the ``Mapping.get`` method."

        if key not in self.keys():
            return default

        return self.column(key)

    @abc.abstractmethod
    def select(self, *keys: str) -> Self:
        """
        Select multiple columns from the ``Tabular`` object.

        If a key is missing, a ``KeyError`` is raised.

        Returns:
            A ``Tabular`` that wraps the result.
        """
