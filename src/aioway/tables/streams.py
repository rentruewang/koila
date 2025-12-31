# Copyright (c) AIoWay Authors - All Rights Reserved

"The ``Stream``s that support random access."

import copy
import dataclasses as dcls
import typing
from collections.abc import Generator
from typing import Self

from tensordict import TensorDict

from aioway.streams import Stream

from .tables import Table

__all__ = ["TableStream"]


@dcls.dataclass
class TableStream(Stream):
    """
    This is a ``Stream`` backed by a ``Table``.
    Since a ``Table`` is in-memory, this just retrives the data from the ``Table``.

    This kind of ``Stream`` supports random access,
    note that the first item (index 0) does not start at the start of the stream,
    but rather offset-ed at the current index,
    for consistency with ``Iterator`` behavior.
    """

    table: "Table"
    """
    The ``Table`` whose data we are interested in.
    """

    @typing.override
    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> TensorDict:
        return self.table[idx]

    @typing.override
    def __iter__(self) -> Self:
        # Shallow copy self.
        return copy.copy(self)

    @typing.override
    def _children(self) -> Generator[Stream]:
        return
        yield

    @typing.override
    def _read(self) -> TensorDict:
        # Just access the ``self.idx`` position in the ``Table``.
        # Index out of bounds -> the iteration has stopped.
        try:
            return self.table[self.idx]
        except IndexError as ie:
            raise StopIteration from ie

    def reset(self) -> Self:
        """
        Create an iterator to the underlying table, with ``idx`` reset.

        Returns:
            A shallow clone of ``self`` with ``idx == 0``.
        """

        return type(self)(table=self.table)
