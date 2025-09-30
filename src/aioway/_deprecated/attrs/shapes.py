# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import logging
import operator
import typing
from collections.abc import Iterable, Sequence
from typing import Any, Self

__all__ = ["Shape"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(eq=False, frozen=True)
class Shape(Sequence[int]):
    """
    ``Shape`` represents a regular (non-jagged) array's dimensions.
    """

    shape: tuple[int, ...]
    """
    The shape of the array.
    """

    def __eq__(self, other: Any) -> bool:
        LOGGER.debug("Computing %s == %s", self, other)

        if isinstance(other, Shape):
            return self.shape == other.shape

        if isinstance(other, Sequence):
            return (
                True
                and len(self) == len(other)
                and all(self[i] == elem for i, elem in enumerate(other))
            )

        return NotImplemented

    @typing.override
    def __len__(self) -> int:
        return len(self.shape)

    @typing.overload
    def __getitem__(self, idx: int) -> int: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> Sequence[int]: ...

    @typing.override
    def __getitem__(self, idx):
        return self.shape[idx]

    def numel(self) -> int:
        return functools.reduce(operator.mul, self.shape, 1)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @classmethod
    def from_iterable(cls, sequence: Iterable[int]) -> Self:
        """
        Convenient method to generate a ``Shape`` instance from a ``Sequence``.
        """

        return cls(tuple(map(int, sequence)))
