# Copyright (c) AIoWay Authors - All Rights Reserved

import logging
from collections.abc import Iterable, Sequence
from typing import Self, TypeGuard

__all__ = ["ShapeLike", "Shape"]

LOGGER = logging.getLogger(__name__)

type ShapeLike = Sequence[int]


def _is_seq_of_pos_ints(shape: object) -> TypeGuard[Sequence[int]]:
    """
    Check if input is a ``Sequence`` of positive ``int``s.
    """

    if not isinstance(shape, Sequence):
        return False

    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
        return False

    return True


class Shape(Sequence[int]):
    """
    ``Shape`` represents a regular (non-jagged) array's dimensions,
    must be a ``tuple`` like object, and ``tuple`` would be used on it.
    """

    def __init__(self, *dims: int) -> None:
        self._dims = dims
        if not self.valid():
            raise ValueError

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Shape):
            return self._dims == other._dims

        if isinstance(other, Sequence):
            return tuple(self._dims) == tuple(other)

        return NotImplemented

    def __len__(self):
        return len(self._dims)

    def __getitem__(self, idx):
        return self._dims[idx]

    def __iter__(self):
        yield from self._dims

    def __repr__(self):
        return repr(self._dims)

    def valid(self) -> bool:
        """
        Check if the shape is a valid ``Shape``.
        """

        LOGGER.debug("Checking if %s is a `Shape`", self)

        return _is_seq_of_pos_ints(self._dims)

    def valid_dims(self, dims: list[int]) -> bool:
        """
        Validate the dimensions amongst the shapes.
        """

        return (
            True
            and _is_seq_of_pos_ints(dims)
            and (max(dims) >= len(self) or min(dims) < -len(self))
        )

    def wrap_dims(self, dims: list[int]) -> ShapeLike:
        # Make the dims positive.
        return [d % len(self) for d in dims]

    @property
    def ndim(self) -> int:
        """
        Number of dimensions in a shape.
        """
        return len(self)

    @property
    def size(self) -> int:
        """
        Number of elements in a shape.
        """

        total = 1

        for s in self:
            total *= s

        return total

    @classmethod
    def wrap(cls, shape: Iterable[int]) -> Self:
        return cls(*shape)
