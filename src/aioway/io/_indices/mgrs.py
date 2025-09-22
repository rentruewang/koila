# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Iterator, Mapping
from typing import NamedTuple

from numpy.typing import NDArray

from aioway._errors import AiowayError
from aioway.io import Frame

from .indices import Index
from .ops import IndexOp

__all__ = ["MultiOpIndex"]

type MultiCol = tuple[str, ...]


@dcls.dataclass(frozen=True)
class MultiOpIndex:
    """
    The indices based on the type of operations they support,
    for a given set of columns.
    """

    mgr: "IndexManager"
    columns: MultiCol
    indices: dict[type[IndexOp], Index]

    def __call__(self, op: IndexOp, value: NDArray) -> NDArray:
        index = self.indices[type(op)]
        return index(op=op, value=value)


class _ColTypeIndex(NamedTuple):
    """
    A container storing columns, op types, and indices themselves.
    """

    cols: MultiCol
    ops: type[IndexOp]
    idx: Index


@typing.final
@dcls.dataclass(frozen=True)
class IndexManager(Mapping[MultiCol, MultiOpIndex]):
    """
    The ``IndexManager`` class is acts as a dictionary,
    providing some additional utility to make the API easy to use.
    """

    frame: Frame
    """
    The framem for which the ``IndexManager`` manages indices.
    """

    indices: list[_ColTypeIndex] = dcls.field(default_factory=list, init=False)
    """
    All the indices curerntly stored.
    """

    def __iter__(self) -> Iterator[MultiCol]:
        seen: set[MultiCol] = set()

        for cols, _, _ in self.indices:
            if cols in seen:
                continue

            yield cols
            seen.add(cols)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, key: MultiCol) -> MultiOpIndex:
        # Filter out the desired indices.
        selected = [idx_info for idx_info in self.indices if idx_info.cols == key]

        # We can directly do this without checking because `create`
        # prevents index collision.
        return MultiOpIndex(
            mgr=self, columns=key, indices={op: idx for _, op, idx in selected}
        )


class MultiColInitError(AiowayError, TypeError): ...


class NoItemFoundError(AiowayError, KeyError): ...
