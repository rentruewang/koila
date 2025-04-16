# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Iterator, Mapping
from typing import Any, NamedTuple

from numpy.typing import NDArray

from aioway.errors import AiowayError
from aioway.frames import Frame

from . import factories
from .indices import Index, IndexContext
from .ops import IndexOp

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


# TODO
#   Improve efficiency as currently we are using lists and linear lookup.
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

    def create(self, key: MultiCol, op: type[IndexOp], **kwargs: Any) -> None:
        index = self._create(key=key, op=op, **kwargs)
        self.indices.append(_ColTypeIndex(cols=key, ops=op, idx=index))

    def _create(self, key: MultiCol, op: type[IndexOp], **kwargs: Any) -> Index:
        return factories.index_factory(
            key=op, ctx=IndexContext(frame=self.frame, columns=key, **kwargs)
        )


class MultiColInitError(AiowayError, TypeError): ...


class NoItemFoundError(AiowayError, KeyError): ...
