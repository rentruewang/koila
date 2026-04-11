# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections import abc as cabc

import numpy as np
from numpy import typing as npt

__all__ = ["SeqKeysView", "SetKeysView", "IntArray", "BatchIndex"]


@dcls.dataclass(frozen=True)
class _ContainerKeysView[C: cabc.Sequence[str] | cabc.Set[str]](cabc.KeysView[str]):
    seq: C

    @typing.override
    def __contains__(self, key: object) -> bool:
        return key in self.seq

    @typing.override
    def __iter__(self):
        yield from self.seq


class SeqKeysView(_ContainerKeysView[cabc.Sequence[str]]): ...


class SetKeysView(_ContainerKeysView[set[str]]): ...


type IntArray = npt.NDArray[np.int_]
"Integer numpy array."

type BatchIndex = slice | list[int] | IntArray
"The types that can be used for bath indexing."
