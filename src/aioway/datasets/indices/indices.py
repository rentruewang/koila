# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC
from collections import abc as cabc

import numpy as np
from numpy import typing as npt

from ..frames import Frame
from .ops import IndexPlan

__all__ = ["Index", "IndexContext"]


@dcls.dataclass(frozen=True)
class IndexContext:
    """
    The indexing information for looking upu an index.
    """

    frame: Frame
    """
    The `Frame` to apply the index on.
    """

    columns: cabc.Sequence[str]
    """
    The columns on which the index works. The order of the indices matter.
    """

    def __str__(self):
        col_args = ", ".join(self.columns)
        return f"{self.frame}({col_args})"


@dcls.dataclass(frozen=True, eq=False)
class Index(ABC):
    """
    `Index` corresponds to different types of backends, e.g. `faiss`, `b-tree`,
    and is responsible for routing to different `Index` types.

    Note:
        `Index` currently is static. i.e. it does not support updating.
        This is fine so long as most other constructs in the project
        are designed to be immutable, favoring creation over mutation.
    """

    ctx: IndexContext
    """
    The context of the `Index`.
    """

    def __call__(self, op: IndexPlan, value: npt.ArrayLike) -> npt.NDArray:
        arr = np.array(value)
        _, *dims = arr.shape

        if self.dims != tuple(dims):
            raise ValueError(
                f"Index {self} only supports query of shape: {self.dims}. Got {arr.shape}."
            )

        return self.search(op, arr)

    @abc.abstractmethod
    def search(self, op: IndexPlan, value: npt.NDArray, /) -> npt.NDArray: ...

    @property
    @abc.abstractmethod
    def dims(self) -> tuple[int, ...]:
        """
        The shape of the input query.
        Empty tuple denotes scalars.

        Note:
            I have been thinking whether or not handling multiple dimensions make sense,
            but decided against it, because that makes the design much more complicated,
            and not too much to gain (perhaps able to broadcast but that's all.)
        """

        raise NotImplementedError
