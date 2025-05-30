# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aioway.errors import AiowayError
from aioway.execs import DataLoaderCfg, DataLoaderCfgLike, FrameExec
from aioway.frames import Frame

from .ops import IndexOp

__all__ = ["Index", "IndexContext"]


@dcls.dataclass(frozen=True)
class IndexContext:
    """
    The indexing information for looking upu an index.
    """

    frame: Frame
    """
    The ``Frame`` to apply the index on.
    """

    columns: Sequence[str]
    """
    The columns on which the index works. The order of the indices matter.
    """

    def __str__(self):
        col_args = ", ".join(self.columns)
        return f"{self.frame}({col_args})"


@dcls.dataclass(frozen=True, eq=False)
class Index(ABC):
    """
    ``Index`` corresponds to different types of backends, e.g. ``faiss``, ``b-tree``,
    and is responsible for routing to different ``Index`` types.

    Note:
        ``Index`` currently is static. i.e. it does not support updating.
        This is fine so long as most other constructs in the project
        are designed to be immutable, favoring creation over mutation.
    """

    ctx: IndexContext
    """
    The context of the ``Index``.
    """

    def __call__(self, op: IndexOp, value: ArrayLike) -> NDArray:
        arr = np.array(value)
        _, *dims = arr.shape

        if self.dims != tuple(dims):
            raise IndexDimsError(
                f"Index {self} only supports query of shape: {self.dims}. Got {arr.shape}."
            )

        return self.search(op, arr)

    @abc.abstractmethod
    def search(self, op: IndexOp, value: NDArray, /) -> NDArray: ...

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

        ...

    @staticmethod
    def load_frame(ctx: IndexContext, dl_opts: "DataLoaderCfgLike") -> NDArray:
        """
        Load the frame's contents into an numpy array.

        Args:
            ctx: The frame and columns to index.
            dl_opts: _description_. Defaults to DataLoaderOpt().

        Raises:
            IndexDimsError:
                This shouldn't happen because ``to_tensor`` should convert to the proper 2D ``Tensor``.

        Returns:
            A 2D numpy array.
        """

        dl_opts = DataLoaderCfg.parse(dl_opts)

        # Dims should be the flattened shapes, due to `to_tensor`'s logic.
        dims = sum(ctx.frame.attrs[col].shape.numel() for col in ctx.columns)
        values = np.concatenate(
            [
                block[list(ctx.columns)].to_tensor().cpu().numpy()
                for block in FrameExec(ctx.frame, dl_opts)
            ],
            axis=0,
        )

        if values.ndim != 2 or values.shape[1] != dims:
            raise IndexDimsError(f"Data of tensors must have {dims} dimensions.")

        return values


class IndexDimsError(AiowayError, ValueError): ...


class IndexInitSubclassError(AiowayError, TypeError): ...
