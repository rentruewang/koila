# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Callable, Iterator
from typing import Any, Self

import tensordict
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

from aioway.errors import AiowayError
from aioway.frames import Frame

from .ops import Op0

__all__ = ["FrameOp"]


@dcls.dataclass(frozen=True)
class FrameOp(Op0, key="FRAME"):
    """
    An ``Op`` that wraps a ``Frame`` and a ``DataLoader``.
    """

    dataset: "Frame" = dcls.field(repr=False)
    """
    The backing ``Frame``, stored in order to reset.
    """

    opt: "FrameDataLoaderCfgLike" = dcls.field(default_factory=dict)
    """
    The option for the ``DataLoaderAdaptor``,
    which is responsible for iterating over the ``Frame``.
    """

    def __hash__(self) -> int:
        """
        The hash function for ``Frame`` op.

        Note:
            As of now, using ``id`` s.t. it will work for cases,
            where only if the ``Frame`` is the same instance, for safety.
        """

        return id(self)

    @typing.override
    def stream(self):
        yield from FrameDataLoaderCfg.parse(self.opt).iterator_of(self.dataset)


type FrameDataLoaderCfgLike = FrameDataLoaderCfg | dict[str, Any]
"The parsable dataloader configuration."


@typing.final
@dcls.dataclass(frozen=True)
class FrameDataLoaderCfg:
    """
    Configuration for the ``DataLoader``.

    Note:
        The flag ``num_workers`` is removed despite being a ``DataLoader`` flag.
        Per #88, this is really slow when ``num_workers != 0``.
        After some profiling, seems like most of the time is stuck communicating,
        which makes sense as the dataset is already in-memory,
        and setting up additional multiprocessing queues is unecessary.
    """

    batch_size: int = 64
    """
    Batch size for the dataloader.
    """

    pin_memory: bool = False
    """
    Whether to pin memory for the dataloader.
    """

    drop_last: bool = False
    """
    Whether to drop the last incomplete batch.
    """

    in_order: bool = True
    """
    Whether to keep the order of the dataset when sampling.
    """

    collate_fn: Callable = dcls.field(default_factory=lambda: _maybe_stack_td)
    """
    Collate function for the dataloader.

    Note:
        Using a ``lambda`` + ``default_factory``
        because I do not want to define ``_maybe_stack_td`` first.

        This means that we would need special handling if we want to serialize it,
        due to ``DataLoader`` using ``pickle`` to serialize the function.
    """

    def iterator_of(self, dataset: Dataset[TensorDict]) -> Iterator[TensorDict]:
        return _dl_iter(dataset, self)

    @classmethod
    def parse(cls, opt: "FrameDataLoaderCfgLike") -> Self:
        """
        Parse the input option to a ``DataLoaderCfg``.
        """

        if isinstance(opt, cls):
            return opt

        if isinstance(opt, dict):
            return cls(**opt)

        raise DataLoaderOptTypeError(f"Option: {type(opt)=} is not supported.")


def _check_tensordict_batched(td: TensorDict, /, *, is_batched: bool) -> None:
    if bool(td.batch_dims) == is_batched:
        return

    raise TabularBatchError


def _dl_iter(dataset: Dataset[TensorDict], opt: FrameDataLoaderCfgLike):
    # Convert to ``DataLoaderOpt`` first to ensure that the default configs are set.
    opt = FrameDataLoaderCfg.parse(opt)

    loader = DataLoader(dataset, **dcls.asdict(opt))

    # Convert batch to ``Block`` and check for ``attrs``.
    for batch in loader:
        if not isinstance(batch, TensorDict):
            raise TabularBatchError(
                f"A batch should be of type ``TensorDict``, got {type(batch)=}."
            )

        if not batch.batch_size:
            raise TabularBatchError(f"TensorDict batch must have `batch_size`.")

        yield batch


def _maybe_stack_td(items: TensorDict | list[TensorDict]) -> TensorDict:
    """
    Just your normal ``tensordict.stack``, but skips if the input is a ``TensorDict``.

    The input would be of type ``TensorDict`` when ``__getitems__`` is implemented.

    Args:
        items: A ``TensorDict`` (batched) or a list of ``TensorDict``s (no batch).

    Raises:
        TabularBatchError:
            If the input is of ``TensorDict`` type, but is not batched.
            Or if the input is of ``list[TensorDict]`` type, but is batched.

    Returns:
        A collated ``TensorDict``.
    """

    if isinstance(items, TensorDict):
        _check_tensordict_batched(items, is_batched=True)
        return items

    if isinstance(items, list):
        for item in items:
            if not isinstance(item, TensorDict):
                raise TabularBatchError(
                    "All instances of the list should be a ``TensorDict``."
                )

            _check_tensordict_batched(item, is_batched=False)

        return tensordict.stack(items, dim=0)

    raise TabularBatchError(
        "Input must be a batched `TensorDict` or a list of unbatched `TensorDict`."
    )


class TabularBatchError(AiowayError, ValueError): ...


class DataLoaderOptTypeError(AiowayError, TypeError): ...
