# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import functools
import typing
from collections.abc import Callable, Iterator
from typing import Any, Self

import tensordict
from tensordict import TensorDict
from torch.utils.data import DataLoader

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError

from .nullary import NullaryExec

if typing.TYPE_CHECKING:
    from aioway.frames import Frame

__all__ = ["FrameExec", "DataLoaderCfg", "DataLoaderCfgLike"]


@typing.final
@dcls.dataclass
class FrameExec(NullaryExec, key="FRAME"):
    """
    An ``Exec`` that wraps a ``Frame`` and a ``DataLoaderAdaptor``.
    """

    dataset: "Frame" = dcls.field(repr=False)
    """
    The backing ``Frame``, stored in order to reset.
    """

    opt: "DataLoaderCfg | dict[str, Any]" = dcls.field(default_factory=dict)
    """
    The option for the ``DataLoaderAdaptor``,
    which is responsible for iterating over the ``Frame``.
    """

    def __post_init__(self) -> None:
        self.reset()

    @typing.override
    def __next__(self) -> Block:
        item = next(self._iterator)
        assert isinstance(item, Block), f"Item must be a `Block`, got {type(item)=}."
        return item

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Block:
        return self.dataset[index]

    def __getitems__(self, indices: list[int]) -> Block:
        return self.dataset.__getitems__(indices)

    def reset(self) -> None:
        if hasattr(self, "_iterator"):
            del self._iterator

    @functools.cached_property
    def _iterator(self) -> Iterator[Block]:
        """
        The actual iterator that will be used to iterate over the ``Frame``.
        """

        return self._new_iterator()

    def _new_iterator(self):
        yield from DataLoaderCfg.parse(self.opt).iterator_of(self.dataset)

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.dataset.attrs


type DataLoaderCfgLike = DataLoaderCfg | dict[str, Any]


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

    return _maybe_stack_td_impl(items)


@typing.final
@dcls.dataclass(frozen=True)
class DataLoaderCfg:
    """
    Configuration for the ``DataLoader``.
    """

    batch_size: int = 64
    """
    Batch size for the dataloader.
    """

    num_workers: int = 1
    """
    Number of worker processes for data loading.
    This config is directly passed to ``torch.utils.data.DataLoader``,
    and the default value 1 means to use a separate process for loading data.
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

    collate_fn: Callable = _maybe_stack_td
    """
    Collate function for the dataloader.

    Note:
        Function `_maybe_stack_td` cannot be defined later,
        because it cannot be a `lambda` if we want to serialize it,
        due to `DataLoader` using `pickle` to serialize the function.
    """

    def iterator_of(self, dataset: "Frame") -> Iterator[Block]:
        return _dl_iter(dataset, self)

    @classmethod
    def parse(cls, opt: "DataLoaderCfgLike") -> Self:
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


def _dl_iter(dataset: "Frame", opt: DataLoaderCfg | dict[str, Any]) -> Iterator[Block]:
    # Convert to ``DataLoaderOpt`` first to ensure that the default configs are set.
    opt = DataLoaderCfg.parse(opt)

    loader = DataLoader(dataset, **dcls.asdict(opt))

    # Convert batch to ``Block`` and check for ``attrs``.
    for batch in loader:
        if not isinstance(batch, TensorDict):
            raise TabularBatchError(
                f"A batch should be of type ``TensorDict``, got {type(batch)=}."
            )

        if not batch.batch_size:
            raise TabularBatchError(f"TensorDict batch must have `batch_size`.")

        block = Block(batch)
        block.require_attrs(dataset.attrs)
        yield block


def _maybe_stack_td_impl(items: TensorDict | list[TensorDict]) -> TensorDict:

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
