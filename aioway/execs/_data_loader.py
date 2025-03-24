# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Callable, Iterator
from typing import Any, Self

import tensordict
from tensordict import TensorDict
from torch.utils.data import DataLoader

from aioway.blocks import Block
from aioway.errors import AiowayError

if typing.TYPE_CHECKING:
    from aioway.frames import Frame
    from aioway.streams import Stream

__all__ = ["DataLoaderAdaptor", "DataLoaderAdaptorLike"]


def maybe_stack_tensordict(items: TensorDict | list[TensorDict]) -> TensorDict:
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


def _check_tensordict_batched(td: TensorDict, /, *, is_batched: bool) -> None:
    if bool(td.batch_dims) == is_batched:
        return

    raise TabularBatchError


@typing.final
@dcls.dataclass(frozen=True)
class DataLoaderAdaptor:
    batch_size: int = 64
    num_workers: int = 1
    pin_memory: bool = False
    drop_last: bool = False
    in_order: bool = True
    collate_fn: Callable = maybe_stack_tensordict

    def iterator_of(self, dataset: "Frame | Stream") -> Iterator[Block]:
        return _dl_iter(dataset, self)

    @classmethod
    def parse(cls, opt: "DataLoaderAdaptorLike") -> Self:
        if isinstance(opt, cls):
            return opt

        if isinstance(opt, dict):
            return cls(**opt)

        raise DataLoaderOptTypeError(f"Option: {type(opt)=} is not supported.")


type DataLoaderAdaptorLike = DataLoaderAdaptor | dict[str, Any]


def _dl_iter(
    dataset: "Frame | Stream", opt: DataLoaderAdaptor | dict[str, Any]
) -> Iterator[Block]:
    # Convert to ``DataLoaderOpt`` first to ensure that the default configs are set.
    opt = DataLoaderAdaptor.parse(opt)

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


class TabularBatchError(AiowayError, ValueError): ...


class DataLoaderOptTypeError(AiowayError, TypeError): ...
