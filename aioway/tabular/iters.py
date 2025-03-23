# Copyright (c) RenChu Wang - All Rights Reserved

from collections.abc import Iterator

import tensordict
from tensordict import TensorDict
from torch.utils.data import DataLoader

from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.execs import Exec, IteratorExec

from .frames import Frame
from .streams import Stream


def tabular_iterator(dataset: Frame | Stream, **kwargs) -> Exec:
    return IteratorExec(_tabular_iterator(dataset, **kwargs), dataset.attrs)


def _tabular_iterator(dataset: Frame | Stream, **kwargs) -> Iterator[Block]:
    loader = DataLoader(
        dataset,
        **kwargs,
        collate_fn=stack_with_checks,
    )

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


def stack_with_checks(items: TensorDict | list[TensorDict]) -> TensorDict:
    if isinstance(items, TensorDict):
        _check_tensordict_batched(items, is_batched=True)
        return items

    if isinstance(items, list):
        for item in items:
            if not isinstance(item, TensorDict):
                raise TabularBatchError("All instances of")

            _check_tensordict_batched(item, is_batched=False)

        return tensordict.stack(items, dim=0)

    raise TabularBatchError(
        "Input must be a batched `TensorDict` or a list of unbatched `TensorDict`."
    )


def _check_tensordict_batched(td: TensorDict, /, *, is_batched: bool) -> None:
    if bool(td.batch_dims) != is_batched:
        raise TabularBatchError("`TensorDict` must be batched.")


class TabularBatchError(AiowayError, ValueError): ...
