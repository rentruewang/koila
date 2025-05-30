# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Mapping, Sequence
from typing import Self

from tensordict import TensorDict
from torch import Tensor

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError

from .frames import Frame

__all__ = ["BlockFrame", "TensorDictFrame"]


@dcls.dataclass(frozen=True)
class BlockFrame(Frame):
    """
    A ``Frame`` backed by a ``Block``.
    This means that it is non-distributed, and volatile.
    """

    block: Block
    """
    The underlying data of the ``Frame``.
    """

    @typing.override
    def __len__(self):
        return len(self.block)

    def __getitem(self, idx):
        return self.block.data[idx]

    __getitem__ = __getitems__ = __getitem

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.block.attrs

    @property
    def device(self):
        return self.block.device


@dcls.dataclass(frozen=True)
class TensorDictFrame(Frame):
    """
    A ``Frame`` backed by a ``TensorDict``.
    This means that it is non-distributed, and volatile.
    """

    td: TensorDict
    """
    The underlying data of the ``Frame``.
    """

    attributes: AttrSet
    """
    The attribut sets for the current ``TensorDict`.
    """

    @typing.override
    def __len__(self):
        return len(self.td)

    def __getitem(self, idx):
        return self.td[idx]

    __getitem__ = __getitems__ = __getitem

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.attributes

    @property
    def device(self):
        return self.td.device

    @classmethod
    def from_dict(
        cls,
        dicts: Mapping[str, Tensor],
        /,
        *,
        batch_size: Sequence[int] | None = None,
        device: str = "cpu",
    ) -> Self:
        """
        Create a ``TensorDictFrame`` from a ``TensorDict``.

        The result ``TensorDict`` is always assumed to have a batch dimension of 1.
        """

        # Check if dicts is a dict, and if the k-v pairs are valid.
        if not isinstance(dicts, Mapping):
            raise TensorDictNotDictError(
                f"Expected a dict, got {type(dicts)=} instead."
            )

        if not all(
            isinstance(k, str) and isinstance(v, Tensor) for k, v in dicts.items()
        ):
            raise TensorDictWrongAttrsError(
                f"Expected a dict of str to Tensor, got {dicts=}"
            )

        # If the dicts is a TensorDict, we need to check if the batch size is correct.
        if isinstance(dicts, TensorDict):
            td = dicts.to(device=device)
        else:
            td = TensorDict(dicts, device=device, batch_size=batch_size)

        # Set the batch size to 1.
        td = td.auto_batch_size_(batch_dims=1)

        # Check if the batch size is correct.
        attrs = AttrSet.parse_tensor_dict(dicts)

        return cls(td=td, attributes=attrs)


class TensorDictWrongAttrsError(AiowayError, ValueError): ...


class TensorDictNotDictError(AiowayError, TypeError): ...
