# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import functools
import math
import typing
from abc import ABC
from typing import ClassVar

from tensordict import TensorDict

from aioway._errors import AiowayError

from .frames import Frame

__all__ = ["BatchFrame"]


@dcls.dataclass(frozen=True)
class BatchFrame[S](Frame, ABC):
    """
    A ``Frame`` backed by a ``TensorDict`` (aka a batch in ``aioway``).
    This means that it is non-distributed, and volatile.
    """

    KLASS: ClassVar[type]
    """
    The class variable to ensure that ``source`` is the correct type.
    """

    source: S
    """
    The underlying data of the ``Frame``.
    """

    batch: int
    """
    The batch size to use.
    """

    drop_last: bool = False
    """
    Whether to truncate the last batch that doesn't have length ``batch_size``.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.source, self.KLASS):
            raise TensorFrameTypeError(
                f"Expected {type(self.source)=} to be {self.KLASS}."
            )

        if not isinstance(self.data, TensorDict):
            raise TensorFrameTypeError(
                f"Expected {type(self.data)=} to be `TensorDict`."
            )

    @typing.override
    def __len__(self) -> int:
        truncate = math.ceil if self.drop_last else math.floor
        return truncate(len(self.data) / self.batch)

    @typing.override
    def __getitem__(self, idx: int) -> TensorDict:
        start = idx * self.batch
        end = min(start + self.batch, len(self.data))
        return self.data[start:end]

    @functools.cached_property
    def data(self) -> TensorDict:
        """
        This is ``self.data`` converted to ``TensorDict``.
        """

        return self.convert_tensordict(self.source)

    @property
    def device(self):
        return self.data.device

    @classmethod
    @abc.abstractmethod
    def convert_tensordict(cls, data: S) -> TensorDict: ...


class TensorFrameTypeError(AiowayError, TypeError): ...


class BatchListTensorFrameTypeError(AiowayError, TypeError): ...
