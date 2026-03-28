# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from abc import ABC
from collections.abc import Iterator
from typing import Any

from tensordict import TensorDict

from aioway import fake
from aioway.fn import Fn

__all__ = ["TensorDictFn"]


class TensorDictFn(Fn[TensorDict], ABC):
    def __init__(self) -> None:
        super().__init__()
        assert all(fake.is_fake_tensor(tensor) for tensor in self._fake_result.values())

    @abc.abstractmethod
    @typing.override
    def _deps(self) -> Iterator[Fn[Any]]:
        raise NotImplementedError
