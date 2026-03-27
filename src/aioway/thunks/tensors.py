# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import functools
from abc import ABC

from torch import Tensor
from torch._subclasses import FakeTensor

from aioway import fake
from aioway.attrs import Attr


class TensorThunk(ABC):
    @functools.cached_property
    def preview(self) -> FakeTensor:
        preview = self._preview()
        if not fake.is_fake_tensor(preview):
            raise AssertionError("Does not return a fake tensor")
        return Attr.from_tensor(preview)

    @abc.abstractmethod
    def _preview(self) -> FakeTensor: ...

    @abc.abstractmethod
    def compute(self) -> Tensor: ...
