# Copyright (c) AIoWay Authors - All Rights Reserved

"The dense layers from `torch.nn`."

import dataclasses as dcls
import typing

from torch.nn import Linear as _Linear

from aioway._tracking import logging
from aioway.attrs import Attr

from .previews import Preview

__all__ = ["Linear"]

LOGGER = logging.get_logger(__name__)


@dcls.dataclass(frozen=True)
class Linear(Preview):
    """
    The wrapper for `torch.nn.Linear`.
    """

    MODULE_TYPE = _Linear

    in_features: int
    out_features: int
    bias: bool = True

    @typing.override
    def _preview(self, attr: Attr) -> Attr:
        shape = [*attr.shape[:-1], self.out_features]
        return Attr.parse(
            device=attr.device,
            dtype=(attr.dtype.term * "float32").unpack(),
            shape=shape,
        )
