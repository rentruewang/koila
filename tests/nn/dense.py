# Copyright (c) AIoWay Authors - All Rights Reserved

"The dense layers from `torch.nn`."

import functools

from torch import Tensor
from torch.nn import Linear as _Linear

from aioway import _logging
from aioway.attrs import Attr

__all__ = ["Linear"]

LOGGER = _logging.get_logger(__name__)


class Linear:
    """
    The wrapper for `torch.nn.Linear`.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: str = "cpu",
    ) -> None:
        LOGGER.info(
            "Constructing `Linear(in_features=%s, out_features=%s, bias=%s)`",
            in_features,
            out_features,
            bias,
        )

        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias

        self._device = device
        self._dtype = "float32"

    def forward(self, data: Tensor) -> Tensor:
        return self._model(data)

    def attr(self, input_attr: Attr) -> Attr:
        # Right now using `NotImplemented` to describe whether or not this is OK.
        if input_attr.device != self._device:
            return NotImplemented

        if input_attr.shape[-1] != self._in_features:
            return NotImplemented

        shape = [input_attr.shape[:-1], self._out_features]
        return Attr.parse(
            device=input_attr.device,
            dtype=(input_attr.dtype.term * self._dtype).unpack(),
            shape=shape,
        )

    @functools.cached_property
    def _model(self):
        return _Linear(
            in_features=self._in_features,
            out_features=self._out_features,
            bias=self._bias,
        )
