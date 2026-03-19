# Copyright (c) AIoWay Authors - All Rights Reserved

"The dense layers from `torch.nn`."

import functools

from torch.nn import Linear as _Linear

from aioway import _logging

__all__ = ["Linear"]

LOGGER = _logging.get_logger(__name__)


class Linear:
    """
    The wrapper for `torch.nn.Linear`.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        LOGGER.info(
            "Constructing `Linear(in_features=%s, out_features=%s, bias=%s)`",
            in_features,
            out_features,
            bias,
        )

        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias

    @functools.cached_property
    def _model(self):
        return _Linear(
            in_features=self._in_features,
            out_features=self._out_features,
            bias=self._bias,
        )

    @property
    def in_attrs(self):
        raise NotImplementedError

    @property
    def out_attrs(self):
        raise NotImplementedError
