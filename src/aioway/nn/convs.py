# Copyright (c) AIoWay Authors - All Rights Reserved

"Convolution layers."

import functools

from torch.nn import Conv1d as _Conv1d


class Conv1d:
    def __init__(self, in_channels: int, out_channels: int, kernel_size, padding):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._padding = padding

    @functools.cached_property
    def _model(self):
        return _Conv1d(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            kernel_size=self._kernel_size,
            padding=self._padding,
        )
