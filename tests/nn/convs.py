# Copyright (c) AIoWay Authors - All Rights Reserved

"Convolution layers."

from torch.nn import Conv1d as _Conv1d, Conv2d as _Conv2d, Conv3d as _Conv3d

import functools


class Conv1d:
    def __init__(self, )
    @functools.cached_property
    def _model(self):
        return _Conv1d(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            kernel_size=self._kernel_size,
            padding=self._padding,
        )
