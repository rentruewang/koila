# Copyright (c) AIoWay Authors - All Rights Reserved

"""
This is the kernels library for ``aioway``,
based on the same idea of my other library, ``koila``.

It's different from ``koila`` in the way that it doesn't just operate on ``Tensor``s,
and does not keep compatibility with ``torch`` API.

This means it's possible to develop "kernels" for any ``Module``,
as well as ``Tensor``s (for example, this makes ``BatchNorm`` trivial).
"""

from .arrays import *
from .kernels import *
