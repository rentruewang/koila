# Copyright (c) AIoWay Authors - All Rights Reserved


import dataclasses as dcls

from aioway.attrs import DType, Shape
from aioway.attrs.shapes import ShapeLike

__all__ = ["Array"]


@dcls.dataclass(init=False)
class Array:
    """
    The data structure for storing the costs in each categories.

    Todo:
        Extend this to take care of different devices.
    """

    shape: Shape
    """
    The array's shape.
    """

    dtype: DType
    """
    The data type of the array, when evaluated.
    """

    cost: int
    """
    The computation power needed.
    For now, this is just the number of operations.
    """

    def __init__(self, shape: ShapeLike, dtype: DType, cost: int):
        self.shape = Shape.wrap(shape)
        self.dtype = dtype
        self.cost = cost

    @property
    def storage(self):
        total = 1

        for dim in self.shape:
            total *= dim

        return total * self.dtype.bits
